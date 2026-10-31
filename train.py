"""FastGen training: teacher, teacher pairs, student distillation."""

import argparse
import contextlib
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import PokemonDataset, load_or_save_category_mapping
from model import DEFAULT_PRESET, PRESETS, ModelFewNFE, ModelOneNFE, build_backbone, time_grid
from src.utils import print_parameter_summary

NUM_CLASSES = 151
MAX_PARAMS = 100_000_000
MAX_VRAM_GIB = 19.5
SPLIT_DIR = Path("data/pokemon-generation-one-22k")
SPLIT_FILES = {"dev_train": "dev_train.txt", "train": "train_split.txt"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AMP:

    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = None
        self.scaler = None
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
        print(f"device={device} amp_dtype={self.dtype}")

    def autocast(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)

    def step(self, loss, opt, params, clip: float = 1.0):
        params = list(params)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, clip)
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(params, clip)
            opt.step()


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(
            list(self.model.parameters()), list(model.parameters()), 1.0 - d
        )


def load_backbone(net: nn.Module, path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {(k[len("backbone."):] if k.startswith("backbone.") else k): v for k, v in sd.items()}
    net.load_state_dict(sd)


def check_vram(device):
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        if peak > MAX_VRAM_GIB:
            raise RuntimeError(f"peak VRAM {peak:.1f} GiB exceeds the 20 GB course limit")
        return peak
    return 0.0


def load_train_tensors(device, split="dev_train"):
    cache = Path("data") / f"cache_{split}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        return d["x"].to(device), d["y"].to(device)
    split_file = SPLIT_DIR / SPLIT_FILES[split]
    cats = load_or_save_category_mapping(SPLIT_DIR, "train_split.txt", "val_split.txt")
    ds = PokemonDataset(SPLIT_DIR, split_file, return_category=True, category_to_id=cats)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, drop_last=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8))
        ys.append(y)
    x, y = torch.cat(xs), torch.cat(ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, cache)
    print(f"cached {len(x)} images from {split_file.name} -> {cache}")
    return x.to(device), y.to(device)


def save_grid(x: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(((x.float().clamp(-1, 1) + 1) / 2).cpu(), path, nrow=8)


def grid_classes(device):
    return torch.arange(0, NUM_CLASSES, 10, device=device)[:16]


@torch.no_grad()
def ode_sample(net, eps, y, steps: int, cfg: float, amp: AMP):
    x = eps
    B = x.shape[0]
    null = torch.full_like(y, NUM_CLASSES)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=x.device)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        tt = t.expand(B)
        with amp.autocast():
            if cfg != 1.0:
                v = net(torch.cat([x, x]), torch.cat([tt, tt]),
                        category=torch.cat([y, null])).float()
                v_c, v_u = v.chunk(2)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = net(x, tt, category=y).float()
        x = x + (t_next - t) * v
    return x


def lr_at(it, total, base_lr, warmup=200):
    warm = min(1.0, (it + 1) / warmup)
    cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(it, total) / max(total, 1)))
    return base_lr * warm * cos


def train_loop(a, stage, net, make_loss, device, amp, on_sample, on_final):
    ckpt_dir = Path(a.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.results_dir) / "logs" / f"{stage}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ema = EMA(net, a.ema)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    last = ckpt_dir / f"{stage}_last.pt"
    it = 0
    if a.resume and last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(st["net"])
        ema.model.load_state_dict(st["ema"])
        ema.n = st["ema_n"]
        opt.load_state_dict(st["opt"])
        it = st["it"]
        print(f"resumed {stage} from iteration {it}")
    print_parameter_summary(net)

    def save_last():
        torch.save({"net": net.state_dict(), "ema": ema.model.state_dict(),
                    "ema_n": ema.n, "opt": opt.state_dict(), "it": it}, last)

    session_t0 = time.time()
    log_t0 = session_t0
    run_loss, run_n = 0.0, 0
    net.train()
    while it < a.iters:
        for g in opt.param_groups:
            g["lr"] = lr_at(it, a.iters, a.lr)
        opt.zero_grad(set_to_none=True)
        with amp.autocast():
            loss = make_loss(net)
        amp.step(loss, opt, net.parameters())
        ema.update(net)
        it += 1
        run_loss += loss.item()
        run_n += 1

        if it % a.log_every == 0:
            avg = run_loss / run_n
            now = time.time()
            print(f"[{stage}] it {it}/{a.iters} loss {avg:.4f} "
                  f"{run_n / max(now - log_t0, 1e-6):.2f} it/s "
                  f"(this session {(now - session_t0) / 60:.1f} min) "
                  f"peak VRAM {check_vram(device):.1f} GiB")
            with open(log_path, "a") as f:
                f.write(f"{it},{avg:.6f},{now - session_t0:.1f}\n")
            run_loss, run_n = 0.0, 0
            log_t0 = now
        if it % a.sample_every == 0:
            on_sample(ema.model, it)
        if it % a.save_every == 0:
            save_last()
        if a.max_minutes and (time.time() - session_t0) > a.max_minutes * 60 and it < a.iters:
            print(f"time box reached at iteration {it}; saving and stopping")
            save_last()
            on_final(ema.model, it)
            return
    save_last()
    on_final(ema.model, it)


def cmd_bench(a):
    device = get_device()
    amp = AMP(device)
    net = build_backbone(a.preset).to(device)
    print(f"preset: {a.preset} {PRESETS[a.preset]}")
    print_parameter_summary(net)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.randn(a.bs, 3, 64, 64, device=device)
    y = torch.randint(0, NUM_CLASSES, (a.bs,), device=device)
    t = torch.rand(a.bs, device=device)

    def one():
        opt.zero_grad(set_to_none=True)
        with amp.autocast():
            loss = F.mse_loss(net(x, t, category=y).float(), x)
        amp.step(loss, opt, net.parameters())

    for _ in range(5):
        one()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(20):
        one()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    print(f"{dt:.3f} s/iter at bs={a.bs} -> {1 / dt:.2f} it/s, "
          f"{a.bs / dt:.0f} img/s")
    if device.type == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"1 epoch of dev_train (14059 imgs) ~ {14059 / (a.bs / dt) / 60:.1f} min")


def cmd_teacher(a):
    device = get_device()
    amp = AMP(device)
    X, Y = load_train_tensors(device, a.split)
    N = len(X)
    print(f"training on split={a.split} ({N} images), preset={a.preset}")
    net = build_backbone(a.preset).to(device)
    if a.init and not a.resume:
        load_backbone(net, a.init)
        print(f"initialised from {a.init}")
    assert sum(p.numel() for p in net.parameters()) <= MAX_PARAMS

    def make_loss(net):
        idx = torch.randint(0, N, (a.bs,), device=device)
        x0 = X[idx].float() / 127.5 - 1
        y = Y[idx]
        flip = torch.rand(a.bs, device=device) < 0.5
        x0 = torch.where(flip[:, None, None, None], x0.flip(3), x0)
        drop = torch.rand(a.bs, device=device) < a.p_uncond
        y_in = torch.where(drop, torch.full_like(y, NUM_CLASSES), y)
        eps = torch.randn_like(x0)
        if a.t_dist == "logitnormal":
            t = torch.sigmoid(torch.randn(a.bs, device=device))
        else:
            t = torch.rand(a.bs, device=device)
        tb = t.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * eps
        v = net(xt, t, category=y_in)
        return F.mse_loss(v.float(), eps - x0)

    cls = grid_classes(device)
    gen = torch.Generator(device=device).manual_seed(0)

    def on_sample(ema_net, it):
        gen.manual_seed(0)
        eps = torch.randn(len(cls), 3, 64, 64, device=device, generator=gen)
        x = ode_sample(ema_net, eps, cls, 24, a.cfg, amp)
        save_grid(x, Path(a.results_dir) / "samples" / f"teacher_{it:06d}.png")

    def on_final(ema_net, it):
        torch.save({"state_dict": ema_net.state_dict(), "preset": a.preset,
                    "split": a.split, "iters": it},
                   Path(a.ckpt_dir) / "teacher_ema.pt")
        print(f"saved {Path(a.ckpt_dir) / 'teacher_ema.pt'}")

    train_loop(a, "teacher", net, make_loss, device, amp, on_sample, on_final)


class TeacherSampler:

    def __init__(self, net, steps, cfg, amp):
        self.net, self.steps, self.cfg, self.amp = net, steps, cfg, amp

    def eval(self):
        return self

    @torch.no_grad()
    def sample(self, shape, *, device, category=None, **kwargs):
        eps = torch.randn(shape, device=device)
        return ode_sample(self.net, eps, category, self.steps, self.cfg, self.amp).clamp(-1, 1)


def load_teacher(a, device):
    ck = torch.load(Path(a.ckpt_dir) / "teacher_ema.pt", map_location="cpu", weights_only=False)
    net = build_backbone(ck.get("preset", a.preset)).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net


def cmd_teacher_fid(a):
    from dev_fid import dev_fid_for_model

    device = get_device()
    amp = AMP(device)
    net = load_teacher(a, device)
    out = Path(a.results_dir) / "dev_eval" / f"teacher_cfg{a.cfg}_steps{a.steps}"
    score = dev_fid_for_model(TeacherSampler(net, a.steps, a.cfg, amp), out,
                              a.results_dir, a.bs, device)
    print(f"teacher dev FID (holdout from train split; cfg={a.cfg}, steps={a.steps}): {score:.2f}")


def cmd_pairs(a):
    device = get_device()
    amp = AMP(device)
    net = load_teacher(a, device)
    out = Path(a.pairs_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_shard = a.bs * a.shard_batches
    total_batches = math.ceil(a.n / a.bs)
    done_shards = sorted(out.glob("shard_*.pt"))
    batch_idx = len(done_shards) * a.shard_batches
    g = torch.Generator(device=device)
    buf = {"eps": [], "x0": [], "y": []}
    s = len(done_shards)
    x0 = None
    t0 = time.time()
    if batch_idx >= total_batches:
        print(f"{len(done_shards)} shards already cover --n {a.n}; nothing to do")
        return
    while batch_idx < total_batches:
        g.manual_seed(a.seed * 100003 + batch_idx)
        y = (torch.arange(a.bs, device=device) + batch_idx * a.bs) % NUM_CLASSES
        eps = torch.randn(a.bs, 3, 64, 64, device=device, generator=g)
        x0 = ode_sample(net, eps, y, a.steps, a.cfg, amp).clamp(-1, 1)
        buf["eps"].append(eps.half().cpu())
        buf["x0"].append(x0.half().cpu())
        buf["y"].append(y.cpu())
        batch_idx += 1
        if len(buf["y"]) == a.shard_batches or batch_idx == total_batches:
            torch.save({k: torch.cat(v) for k, v in buf.items()}, out / f"shard_{s:04d}.pt")
            print(f"[pairs] shard {s} saved ({batch_idx * a.bs} pairs, "
                  f"{(time.time() - t0) / 60:.1f} min)")
            buf = {"eps": [], "x0": [], "y": []}
            s += 1
            if a.max_minutes and (time.time() - t0) > a.max_minutes * 60 and batch_idx < total_batches:
                print("time box reached; rerun the same command to continue")
                return
    save_grid(x0[:16], Path(a.results_dir) / "samples" / "pairs_preview.png")


def cmd_distill(a):
    device = get_device()
    amp = AMP(device)
    shards = sorted(Path(a.pairs_dir).glob("shard_*.pt"))
    assert shards, f"no pair shards in {a.pairs_dir}; run `train.py pairs` first"
    parts = [torch.load(s, map_location="cpu") for s in shards]
    EPS = torch.cat([p["eps"] for p in parts]).to(device)
    X0 = torch.cat([p["x0"] for p in parts]).to(device)
    Y = torch.cat([p["y"] for p in parts]).to(device)
    N = len(Y)
    print(f"{N} teacher pairs loaded")

    one = a.mode == "one"
    Cls = ModelOneNFE if one else ModelFewNFE
    out_name = "one_nfe.ckpt" if one else "few_nfe.ckpt"
    ck = torch.load(Path(a.ckpt_dir) / "teacher_ema.pt", map_location="cpu", weights_only=False)
    preset = ck.get("preset", a.preset)
    if preset != DEFAULT_PRESET:
        print(f"WARNING: set DEFAULT_PRESET = {preset!r} in model.py, otherwise "
              f"evaluate.py will build the wrong architecture for these checkpoints")
    student = Cls(device=device, preset=preset)
    assert student.count_parameters() <= MAX_PARAMS
    net = build_backbone(preset).to(device)
    net.load_state_dict(ck["state_dict"])
    times = torch.tensor([1.0] if one else time_grid(4)[:4], device=device)

    def make_loss(net):
        idx = torch.randint(0, N, (a.bs,), device=device)
        eps, x0, y = EPS[idx].float(), X0[idx].float(), Y[idx]
        t = times[torch.randint(len(times), (a.bs,), device=device)]
        tb = t.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * eps
        v = net(xt, t, category=y)
        x0_hat = xt - tb * v.float()
        return F.mse_loss(x0_hat, x0)

    cls = grid_classes(device)

    def sync(ema_net):
        student.backbone.load_state_dict(ema_net.state_dict())
        student.eval()

    def on_sample(ema_net, it):
        sync(ema_net)
        x = student.sample((len(cls), 3, 64, 64), device=device, category=cls)
        save_grid(x, Path(a.results_dir) / "samples" / f"{a.mode}_nfe_{it:06d}.png")

    def on_final(ema_net, it):
        sync(ema_net)
        path = Path(a.ckpt_dir) / out_name
        torch.save({"state_dict": student.state_dict()}, path)
        print(f"saved {path} ({student.count_parameters():,} parameters)")

    train_loop(a, f"distill_{a.mode}", net, make_loss, device, amp, on_sample, on_final)


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    common.add_argument("--split", choices=sorted(SPLIT_FILES), default="dev_train",
                        help="dev_train keeps dev_holdout unseen; use 'train' only for the final run")
    common.add_argument("--bs", type=int, default=64)
    common.add_argument("--lr", type=float, default=1e-4)
    common.add_argument("--iters", type=int, default=6000)
    common.add_argument("--ema", type=float, default=0.999)
    common.add_argument("--ckpt_dir", default="checkpoints")
    common.add_argument("--results_dir", default="results")
    common.add_argument("--pairs_dir", default="data/pairs")
    common.add_argument("--resume", action="store_true")
    common.add_argument("--max_minutes", type=float, default=0)
    common.add_argument("--log_every", type=int, default=100)
    common.add_argument("--sample_every", type=int, default=1000)
    common.add_argument("--save_every", type=int, default=500)
    common.add_argument("--seed", type=int, default=0)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bench", parents=[common]).set_defaults(fn=cmd_bench)

    t = sub.add_parser("teacher", parents=[common])
    t.add_argument("--init", default="", help="old Model checkpoint to warm start from")
    t.add_argument("--p_uncond", type=float, default=0.1)
    t.add_argument("--t_dist", choices=("uniform", "logitnormal"), default="uniform")
    t.add_argument("--cfg", type=float, default=1.5, help="guidance for preview grids")
    t.set_defaults(fn=cmd_teacher, ema=0.9995)

    f = sub.add_parser("teacher_fid", parents=[common])
    f.add_argument("--cfg", type=float, default=1.0)
    f.add_argument("--steps", type=int, default=24)
    f.set_defaults(fn=cmd_teacher_fid)

    q = sub.add_parser("pairs", parents=[common])
    q.add_argument("--n", type=int, default=20000)
    q.add_argument("--steps", type=int, default=24)
    q.add_argument("--cfg", type=float, default=1.5)
    q.add_argument("--shard_batches", type=int, default=32)
    q.set_defaults(fn=cmd_pairs)

    d = sub.add_parser("distill", parents=[common])
    d.add_argument("--mode", choices=("one", "few"), required=True)
    d.set_defaults(fn=cmd_distill)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    args.fn(args)