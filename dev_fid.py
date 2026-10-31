"""FID against a holdout taken from train_split.txt."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import evaluate as official
from dataset import PokemonDataset

SPLIT_DIR = Path("data/pokemon-generation-one-22k")
NUM_CLASSES = 151
MAX_PARAMS = 100_000_000


class _AsValModule:

    def __init__(self, split_file):
        self.ds = PokemonDataset(SPLIT_DIR, SPLIT_DIR / split_file, return_category=False)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=64, shuffle=False, num_workers=2)


def export_split(split_file: str, out_dir) -> Path:
    out_dir = Path(out_dir)
    if not any(out_dir.glob("*.png")):
        official.prepare_reference_set(_AsValModule(split_file), out_dir)
    return out_dir


def dev_reference(results_dir="results") -> Path:
    return export_split("dev_holdout.txt", Path(results_dir) / "dev_reference")


def dev_fid_for_model(model, output_dir, results_dir="results", batch_size=64, device="cuda"):
    categories = torch.arange(NUM_CLASSES, dtype=torch.long).repeat_interleave(
        official.SAMPLES_PER_CATEGORY
    )
    return official.evaluate(
        model, dev_reference(results_dir), output_dir, categories,
        batch_size=batch_size, device=device,
    )


def main(a):
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.floor:
        floor_dir = export_split("dev_floor_ref.txt", Path(a.results_dir) / "dev_floor")
        score = official.compute_fid(floor_dir, dev_reference(a.results_dir),
                                     device=device, batch_size=a.batch_size)
        print(f"real-vs-real floor (dev_floor_ref vs dev_holdout): {score:.2f}")
        return

    from model import Model

    model = Model.load_checkpoint(a.model_checkpoint, evaluate_mode=a.evaluate_mode, device=device)
    n = model.count_parameters()
    print(f"parameters: {n:,}")
    assert n <= MAX_PARAMS, "model exceeds the 100M parameter limit"
    out = Path(a.results_dir) / "dev_eval" / a.evaluate_mode
    score = dev_fid_for_model(model, out, a.results_dir, a.batch_size, device)
    print(f"{a.evaluate_mode} dev FID (holdout from train split): {score:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint")
    ap.add_argument("--evaluate_mode", choices=("one_nfe", "few_nfe"))
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.floor and not (args.model_checkpoint and args.evaluate_mode):
        ap.error("give --model_checkpoint and --evaluate_mode, or use --floor")
    main(args)