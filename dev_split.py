"""Create dev_train.txt, dev_holdout.txt and dev_floor_ref.txt from train_split.txt."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

PER_CATEGORY = 20


def category(path: str) -> str:
    return Path(path).parent.name


def main(split_dir: str, seed: int):
    split_dir = Path(split_dir)
    lines = [l.strip() for l in (split_dir / "train_split.txt").read_text().splitlines() if l.strip()]
    by_cat = defaultdict(list)
    for p in lines:
        by_cat[category(p)].append(p)

    rng = random.Random(seed)
    holdout, train, floor = [], [], []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat])
        assert len(items) > 2 * PER_CATEGORY, f"{cat} has only {len(items)} train images"
        rng.shuffle(items)
        holdout += items[:PER_CATEGORY]
        rest = items[PER_CATEGORY:]
        train += rest
        floor += rest[:PER_CATEGORY]

    for name, items in (("dev_holdout", holdout), ("dev_train", train), ("dev_floor_ref", floor)):
        (split_dir / f"{name}.txt").write_text("\n".join(sorted(items)) + "\n")

    sizes = sorted(len(v) for v in by_cat.values())
    print(f"categories: {len(by_cat)}  train images per category: min {sizes[0]}, max {sizes[-1]}")
    print(f"dev_holdout {len(holdout)}  dev_train {len(train)}  dev_floor_ref {len(floor)}")
    assert not set(holdout) & set(train)
    assert set(holdout) | set(train) == set(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="data/pokemon-generation-one-22k")
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))