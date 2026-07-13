#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parent

REPO_MAP = {
    "cola": {
        "url": "https://github.com/alvin-zyl/CoLA",
        "dir": "CoLA",
    },
    "galore": {
        "url": "https://github.com/jiaweizzhao/GaLore",
        "dir": "GaLore",
    },
    "relora": {
        "url": "https://github.com/Guitaricet/relora",
        "dir": "ReLoRA",
    },
    "switchlora": {
        "url": "https://github.com/oddForPapergweiowio/SwitchLoRA",
        "dir": "SwitchLoRA",
    },
    "fira": {
        "url": "https://github.com/xichen-fy/Fira",
        "dir": "Fira",
    },
    "sltrain": {
        "url": "https://github.com/andyjm3/SLTrain",
        "dir": "SLTrain",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone method repositories into training/")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(REPO_MAP.keys()),
        choices=list(REPO_MAP.keys()),
        help="Method repositories to clone",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Re-clone repo if destination already exists")
    return parser.parse_args()


def clone_repo(method: str, depth: int, force: bool) -> int:
    repo = REPO_MAP[method]
    dest = TRAINING_ROOT / repo["dir"]

    if dest.exists() and not force:
        print(f"[SKIP] {method}: {dest} already exists")
        return 0

    if dest.exists() and force:
        subprocess.run(["rm", "-rf", str(dest)], check=False)

    cmd = ["git", "clone", "--depth", str(depth), repo["url"], str(dest)]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        print(f"[OK]   {method}: cloned to {dest}")
    else:
        print(f"[FAIL] {method}: clone failed")
    return proc.returncode


def main() -> int:
    args = parse_args()
    codes = [clone_repo(m, args.depth, args.force) for m in args.methods]
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
