"""
Scan a dataset directory for common issues before training.

Usage:
    python check_data.py --data-dir ./project_img
    python check_data.py --data-dir ./test
"""
import argparse
import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset(data_dir: Path) -> None:
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    if not class_dirs:
        log.error("No class subdirectories found in %s", data_dir)
        return

    print(f"\nDataset : {data_dir.resolve()}")
    print(f"Classes : {[d.name for d in class_dirs]}\n")

    counts: dict[str, int] = {}
    corrupt: list[Path] = []
    too_small: list[tuple[Path, int, int]] = []

    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        counts[cls_dir.name] = len(images)

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w < 32 or h < 32:
                        too_small.append((img_path, w, h))
            except (UnidentifiedImageError, OSError):
                corrupt.append(img_path)

    # --- class balance table ---
    total = sum(counts.values())
    max_count = max(counts.values(), default=1)
    print(f"{'Class':<20} {'Images':>7}  Distribution")
    print("-" * 55)
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * int(30 * n / max_count)
        print(f"{cls:<20} {n:>7}  {bar}")
    print(f"\n{'TOTAL':<20} {total:>7}")

    # --- warnings ---
    issues = 0

    if corrupt:
        issues += len(corrupt)
        print(f"\n[CORRUPT]  {len(corrupt)} unreadable images — delete these before training:")
        for p in corrupt[:10]:
            print(f"  {p}")
        if len(corrupt) > 10:
            print(f"  ... and {len(corrupt) - 10} more")

    if too_small:
        issues += len(too_small)
        print(f"\n[TOO SMALL]  {len(too_small)} images under 32x32 (model resizes to 64x64, these will be blurry):")
        for p, w, h in too_small[:5]:
            print(f"  {p}  ({w}x{h})")

    if counts:
        min_cls = min(counts, key=counts.get)
        max_cls = max(counts, key=counts.get)
        ratio = counts[max_cls] / max(counts[min_cls], 1)
        if ratio > 5:
            issues += 1
            print(
                f"\n[IMBALANCE]  '{max_cls}' has {counts[max_cls]} images but "
                f"'{min_cls}' has only {counts[min_cls]}  (ratio {ratio:.1f}x). "
                f"The model will be biased toward '{max_cls}'."
            )

    under_50 = [cls for cls, n in counts.items() if n < 50]
    if under_50:
        issues += 1
        print(f"\n[FEW SAMPLES]  These classes have fewer than 50 images: {under_50}")
        print("  Aim for at least 100-200 per class for reliable training.")

    if issues == 0:
        print("\nNo issues found — dataset looks good.")
    else:
        print(f"\n{issues} issue(s) found. Fix them before training.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Check a dataset folder for training issues")
    parser.add_argument("--data-dir", required=True, help="Root directory with one subfolder per class")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        log.error("Directory not found: %s", data_dir)
        return

    check_dataset(data_dir)


if __name__ == "__main__":
    main()
```
