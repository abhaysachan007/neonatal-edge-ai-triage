"""
Download OpenPOCUS Dataset
===========================
OpenPOCUS adult lung ultrasound data — publicly available on Zenodo (CC-BY 4.0).

Manual download (preferred):
  https://zenodo.org/records/7842167

After download, place frames as:
  data/raw/normal/     <- normal LUS frames
  data/raw/abnormal/   <- abnormal LUS frames

This script attempts auto-download via the Zenodo API.
If network/Zenodo is unavailable, it prints the manual URL and exits cleanly.

Usage:
  python scripts/download_openpocus.py --output data/raw
"""

import argparse
import sys
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "7842167"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
MANUAL_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"


def try_download(output_dir: Path) -> bool:
    try:
        import requests
    except ImportError:
        print("[WARN] requests not installed. pip install requests")
        return False

    print(f"Querying Zenodo record {ZENODO_RECORD_ID}...")
    try:
        resp = requests.get(ZENODO_API_URL, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Cannot reach Zenodo API: {e}")
        return False

    record = resp.json()
    files = record.get("files", [])
    if not files:
        print("[WARN] No files found in Zenodo record metadata.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        fname = f["key"]
        url = f["links"]["self"]
        dest = output_dir / fname
        if dest.exists():
            print(f"[SKIP] Already exists: {fname}")
            continue
        print(f"Downloading {fname} ({f.get('size', '?')} bytes)...")
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            print(f"  Saved: {dest}")
            if fname.endswith(".zip"):
                print(f"  Extracting {fname}...")
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(output_dir)
                print(f"  Extracted to {output_dir}")
        except Exception as e:
            print(f"[ERROR] Download failed for {fname}: {e}")
            return False

    return True


def check_existing(output_dir: Path) -> dict:
    counts = {}
    for cls in ["normal", "abnormal"]:
        cls_dir = output_dir / cls
        if cls_dir.exists():
            imgs = [p for p in cls_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            counts[cls] = len(imgs)
        else:
            counts[cls] = 0
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw", help="Destination for raw frames")
    args = parser.parse_args()

    output_dir = Path(args.output)

    existing = check_existing(output_dir)
    total = sum(existing.values())
    if total > 0:
        print(f"[OK] Data already present in {output_dir}:")
        for cls, n in existing.items():
            print(f"  {cls}: {n} images")
        return

    print("=" * 55)
    print("  OpenPOCUS Download Helper")
    print("=" * 55)
    print(f"Target: {output_dir.resolve()}")
    print()

    success = try_download(output_dir)

    if not success:
        print()
        print("=" * 55)
        print("  MANUAL DOWNLOAD REQUIRED")
        print("=" * 55)
        print(f"  URL: {MANUAL_URL}")
        print()
        print("  After downloading, organise as:")
        print(f"    {output_dir}/normal/     <- normal LUS .png files")
        print(f"    {output_dir}/abnormal/   <- abnormal LUS .png files")
        print()
        print("  Then re-run: python scripts/build_manifest.py")
        print("=" * 55)
        sys.exit(0)

    final = check_existing(output_dir)
    print("\nDownload complete.")
    for cls, n in final.items():
        print(f"  {cls}: {n} images")


if __name__ == "__main__":
    main()
