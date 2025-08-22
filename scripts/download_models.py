# scripts/download_models.py
"""
Качает InsightFace bundles:
  - antelopev2 (SCRFD + glintr100 + прочие onnx)
  - buffalo_l

Пример:
  python -m scripts.download_models                    # antelopev2 в ./weights/models/antelopev2
  python -m scripts.download_models --model-name buffalo_l
"""

import argparse
import zipfile
from pathlib import Path
import sys
import urllib.request
import shutil

try:
    import tomllib  # Python 3.11+
except Exception as e:
    print("Python 3.11+ required for tomllib (or install 'tomli')", file=sys.stderr)
    raise

BUNDLE_URLS = {
    "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
    "buffalo_l":  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}

DEFAULTS = {
    "bundle":   {"name": "antelopev2"},
    "paths":    {"weights_dir": "weights"},
    "download": {"fix_nested": True, "overwrite": False, "timeout": 60},
}

def load_config(path: Path) -> dict:
    if not path.exists():
        return DEFAULTS
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    out = DEFAULTS | cfg
    out["bundle"]   = DEFAULTS["bundle"]   | cfg.get("bundle", {})
    out["paths"]    = DEFAULTS["paths"]    | cfg.get("paths", {})
    out["download"] = DEFAULTS["download"] | cfg.get("download", {})
    return out

def download_to(url: str, dst: Path, timeout: int = 60, overwrite: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        print(f"Found: {dst}")
        return
    tmp = dst.with_suffix(".part")
    print(f"Downloading {url} -> {dst}")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dst)

def ensure_unzip(zip_path: Path, target_dir: Path, fix_nested: bool = True):
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    if fix_nested:
        nested = target_dir / target_dir.name
        if nested.is_dir() and any(nested.iterdir()):
            print(f"Flatten nested structure: {nested} -> {target_dir}")
            for p in nested.iterdir():
                p.rename(target_dir / p.name)
            nested.rmdir()

def main():
    ap = argparse.ArgumentParser(description="Download InsightFace bundle to weights/models/<bundle>")
    ap.add_argument("--model-name", choices=BUNDLE_URLS.keys(), help="bundle: antelopev2 | buffalo_l")
    ap.add_argument("--weights-dir", help="root dir for models (default from config.toml)")
    ap.add_argument("--config", default="config.toml", help="path to config.toml")
    ap.add_argument("--overwrite", type=int, default=None, help="1=force re-download zip")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    bundle = args.model_name or cfg["bundle"]["name"]
    weights_dir = Path(args.weights_dir or cfg["paths"]["weights_dir"]).resolve()
    fix_nested = bool(cfg["download"]["fix_nested"])
    overwrite = bool(cfg["download"]["overwrite"] if args.overwrite is None else args.overwrite)
    timeout = int(cfg["download"]["timeout"])

    if bundle not in BUNDLE_URLS:
        raise SystemExit(f"Unknown bundle '{bundle}'. Allowed: {', '.join(BUNDLE_URLS)}")

    models_dir = weights_dir / "models" / bundle
    zip_path = weights_dir / "models" / f"{bundle}.zip"

    download_to(BUNDLE_URLS[bundle], zip_path, timeout=timeout, overwrite=overwrite)
    ensure_unzip(zip_path, models_dir, fix_nested=fix_nested)

    onnx_files = sorted(p.name for p in models_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"ONNX files not found in {models_dir}. Check archive contents.")
    print("✅ Models ready:\n  - " + "\n  - ".join(onnx_files))

    print("\nUse in code:")
    print(f"  from insightface.app import FaceAnalysis")
    print(f"  app = FaceAnalysis(name='{bundle}', root=r'{weights_dir}')")
    print(f"  app.prepare(ctx_id=-1, det_size=(960, 960))  # CPU example")

if __name__ == "__main__":
    main()
