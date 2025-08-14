from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging as hf_logging
from pathlib import Path
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. tr33hugg3r/knot-crossings")
    ap.add_argument("--revision", default="main", help="HF tag/branch/commit")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset","model","space"])
    ap.add_argument("--dest-root", default="data/raw")
    ap.add_argument("--max-workers", type=int, default=3)       # fewer threads = steadier on flaky nets
    ap.add_argument("--etag-timeout", type=float, default=120)  # seconds
    ap.add_argument("--allow", nargs="*", default=[
        "train/**/*.png","train/**/*.jpg","train/**/*.jpeg",
        "test/**/*.png","test/**/*.jpg","test/**/*.jpeg",
        "README.md",".gitattributes"
    ])
    args = ap.parse_args()

    hf_logging.set_verbosity_info()
    name = args.repo_id.split("/")[-1]
    local_root = Path(args.dest_root) / f"{name}@{args.revision}"
    local_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] syncing to {local_root} …")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=str(local_root),
        allow_patterns=args.allow,
        max_workers=args.max_workers,
        etag_timeout=args.etag_timeout,
        force_download=False,
    )

    (local_root / "_manifest.json").write_text(json.dumps({
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "local_root": str(local_root.resolve()),
        "allow_patterns": args.allow,
        "max_workers": args.max_workers,
        "etag_timeout": args.etag_timeout
    }, indent=2))
    print(f"[ok] Snapshot ready at {local_root}")

if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")
    # optional accelerator:
    # os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()
