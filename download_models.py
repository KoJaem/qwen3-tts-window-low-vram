# download_models.py
"""
Download Qwen3-TTS models to local folders (Windows-friendly).
- Downloads:
  1) Qwen/Qwen3-TTS-12Hz-0.6B-Base
  2) Qwen/Qwen3-TTS-Tokenizer-12Hz
- Supports resume, local_dir, and optional HF token for gated/private repos.

Usage (CMD):
  (.venv) D:\qwen3-tts> python download_models.py

Optional:
  set HF_TOKEN=hf_xxx   (Windows CMD)
  python download_models.py --out-dir models --revision main
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def download_repo(repo_id: str, local_dir: Path, revision: str, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==> Downloading: {repo_id}")
    print(f"    to: {local_dir.resolve()}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # Windows에서 안전 (실제 파일 복사)
        resume_download=True,
        token=token,
    )
    print(f"    done: {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="models", help="Base output directory (default: models)")
    parser.add_argument("--revision", default="main", help="Model revision/branch (default: main)")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    revision = args.revision
    token = args.token

    print("Hugging Face download (snapshot_download)")
    print(f"- out_dir : {out_dir.resolve()}")
    print(f"- revision: {revision}")
    print(f"- token   : {'(set)' if token else '(not set)'}")

    try:
        download_repo(TOKENIZER_REPO, out_dir / "Qwen3-TTS-Tokenizer-12Hz", revision, token)
        download_repo(MODEL_REPO, out_dir / "Qwen3-TTS-12Hz-0.6B-Base", revision, token)
    except KeyboardInterrupt:
        print("\nInterrupted. You can rerun; it will resume.")
        sys.exit(130)

    print("\nAll downloads completed.")
    print("Result folders:")
    print(f" - { (out_dir / 'Qwen3-TTS-Tokenizer-12Hz').resolve() }")
    print(f" - { (out_dir / 'Qwen3-TTS-12Hz-0.6B-Base').resolve() }")


if __name__ == "__main__":
    main()
