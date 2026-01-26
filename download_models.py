# download_models.py
"""
Download models to local folders (Windows-friendly).

Downloads (selectable):
  Qwen3-TTS:
    1) Qwen/Qwen3-TTS-12Hz-0.6B-Base
    2) Qwen/Qwen3-TTS-12Hz-1.7B-Base
    3) Qwen/Qwen3-TTS-Tokenizer-12Hz (always)
  Whisper (optional):
    - openai/whisper-large-v3-turbo

Supports:
- resume download
- local_dir (no symlinks, Windows-safe)
- optional HF token for gated/private repos

Usage:
  (.venv) python download_models.py

Optional:
  set HF_TOKEN=hf_xxx
  python download_models.py --out-dir models --revision main
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


# Qwen3 TTS
MODEL_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# Whisper
WHISPER_REPO = "openai/whisper-large-v3-turbo"


def download_repo(repo_id: str, local_dir: Path, revision: str, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==> Downloading: {repo_id}")
    print(f"    to: {local_dir.resolve()}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # Windows-safe (real files)
        resume_download=True,
        token=token,
    )
    print(f"    done: {repo_id}")


def ask_qwen_choice() -> str:
    print("\nSelect Qwen3-TTS models to download:")
    print("  1) 0.6B only")
    print("  2) 1.7B only")
    print("  3) Both (0.6B + 1.7B)")
    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    """
    Returns True/False based on user input.
    default_yes=True: Enter => yes
    """
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
        if ans == "" and default_yes:
            return True
        if ans == "" and not default_yes:
            return False
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter y or n.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="models", help="Base output directory (default: models)")
    parser.add_argument("--revision", default="main", help="Model revision/branch (default: main)")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var).",
    )

    # Qwen selection
    parser.add_argument(
        "--qwen",
        choices=["0.6b", "1.7b", "both"],
        help="Skip prompt and select Qwen3-TTS model(s) directly",
    )

    # Whisper selection
    parser.add_argument(
        "--whisper",
        choices=["yes", "no"],
        help="Download whisper-large-v3-turbo (yes/no). If omitted, prompt.",
    )
    parser.add_argument(
        "--whisper-revision",
        default="main",
        help="Whisper revision/branch (default: main)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    qwen_revision = args.revision
    whisper_revision = args.whisper_revision
    token = args.token

    print("Hugging Face download (snapshot_download)")
    print(f"- out_dir          : {out_dir.resolve()}")
    print(f"- qwen revision    : {qwen_revision}")
    print(f"- whisper revision : {whisper_revision}")
    print(f"- token            : {'(set)' if token else '(not set)'}")

    # Decide Qwen models
    if args.qwen:
        qwen_choice = {"0.6b": "1", "1.7b": "2", "both": "3"}[args.qwen]
    else:
        qwen_choice = ask_qwen_choice()

    # Decide Whisper download
    if args.whisper == "yes":
        download_whisper = True
    elif args.whisper == "no":
        download_whisper = False
    else:
        download_whisper = ask_yes_no("Also download Whisper (openai/whisper-large-v3-turbo)?", default_yes=True)

    try:
        # Tokenizer is always required for Qwen3-TTS
        download_repo(
            TOKENIZER_REPO,
            out_dir / "Qwen3-TTS-Tokenizer-12Hz",
            qwen_revision,
            token,
        )

        if qwen_choice == "1":
            download_repo(
                MODEL_06B,
                out_dir / "Qwen3-TTS-12Hz-0.6B-Base",
                qwen_revision,
                token,
            )
        elif qwen_choice == "2":
            download_repo(
                MODEL_17B,
                out_dir / "Qwen3-TTS-12Hz-1.7B-Base",
                qwen_revision,
                token,
            )
        elif qwen_choice == "3":
            download_repo(
                MODEL_06B,
                out_dir / "Qwen3-TTS-12Hz-0.6B-Base",
                qwen_revision,
                token,
            )
            download_repo(
                MODEL_17B,
                out_dir / "Qwen3-TTS-12Hz-1.7B-Base",
                qwen_revision,
                token,
            )

        # Optional: Whisper
        if download_whisper:
            download_repo(
                WHISPER_REPO,
                out_dir / "whisper-large-v3-turbo",
                whisper_revision,
                token,  # Whisper는 보통 토큰 필요 없지만, 프록시/인증 환경 대비
            )

    except KeyboardInterrupt:
        print("\nInterrupted. You can rerun; it will resume.")
        sys.exit(130)

    print("\nAll downloads completed.")
    print("Result folders:")
    print(f" - { (out_dir / 'Qwen3-TTS-Tokenizer-12Hz').resolve() }")
    if qwen_choice in {"1", "3"}:
        print(f" - { (out_dir / 'Qwen3-TTS-12Hz-0.6B-Base').resolve() }")
    if qwen_choice in {"2", "3"}:
        print(f" - { (out_dir / 'Qwen3-TTS-12Hz-1.7B-Base').resolve() }")
    if download_whisper:
        print(f" - { (out_dir / 'whisper-large-v3-turbo').resolve() }")


if __name__ == "__main__":
    main()
