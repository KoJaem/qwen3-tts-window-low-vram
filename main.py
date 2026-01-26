from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949").strip()


def ask_model_choice() -> Path:
    print("\nSelect Qwen3 TTS model:")
    print("  1) 0.6B (low VRAM, recommended for GTX 1050Ti)")
    print("  2) 1.7B (higher quality, requires more VRAM)")
    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice == "1":
            return Path("models/Qwen3-TTS-12Hz-0.6B-Base")
        elif choice == "2":
            return Path("models/Qwen3-TTS-12Hz-1.7B-Base")
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    base_dir = Path(__file__).resolve().parent

    ref_audio = base_dir / "reference.wav"
    ref_text_path = base_dir / "reference.txt"
    tts_text_path = base_dir / "text.txt"

    for p in [ref_audio, ref_text_path, tts_text_path]:
        if not p.exists():
            raise FileNotFoundError(f"필수 파일이 없습니다: {p}")

    ref_text = read_text_file(ref_text_path)
    tts_text = read_text_file(tts_text_path)

    if not ref_text:
        raise ValueError("reference.txt가 비어있습니다.")
    if not tts_text:
        raise ValueError("text.txt가 비어있습니다.")

    # 🔽 모델 선택
    model_dir = (base_dir / ask_model_choice()).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"선택한 모델 폴더가 존재하지 않습니다: {model_dir}")

    use_cuda = torch.cuda.is_available()

    # ✅ 1050Ti 안정성: SDPA 커널을 math로 고정
    if use_cuda:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cuda:0" if use_cuda else "cpu",
        dtype=torch.float32,        # ✅ float16 금지 (1050Ti NaN 방지)
        low_cpu_mem_usage=True,
        local_files_only=True,      # ✅ 로컬 모델만 사용
    )

    with torch.inference_mode():
        try:
            wavs, sr = model.generate_voice_clone(
                text=tts_text,
                language="Korean",
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                do_sample=False,      # ✅ 안정성 우선
            )
        except TypeError:
            # qwen-tts 버전에 따라 do_sample 인자를 안 받을 수 있음
            wavs, sr = model.generate_voice_clone(
                text=tts_text,
                language="Korean",
                ref_audio=str(ref_audio),
                ref_text=ref_text,
            )

    out_path = base_dir / "result.wav"
    sf.write(str(out_path), wavs[0], sr, subtype="PCM_16")

    print(f"\nsaved -> {out_path}")
    print(f"sr={sr}, cuda={use_cuda}")
    print(f"model={model_dir.name}")
    print(f"ref_text chars: {len(ref_text)} / tts_text chars: {len(tts_text)}")


if __name__ == "__main__":
    main()
