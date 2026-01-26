# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers>=5.0.0rc1",
#     "torch>=2.0.0",
#     "torchaudio>=2.0.0",
#     "librosa>=0.10.0",
#     "click",
# ]
# ///
"""
reference.wav 파일을 Whisper large-v3로 STT 변환하여 reference.txt에 저장하는 스크립트
로컬에 다운로드된 Whisper 모델을 사용합니다.
"""
import sys
from pathlib import Path

import click

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    import librosa
except ImportError:
    print("❌ 필요한 패키지가 설치되지 않았습니다.")
    print("다음 명령으로 설치하세요: pip install transformers torch torchaudio librosa")
    sys.exit(1)


def get_local_whisper_model_path() -> Path | None:
    """로컬 Whisper 모델 경로를 반환합니다."""
    script_dir = Path(__file__).parent.absolute()
    model_dir = script_dir / "models" / "whisper-large-v3-turbo"
    
    if not model_dir.exists():
        return None
    
    # config.json 파일이 있는지 확인
    config_file = model_dir / "config.json"
    if not config_file.exists():
        return None
    
    return model_dir


def load_whisper_model(model_path: str | Path | None = None, local_files_only: bool = False):
    """Whisper 모델과 프로세서를 로드합니다."""
    if model_path:
        click.echo(f"모델 로딩 중: {model_path}")
        processor = WhisperProcessor.from_pretrained(
            str(model_path), 
            local_files_only=local_files_only
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path), 
            local_files_only=local_files_only,
            torch_dtype=torch.float32
        )
    else:
        click.echo("온라인 모델 로딩 중...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float32
        )
    
    model.eval()  # 평가 모드로 설정
    return processor, model


def transcribe_with_model(processor, model, audio_path: Path, language: str = "ko") -> str:
    """로드된 모델을 사용하여 오디오를 텍스트로 변환합니다."""
    # 오디오 파일 로드
    click.echo(f"오디오 파일 변환 중: {audio_path}")
    audio, sr = librosa.load(str(audio_path), sr=16000)
    
    # STT 변환
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    input_features = input_features.to(torch.float32)
    
    # 언어로 명시적으로 설정
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            language=language
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription.strip()


def transcribe_audio(audio_path: Path, use_local: bool = True, language: str = "ko") -> str:
    """오디오 파일을 텍스트로 변환합니다."""
    try:
        # 로컬 모델 경로 확인
        if use_local:
            local_model_path = get_local_whisper_model_path()
            if local_model_path:
                click.echo(f"로컬 Whisper 모델 사용: {local_model_path}")
                
                # transformers를 사용해서 로컬 모델 로드
                try:
                    click.echo("transformers로 로컬 모델 로딩 중...")
                    processor, model = load_whisper_model(local_model_path, local_files_only=True)
                    return transcribe_with_model(processor, model, audio_path, language)
                    
                except Exception as e:
                    click.echo(f"로컬 모델 로드 실패: {e}")
                    click.echo("온라인에서 모델을 다운로드합니다...")
                    # 폴백: 온라인 모델 사용
                    processor, model = load_whisper_model()
                    return transcribe_with_model(processor, model, audio_path, language)
            else:
                click.echo("로컬 모델을 찾을 수 없습니다. 온라인에서 다운로드합니다.")
                processor, model = load_whisper_model()
                return transcribe_with_model(processor, model, audio_path, language)
        else:
            click.echo("온라인 모델 사용")
            processor, model = load_whisper_model()
            return transcribe_with_model(processor, model, audio_path, language)
        
    except Exception as e:
        raise click.ClickException(f"STT 변환 실패: {e}")


@click.command()
@click.option("-a", "--audio", "audio_file", default="reference.wav", help="입력 오디오 파일 (기본값: reference.wav)")
@click.option("-o", "--output", "output_file", default="reference.txt", help="출력 텍스트 파일 (기본값: reference.txt)")
@click.option("-l", "--language", default="ko", help="언어 코드 (기본값: ko, 한국어)")
@click.option("--use-local", is_flag=True, default=True, help="로컬 모델 사용 (기본값: True)")
@click.option("-v", "--verbose", is_flag=True, help="상세 출력 활성화")
def main(audio_file: str, output_file: str, language: str, use_local: bool, verbose: bool):
    """reference.wav 파일을 STT로 변환하여 reference.txt에 저장합니다."""
    
    click.echo("🚀 STT 변환 스크립트 시작...")
    
    # 현재 스크립트 위치 기준으로 파일 경로 설정
    script_dir = Path(__file__).parent.absolute()
    audio_path = script_dir / audio_file
    output_path = script_dir / output_file
    
    # 오디오 파일 확인
    click.echo(f"📁 오디오 파일 확인 중: {audio_path}")
    if not audio_path.exists():
        raise click.ClickException(
            f"오디오 파일을 찾을 수 없습니다: {audio_path}\n"
            f"동일 경로에 {audio_file} 파일을 준비해주세요."
        )
    click.echo("✅ 오디오 파일 확인 완료")
    
    if verbose:
        click.echo(f"입력 오디오: {audio_path}")
        click.echo(f"출력 파일: {output_path}")
        click.echo(f"언어: {language}")
    
    # STT 변환
    click.echo(f"\n🎤 STT 변환 시작... (언어: {language})")
    try:
        text = transcribe_audio(audio_path, use_local=use_local, language=language)
    except click.ClickException as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    
    # 변환된 텍스트 출력
    click.echo("\n" + "="*60)
    click.echo("변환된 텍스트:")
    click.echo("="*60)
    click.echo(text)
    click.echo("="*60)
    
    # 파일로 저장
    try:
        output_path.write_text(text, encoding="utf-8")
        click.echo(f"\n✅ 텍스트가 저장되었습니다: {output_path}")
    except Exception as e:
        raise click.ClickException(f"파일 저장 실패: {e}")


if __name__ == "__main__":
    main()