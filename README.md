# 초기 설정 (Windows / Low-VRAM 환경)

본 프로젝트는 **Windows 환경에서 Qwen3 TTS를 로컬 모델 + low-VRAM 설정으로 실행**하기 위한 예제입니다.  
Python 가상환경(.venv)을 사용하며, 모델 파일은 Hugging Face에서 **사전 다운로드 후 오프라인 실행**을 권장합니다.

---

## 1. Python 설치 확인

Python 3.9 이상이 필요합니다.

```
python --version
```

또는

```
py --version
```

설치되어 있지 않다면 아래에서 설치하세요.  
https://www.python.org/downloads/windows/

⚠️ 설치 시 **“Add Python to PATH” 반드시 체크**

---

## 2. 프로젝트 디렉토리 이동

터미널(CMD / PowerShell)에서 프로젝트 루트로 이동합니다.

```
cd qwen3-tts-window-low-vram
```

(폴더 이름은 실제 프로젝트 이름에 맞게 수정)

---

## 3. Python 가상환경(.venv) 생성

프로젝트 루트에서 아래 명령어를 실행합니다.

```
python -m venv .venv
```

정상적으로 생성되면 디렉토리 구조는 다음과 같습니다.

```
qwen3-tts-window-low-vram/
 ├─ .venv/
 ├─ models/
 ├─ main.py
 ├─ download_models.py
 ├─ requirements.txt
 └─ README.md
```

---

## 4. 가상환경 활성화

사용 중인 터미널에 맞는 명령어를 실행합니다.

### PowerShell

```
.\.venv\Scripts\Activate.ps1
```

❗ 실행 정책 오류 발생 시 (최초 1회)

```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CMD

```
.venv\Scripts\activate
```

### Git Bash

```
source .venv/Scripts/activate
```

---

## 5. 가상환경 활성화 확인

프롬프트 앞에 `(.venv)` 가 표시되면 정상입니다.

```
(.venv) D:\qwen3-tts-window-low-vram>
```

또는 Python 경로 확인:

```
where python
```

→ `.venv\Scripts\python.exe` 가 출력되면 성공

---

## 6. pip 업데이트 (권장)

```
python -m pip install --upgrade pip
```

---

## 7. PyTorch 설치 (중요)

본 프로젝트는 PyTorch를 사용합니다.  
**환경에 따라 설치 방법이 다르므로 requirements.txt와 분리되어 있습니다.**

### CPU 전용 (CUDA 없음 / low-VRAM 안전)

```
pip install torch torchvision torchaudio
```

### NVIDIA GPU 사용 시 (예: CUDA 12.1)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA 버전 확인:

```
nvidia-smi
```

---

## 8. 기타 의존성 설치

```
pip install -r requirements.txt
```

⚠️ 반드시 `.venv` 가상환경이 활성화된 상태에서 실행해야 합니다.

requirements.txt에는 다음 라이브러리가 포함되어 있습니다.

- transformers (Qwen TTS / Whisper)
- qwen-tts
- librosa (STT용 오디오 로딩)
- soundfile (TTS 결과 저장)
- huggingface-hub
- click (CLI 유틸)

---


## 9. 모델 파일 다운로드 (로컬 실행 필수)

본 프로젝트는 `local_files_only=True` 설정으로 실행되므로  
**모델을 반드시 사전에 다운로드해야 합니다.**

```
python download_models.py --out-dir models --revision main
```

모델 다운로드 후 다음 파일들이 존재해야 합니다.

- models/Qwen3-TTS-12Hz-0.6B-Base/config.json
- models/Qwen3-TTS-12Hz-0.6B-Base/*.safetensors 또는 pytorch_model.bin
- models/Qwen3-TTS-Tokenizer-12Hz/*

토큰이 필요한 경우 (gated repo):

### CMD

```
set HF_TOKEN=hf_xxxxxxxxx
python download_models.py --out-dir models
```

### PowerShell

```
$env:HF_TOKEN="hf_xxxxxxxxx"
python download_models.py --out-dir models
```

---

## 10. 모델 실행

```
python main.py
```

정상 실행 시 `result.wav` 파일이 생성됩니다.

실행 시 사용할 Qwen3 TTS 모델을 선택합니다.

- 0.6B: low-VRAM 환경 권장 (GTX 1050Ti 등)
- 1.7B: 음질 우선, 더 많은 VRAM 필요

---

## 11. 자주 발생하는 오류

### torch 모듈을 찾을 수 없을 때
- PyTorch 미설치
- 가상환경 미활성화

→ PyTorch 설치 후 재실행

### 모델 파일이 없다는 오류
- local_files_only=True 상태에서 로컬 모델 없음
- config.json 누락

→ download_models.py 재실행

### flash-attn 경고
- Windows / low-VRAM 환경에서는 정상
- 무시해도 동작에 문제 없음

---

## 12. 가상환경 종료

```
deactivate
```
