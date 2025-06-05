# Talkware

Talkware는 고성능 대화형 AI 모델을 기반으로 한 챗봇 서비스입니다.

## 📋 목차

- [기능](#-기능)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [설정](#-설정)
- [개발](#-개발)
- [라이선스](#-라이선스)

## ✨ 기능

- 다양한 LLM 모델 지원 (Llama3, Llama4 등)
- 비동기 웹 서버 기반 API 제공
- 실시간 대화 처리
- 상세한 로깅 시스템
- 환경 변수 기반 설정 관리

## 💻 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 최소 16GB RAM
- Linux 운영체제

## 🚀 설치 방법

1. 저장소 클론:
```bash
git clone [repository-url]
cd talkware
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 🎮 사용 방법

1. 설정 파일 준비:
   - `config/application.yml` 파일을 환경에 맞게 수정
   - 필요한 환경 변수 설정

2. 서버 실행:
```bash
python starter.py
```

3. API 엔드포인트:
   - 기본 URL: `http://localhost:8000`
   - API 문서: `http://localhost:8000/docs`

## 📁 프로젝트 구조

```
talkware/
├── bin/            # 실행 스크립트
├── config/         # 설정 파일
├── controller/     # API 컨트롤러
├── dto/           # 데이터 전송 객체
├── logs/          # 로그 파일
├── run/           # 실행 관련 파일
├── service/       # 비즈니스 로직
├── tests/         # 테스트 코드
├── starter.py     # 메인 실행 파일
└── requirements.txt # 의존성 목록
```

## ⚙️ 설정

### 주요 설정 파일
- `config/application.yml`: 메인 설정 파일
- `config/logging.yml`: 로깅 설정
- `.hf_token`: Hugging Face 토큰

### 환경 변수
- `APP_ENV`: 실행 환경 (development/production)
- `MODEL_PATH`: 모델 파일 경로
- `SERVER_HOST`: 서버 호스트
- `SERVER_PORT`: 서버 포트

## 👨‍💻 개발

### 테스트 실행
```bash
python -m pytest tests/
```

### 코드 스타일
- PEP 8 준수
- Type hints 사용
- Docstring 작성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 