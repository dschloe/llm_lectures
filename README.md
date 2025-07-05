# LLM Lectures - 대규모 언어 모델 강의 자료

이 저장소는 대규모 언어 모델(Large Language Models, LLMs)에 대한 실습 코드와 예제들을 포함하고 있습니다. 각 챕터는 특정 주제에 대한 실습과 이론적 배경을 제공합니다.

## 📚 챕터별 소개

### CH02: Transformer 아키텍처 기초
- **파일**: `chapter_2_transformer_with_code.ipynb`
- **주제**: Transformer 모델의 핵심 구성 요소들을 코드로 구현
- **내용**:
  - 토큰화 및 임베딩
  - 위치 인코딩 (Positional Encoding)
  - 어텐션 메커니즘 (Attention Mechanism)
  - 멀티헤드 어텐션 (Multi-Head Attention)
  - 층 정규화 (Layer Normalization)
  - 피드포워드 네트워크 (Feed-Forward Network)

### CH03: Hugging Face Transformers 라이브러리
- **파일**: `chapter_3.ipynb`
- **주제**: Hugging Face Transformers 라이브러리 사용법
- **내용**:
  - AutoModel과 AutoTokenizer 사용법
  - BERT, GPT-2 등 다양한 모델 활용
  - 토크나이저의 다양한 기능들
  - KLUE 데이터셋 활용
  - 한국어 모델 (KLUE/RoBERTa) 사용법

### CH05: 프롬프트 엔지니어링
- **파일**: `chapter_5.ipynb`
- **주제**: 효과적인 프롬프트 작성 기법
- **내용**:
  - 프롬프트 설계 원칙
  - Few-shot 학습
  - Chain-of-Thought 추론
  - 다양한 프롬프트 패턴

### CH06: 모델 평가 및 성능 측정
- **파일**: 
  - `chapter_6.ipynb`
  - `api_request_parallel_processor.py`
  - `utils.py`
- **주제**: LLM 모델의 성능 평가 방법
- **내용**:
  - SQL 생성 태스크를 통한 모델 평가
  - GPT-4를 활용한 자동 평가
  - 병렬 API 요청 처리
  - 한국어 Text-to-SQL 모델 성능 측정

### CH07: 모델 양자화 (Quantization)
- **파일**: `chapter_7.ipynb`
- **주제**: 모델 크기 최적화를 위한 양자화 기법
- **내용**:
  - BitsAndBytes를 활용한 8비트/4비트 양자화
  - GPTQ 양자화
  - AWQ 양자화
  - 양자화된 모델 로딩 및 사용법

### CH08: LLM 서빙 프레임워크
- **파일**: `chapter_8.ipynb`
- **주제**: 대규모 언어 모델 서빙을 위한 프레임워크
- **내용**:
  - vLLM을 활용한 고성능 추론
  - 배치 처리 최적화
  - API 서버 구축
  - 추론 성능 비교 분석

### CH09: 검색 증강 생성 (RAG)
- **파일**: `chapter_9.ipynb`
- **주제**: 검색과 생성이 결합된 RAG 시스템
- **내용**:
  - LlamaIndex를 활용한 RAG 구현
  - 벡터 검색과 LLM 생성 결합
  - LLM 캐싱 시스템
  - ChromaDB를 활용한 벡터 저장소

### CH10: 임베딩과 벡터 검색
- **파일**: `chapter_10.ipynb`
- **이미지**: `cat.jpg`, `dog.jpg`
- **주제**: 문장 임베딩과 의미 검색
- **내용**:
  - Sentence Transformers 활용
  - CLIP 모델을 통한 이미지-텍스트 임베딩
  - FAISS를 활용한 벡터 검색
  - 의미 기반 검색 시스템 구현

### CH11: 임베딩 모델 학습
- **파일**: `chapter_11.ipynb`
- **주제**: 문장 임베딩 모델 직접 학습
- **내용**:
  - KLUE STS 데이터셋 활용
  - Sentence Transformers 학습
  - 코사인 유사도 손실 함수
  - Hugging Face Hub에 모델 업로드

### CH12: 벡터 데이터베이스와 검색 최적화
- **파일**: `chapter_12.ipynb`
- **주제**: 대규모 벡터 검색 시스템
- **내용**:
  - FAISS 인덱스 성능 분석
  - HNSW 알고리즘 파라미터 튜닝
  - Pinecone 벡터 데이터베이스 활용
  - 메모리 사용량과 검색 속도 최적화

### CH14: 멀티모달 모델
- **파일**: `chapter_14.ipynb`
- **주제**: 이미지와 텍스트를 함께 처리하는 CLIP 모델
- **내용**:
  - CLIP 모델 구조 이해
  - 이미지-텍스트 매칭
  - 멀티모달 임베딩 생성

### CH15: AutoGen 에이전트
- **파일**: `chapter_15.ipynb`
- **주제**: 다중 에이전트 시스템
- **내용**:
  - AutoGen 프레임워크 활용
  - RAG 에이전트 구현
  - 에이전트 간 협업 시스템
  - 코드 실행 에이전트

### CH16: Mamba 아키텍처
- **파일**: `chapter_16.ipynb`
- **주제**: Transformer의 대안인 Mamba 모델
- **내용**:
  - Mamba 블록 구현
  - SSM (State Space Model) 메커니즘
  - Selective Scan 알고리즘
  - 선형 복잡도 시퀀스 모델링

## 🛠️ 설치 및 실행

### 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 추가 패키지들
각 챕터별로 필요한 추가 패키지들이 있습니다:

- **CH03**: `transformers`, `datasets`, `huggingface_hub`
- **CH06**: `bitsandbytes`, `accelerate`, `tiktoken`
- **CH07**: `auto-gptq`, `autoawq`, `optimum`
- **CH08**: `vllm`, `openai`
- **CH09**: `llama-index`, `chromadb`, `wandb`
- **CH10**: `sentence-transformers`, `faiss-cpu`
- **CH11**: `sentence-transformers`, `huggingface_hub`
- **CH12**: `pinecone-client`, `psutil`
- **CH14**: `transformers`
- **CH15**: `pyautogen[retrievechat]`

## 📖 학습 순서 추천

1. **CH02**: Transformer 기초 이해
2. **CH03**: Hugging Face 라이브러리 사용법
3. **CH05**: 프롬프트 엔지니어링 기법
4. **CH10**: 임베딩과 벡터 검색 기초
5. **CH09**: RAG 시스템 구현
6. **CH06**: 모델 평가 방법
7. **CH07**: 모델 최적화 (양자화)
8. **CH08**: 서빙 프레임워크
9. **CH11**: 임베딩 모델 학습
10. **CH12**: 벡터 데이터베이스 최적화
11. **CH14**: 멀티모달 모델
12. **CH15**: 에이전트 시스템
13. **CH16**: 최신 아키텍처 (Mamba)

## 🎯 주요 특징

- **실습 중심**: 모든 챕터가 실제 코드 예제를 포함
- **한국어 지원**: 한국어 모델과 데이터셋 활용
- **최신 기술**: 최신 LLM 기술과 도구들 소개
- **단계별 학습**: 기초부터 고급 주제까지 체계적 구성
- **성능 최적화**: 실제 서비스에 적용 가능한 최적화 기법

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 각 챕터의 코드는 개별적으로 실행 가능하며, 실제 프로젝트에 적용할 수 있습니다.

## 🤝 기여

이 저장소는 LLM 학습을 위한 실습 자료입니다. 개선 사항이나 추가 예제가 있다면 Pull Request를 통해 기여해 주세요.
