# 전자공학과 수강신청 챗봇

상명대학교 전자공학과(천안) 학생을 위한 RAG · Text2SQL 기반 수강신청 챗봇입니다.
과목/시간표/경쟁률처럼 데이터가 필요한 질문은 Text2SQL로, 수강신청 절차·졸업요건·교수님 정보처럼
문서 설명이 필요한 질문은 RAG로 자동 라우팅해서 답합니다.

## 아키텍처

```
사용자 질문
   │
   ▼
LangGraph 라우터 (ai/graph.py)
   │  질문 의도를 "document" / "data"로 분류 (with_structured_output)
   │
   ├─ document ─▶ RAG 엔진 (ai/rag.py)
   │              메타데이터 필터로 카테고리 좁힌 뒤
   │              Parent Document Retriever로 Qdrant Cloud 검색
   │              → 근거 페이지 기반 답변 생성
   │
   └─ data ─────▶ Text2SQL 엔진 (ai/text2sql.py)
                  LangChain SQLDatabase로 SQL 생성 → 검증 → 실행
                  → 실행 결과 기반 답변 생성
```

## 폴더 구조

| 경로 | 역할 |
| --- | --- |
| `ai/create_db.py` | 강좌/수강신청현황 데이터를 SQLite(`ai/sugang.db`)로 적재 |
| `ai/text2sql.py` | 자연어 → SQL 변환·검증·실행 엔진 |
| `ai/make_faculty_pdf.py` | 교수진 안내 PDF 생성 |
| `ai/merge_official_pdfs.py` | 학교 공식 PDF 5종을 하나로 병합 |
| `ai/rag.py` | PDF 로드 → 청킹 → Qdrant 적재 → Parent Document Retriever + 메타데이터 필터링 |
| `ai/graph.py` | RAG/Text2SQL 라우팅 (LangGraph) |
| `ai/datasets/` | RAG용 PDF 원본 (공식자료 통합본, 교수진 안내) |
| `demo/app.py` | Streamlit 챗봇 UI |

## 데이터 출처

- **강좌 정보**: 상명대학교 전자공학과(천안) 2026학년도 공식 교육과정 + 2026-2학기 실제 개설강좌 정보
- **안내 문서**: 학교에서 제공한 2026-2학기 수강신청 안내자료, 학사안내 자료, 수강제한 강좌 목록,
  타학과 전공선택 인정 교과목, 개설학과별 시간표 (5종 병합)
- **교수진 정보**: 전자공학과 교수소개 페이지

## 시작하기

### 1. 환경 변수

`.env.example`을 복사해 `.env`를 만들고 키를 채웁니다.

```bash
cp .env.example .env
```

- `OPENAI_API_KEY`: LLM(gpt-5.4-mini) 및 임베딩(text-embedding-3-large)용
- `QDRANT_URL`, `QDRANT_API_KEY`: RAG 벡터 검색용 (Qdrant Cloud)

### 2. 설치

```bash
uv sync
```

### 3. 데이터 준비

```bash
uv run python ai/create_db.py          # 강좌 DB 생성
uv run python ai/make_faculty_pdf.py   # 교수진 안내 PDF 생성
```

(`ai/datasets/2026-2학기_수강신청_공식자료_통합.pdf`는 이미 저장소에 포함되어 있습니다.)

### 4. 실행

```bash
uv run streamlit run demo/app.py
```

## 참고

이 프로젝트는 `rag-system` 실습(Parent Document Retriever, 메타데이터 필터링,
LangGraph 조건부 라우팅)에서 배운 패턴을 그대로 적용해 만들었습니다.
