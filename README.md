# 전자공학과 수강신청 챗봇

상명대학교 전자공학과(천안) 학생을 위한 RAG · Text2SQL 기반 수강신청 챗봇입니다.
과목/시간표/경쟁률처럼 데이터가 필요한 질문은 Text2SQL로, 수강신청 절차·졸업요건·교수님 정보처럼
문서 설명이 필요한 질문은 RAG로, 인사·잡담처럼 검색이 필요 없는 질문은 바로 답변으로
자동 라우팅합니다.

## 아키텍처

```
사용자 질문
   │
   ▼
classify_query — "general" / "document" / "data"로 분류 (with_structured_output,
                  temperature=0으로 고정해 같은 질문엔 항상 같은 결과)
   │
   ├─ general ──▶ 검색 없이 바로 답변 (인사·잡담·챗봇 소개)
   │
   ├─ document ─▶ RAG 엔진 (src/ai/retriever.py)
   │              메타데이터 필터로 카테고리 좁힌 뒤
   │              Parent Document Retriever로 Qdrant Cloud 검색
   │              → 근거 페이지 기반 답변 생성
   │              │
   │              ├─ 불충분하면 rewrite_document_query가 질문을 재작성해
   │              │  document를 다시 시도 (최대 2회 재작성)
   │              └─ 그래도 부족하면 data로 폴백
   │
   └─ data ─────▶ Text2SQL 엔진 (src/ai/text2sql.py)
                  LangChain SQLDatabase로 SQL 생성 → 검증 → 실행
                  → 실행 결과 기반 답변 생성
                  │
                  ├─ 불충분하면 이전 SQL을 피드백 삼아 data 자신을 다시
                  │  시도 (최대 2회 재시도)
                  └─ 그래도 부족하면 document로 폴백
```

각 경로는 먼저 같은 도메인 안에서 재시도해보고, 그래도 안 되면 반대
도메인으로 한 번 더 시도한다 (rag-system 베이스라인의 `vector_search ↔
rewrite_query` 루프, `database_query` 자기 재시도 루프와 같은 구조).

## 폴더 구조

rag-system(`src/ai/{state,nodes,graph}.py` 분리) 구조를 그대로 따르고,
데이터 생성용 일회성 스크립트는 `scripts/`로, 데이터 산출물은
`data/`·`datasets/`·`sugang.db`로 코드와 분리했다.

| 경로 | 역할 |
| --- | --- |
| `src/ai/state.py` | LangGraph State 정의 |
| `src/ai/nodes.py` | 분류/검색/SQL 실행 등 실제 노드 로직 + 폴백 라우팅 함수 |
| `src/ai/graph.py` | 노드를 엣지로 연결해 그래프 조립 (`build_graph`, `ask`) |
| `src/ai/retriever.py` | PDF 로드 → 청킹 → Qdrant 적재 → Parent Document Retriever + 메타데이터 필터링 |
| `src/ai/text2sql.py` | 자연어 → SQL 변환·검증·실행 엔진 |
| `src/demo/app.py` | Streamlit 챗봇 UI |
| `scripts/create_db.py` | 강좌/수강신청현황/교수진 데이터를 SQLite(`sugang.db`)로 적재 |
| `scripts/make_faculty_pdf.py`, `make_pdf.py` | 안내 PDF 생성 |
| `scripts/merge_official_pdfs.py` | 학교 공식 PDF 5종을 하나로 병합 |
| `datasets/` | RAG용 PDF 원본 (공식자료 통합본, 교수진 안내) |
| `data/` | 생성된 CSV (gitignore 대상) |
| `langgraph.json` | `langgraph dev`(LangGraph Studio)용 그래프 진입점 설정 |

## 데이터 출처

- **강좌 정보**: 학교 공식 문서(`datasets/2026-2학기_수강신청_공식자료_통합.pdf`,
  "5. 2026-2학기 개설학과별 시간표" 중 "공과대학 전자공학과" 표)에 실제로 나오는
  과목만 담았다. 전자공학과 커리큘럼 전체가 아니라, 2026-2학기에 개설된다고
  공식 문서로 확인되는 12개 과목만 있다 — 담당교수를 임의로 지어내지 않기 위해서다.
  수강 정원은 공식 문서에 없는 값이라 실제 수강신청 화면 기준으로 조사해둔 값을 썼고,
  수강신청현황(신청 인원)은 경쟁률 데모용으로 생성한 통계값이다(특정인의 신원과
  무관하므로 실제 데이터가 없어도 문제되지 않는다).
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
uv run python scripts/create_db.py          # 강좌 DB 생성
uv run python scripts/make_faculty_pdf.py   # 교수진 안내 PDF 생성
```

(`datasets/2026-2학기_수강신청_공식자료_통합.pdf`는 이미 저장소에 포함되어 있습니다.)

### 4. 실행

```bash
uv run streamlit run src/demo/app.py
```

### 5. (선택) LangGraph Studio로 그래프 구조 확인

```bash
uv run langgraph dev
```

터미널에 뜨는 Studio UI 링크(`https://smith.langchain.com/studio/?baseUrl=...`)를
열면 노드/엣지 그래프를 시각적으로 확인하고 직접 질문을 넣어볼 수 있다.
SQL 조회 결과 표(dataframe)는 JSON 직렬화가 안 돼 Studio에는 `null`로 보이니,
표까지 확인하려면 Streamlit 데모를 쓴다.

## 참고

이 프로젝트는 `rag-system` 실습(Parent Document Retriever, 메타데이터 필터링,
LangGraph 조건부 라우팅)에서 배운 패턴을 그대로 적용해 만들었습니다.
