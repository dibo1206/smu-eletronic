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



