"""챗봇 데모 (Streamlit) — RAG + Text2SQL을 LangGraph로 라우팅."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

load_dotenv(ROOT_DIR / ".env")

from create_db import DB_PATH, main as create_db_main  # noqa: E402
from ai.retriever import PDF_PATHS  # noqa: E402
from ai.graph import ask  # noqa: E402


def ensure_data_ready():
    if not DB_PATH.exists():
        st.info("강좌 DB가 없습니다. 데이터를 생성합니다...")
        create_db_main()
    missing = [p.name for p in PDF_PATHS if not p.exists()]
    if missing:
        st.error(
            "다음 안내 문서 파일이 없습니다: "
            + ", ".join(missing)
            + " — datasets/ 폴더에 넣어주세요."
        )
        st.stop()


st.set_page_config(page_title="공과대학 챗봇", page_icon="📚")
st.title("📚 상명대 공과대학 챗봇")
st.caption("과목/시간표, 절차/졸업요건뿐 아니라 자격증·취업분야·진출직업 같은 진로 질문까지 무엇이든 물어보세요.")

if "ready" not in st.session_state:
    ensure_data_ready()
    st.session_state.ready = True

if "messages" not in st.session_state:
    st.session_state.messages = []

TYPE_LABEL = {"data": "🗂️ DB 조회", "document": "📄 안내문 검색", "general": "💬 일반 대화"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("query_type"):
            label = TYPE_LABEL.get(msg["query_type"], msg["query_type"])
            if msg.get("from_cache"):
                label += " · 🔁 캐시 재사용"
            if msg.get("fallback_used"):
                label += " · ↔️ 다른 경로에서 답 찾음"
            st.caption(label)
        if msg.get("sql"):
            with st.expander("실행된 SQL 보기"):
                st.code(msg["sql"], language="sql")
        if msg.get("dataframe") is not None:
            with st.expander("조회 결과 표로 보기"):
                st.dataframe(msg["dataframe"])
        if msg.get("category"):
            st.caption(f"적용된 필터: category = {msg['category']}")
        if msg.get("pages"):
            st.caption(f"참고 페이지: {', '.join(str(p) for p in msg['pages'])}")

question = st.chat_input("예: 2학년 전공심화 과목 알려줘 / 졸업하려면 전공 학점 몇 점이야?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("조회 중..."):
            result = ask(question)

        st.markdown(result["answer"])
        type_label = TYPE_LABEL.get(result["query_type"], result["query_type"])
        if result.get("from_cache"):
            type_label += " · 🔁 캐시 재사용"
        if result.get("fallback_used"):
            type_label += " · ↔️ 다른 경로에서 답 찾음"
        st.caption(type_label)
        if result.get("sql"):
            with st.expander("실행된 SQL 보기"):
                st.code(result["sql"], language="sql")
        if result.get("dataframe") is not None:
            with st.expander("조회 결과 표로 보기"):
                st.dataframe(result["dataframe"])
        if result.get("category"):
            st.caption(f"적용된 필터: category = {result['category']}")
        if result.get("pages"):
            st.caption(f"참고 페이지: {', '.join(str(p) for p in result['pages'])}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "query_type": result["query_type"],
            "sql": result.get("sql", ""),
            "pages": result.get("pages", []),
            "category": result.get("category", ""),
            "dataframe": result.get("dataframe"),
            "from_cache": result.get("from_cache", False),
            "fallback_used": result.get("fallback_used", False),
        }
    )
