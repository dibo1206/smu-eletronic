"""챗봇 데모 (Streamlit) — RAG + Text2SQL을 LangGraph로 라우팅."""

import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

load_dotenv(ROOT_DIR / ".env")

from create_db import DB_PATH, main as create_db_main  # noqa: E402
from ai.retriever import PDF_PATHS, parse_schedule_to_slots  # noqa: E402
from ai.graph import ask  # noqa: E402

TIMETABLE_PATH = ROOT_DIR / "data" / "my_timetable.json"
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
PERIODS = range(1, 11)


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


def load_timetable() -> list[dict]:
    """앱을 껐다 켜도 담아둔 시간표가 남아있도록 파일에서 불러온다."""
    if not TIMETABLE_PATH.exists():
        return []
    try:
        return json.loads(TIMETABLE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_timetable(courses: list[dict]) -> None:
    TIMETABLE_PATH.parent.mkdir(exist_ok=True)
    TIMETABLE_PATH.write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")


def build_grid(courses: list[dict]) -> dict[tuple[str, int], str]:
    """[(요일, 교시): 과목명, ...] 형태로 시간표 칸을 채운다.

    "요일교시"는 두 가지 형식이 섞여 있다 — RAG(PDF)에서 온 건 "화6,7,8"처럼
    교시 번호, SQL(강좌 DB)에서 온 건 "월 09:00" 또는 "월 09:00~12:00"처럼
    실제 시각이다. retriever.parse_schedule_to_slots가 두 형식 다 인식해서
    (요일, 교시) 집합으로 바꿔준다 — 겹침 판단(retriever.py)과 같은 로직을
    쓰기 위해 파싱 함수 자체를 공유한다."""
    grid: dict[tuple[str, int], str] = {}
    for course in courses:
        name = course["과목명"]
        for day, period in parse_schedule_to_slots(course.get("요일교시", "")):
            if period in PERIODS:
                grid[(day, period)] = name
    return grid


def render_sidebar_timetable():
    with st.sidebar:
        st.header("📅 내 시간표")
        courses = st.session_state.my_timetable

        if not courses:
            st.caption("아직 담은 과목이 없어요. 답변 아래에서 과목을 골라 담아보세요.")
        else:
            used_days = [d for d in DAYS if any(d in c.get("요일교시", "") for c in courses)]
            used_days = used_days or DAYS[:5]
            grid = build_grid(courses)
            used_periods = [p for p in PERIODS if any((d, p) in grid for d in used_days)] or list(range(1, 5))

            header = "".join(f"<th>{d}</th>" for d in used_days)
            rows_html = ""
            for p in used_periods:
                cells = "".join(f"<td>{grid.get((d, p), '')}</td>" for d in used_days)
                rows_html += f"<tr><th>{p}</th>{cells}</tr>"
            # st.markdown은 마크다운 파서를 거치는데, 줄 앞에 공백(들여쓰기)이
            # 4칸 이상이면 마크다운이 "코드 블록"으로 오인해서 HTML이 그대로
            # 이스케이프된 텍스트로 나온다. 그래서 들여쓰기 없이 한 줄로 이어붙인다.
            table_html = (
                "<style>.timetable-grid{border-collapse:collapse;width:100%;font-size:0.75rem;}"
                ".timetable-grid th,.timetable-grid td{border:1px solid rgba(128,128,128,0.4);"
                "padding:4px;text-align:center;}</style>"
                f"<table class='timetable-grid'><tr><th></th>{header}</tr>{rows_html}</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

            st.divider()
            for i, course in enumerate(courses):
                cols = st.columns([4, 1])
                credit = f" · {course['학점']}학점" if course.get("학점") else ""
                cols[0].markdown(f"**{course['과목명']}**{credit}  \n{course.get('요일교시', '')}")
                if cols[1].button("삭제", key=f"remove_course_{i}"):
                    st.session_state.my_timetable.pop(i)
                    save_timetable(st.session_state.my_timetable)
                    st.rerun()

            if st.button("전체 비우기"):
                st.session_state.my_timetable = []
                save_timetable([])
                st.rerun()


def render_course_picker(courses: list[dict], key_prefix: str):
    """답변에 나온 과목 후보 중 골라서 내 시간표에 담는 UI."""
    if not courses:
        return
    options = {f"{c['과목명']} ({c.get('요일교시', '?')})": c for c in courses}
    selected = st.multiselect(
        "담을 과목 선택", list(options.keys()), key=f"{key_prefix}_select"
    )
    if st.button("선택한 과목 시간표에 담기", key=f"{key_prefix}_add"):
        existing_codes = {c.get("과목코드") for c in st.session_state.my_timetable}
        added = 0
        for label in selected:
            course = options[label]
            if course.get("과목코드") not in existing_codes:
                st.session_state.my_timetable.append(course)
                existing_codes.add(course.get("과목코드"))
                added += 1
        if added:
            save_timetable(st.session_state.my_timetable)
            st.success(f"{added}개 과목을 시간표에 담았습니다.")
            st.rerun()


def try_answer_from_my_timetable(question: str) -> dict | None:
    """"담은 과목 총 학점 몇이야"처럼 이미 세션에 있는 "내 시간표" 데이터로
    바로 계산되는 질문은 LLM/DB 없이 코드로 직접 답한다. 이런 질문을
    처리하는 전용 갈래를 classify_query에 새로 만들면 그 분류 자체가
    LLM 판단이라 흔들릴 수 있고(다른 질문이 잘못 그리로 샐 위험),
    "전공 학점 몇 점이야?"(졸업요건 문서 질문)처럼 진짜 학점 관련 질문과
    헷갈릴 수도 있다. 그래서 그래프에 보내기 전에 "담은"/"내 시간표"라는
    명시적 언급이 있을 때만 앱단에서 먼저 가로챈다 — 못 알아들으면 그냥
    None을 반환해서 원래 파이프라인(ask())으로 넘어가게 한다."""
    mentions_timetable = any(kw in question for kw in ["담은", "내 시간표", "내시간표"])
    if not mentions_timetable:
        return None

    courses = st.session_state.my_timetable
    base = {
        "query_type": "my_timetable", "sql": "", "pages": [], "category": "",
        "dataframe": None, "from_cache": False, "fallback_used": False, "courses": [],
    }

    if any(kw in question for kw in ["학점"]):
        if not courses:
            return {**base, "answer": "아직 담은 과목이 없어요."}
        known = [c for c in courses if c.get("학점")]
        total = sum(c["학점"] for c in known)
        answer = f"지금 담은 과목은 총 {len(courses)}개, {total}학점입니다."
        if len(known) < len(courses):
            answer += f" (학점 정보가 없는 {len(courses) - len(known)}개는 합계에서 제외)"
        return {**base, "answer": answer}

    if any(kw in question for kw in ["몇 개", "개수", "목록", "리스트", "뭐 담"]):
        if not courses:
            return {**base, "answer": "아직 담은 과목이 없어요."}
        lines = []
        for c in courses:
            credit = f", {c['학점']}학점" if c.get("학점") else ""
            lines.append(f"- {c['과목명']} ({c.get('요일교시', '?')}{credit})")
        answer = f"지금 담은 과목은 총 {len(courses)}개입니다.\n" + "\n".join(lines)
        return {**base, "answer": answer}

    return None


def render_answer_extras(payload: dict, key_prefix: str):
    """답변 하나에 딸린 배지/SQL/표/근거페이지/과목담기 UI를 렌더링한다.
    과거 메시지(히스토리 루프)와 방금 받은 답변(라이브 블록) 양쪽에서
    같은 로직을 쓰기 위해 함수로 뺐다."""
    if payload.get("query_type"):
        label = TYPE_LABEL.get(payload["query_type"], payload["query_type"])
        if payload.get("from_cache"):
            label += " · 🔁 캐시 재사용"
        if payload.get("fallback_used"):
            label += " · ↔️ 다른 경로에서 답 찾음"
        st.caption(label)
    if payload.get("sql"):
        with st.expander("실행된 SQL 보기"):
            st.code(payload["sql"], language="sql")
    if payload.get("dataframe") is not None:
        with st.expander("조회 결과 표로 보기"):
            st.dataframe(payload["dataframe"])
    if payload.get("category"):
        st.caption(f"적용된 필터: category = {payload['category']}")
    if payload.get("pages"):
        st.caption(f"참고 페이지: {', '.join(str(p) for p in payload['pages'])}")
    render_course_picker(payload.get("courses") or [], key_prefix)


st.set_page_config(page_title="공과대학 챗봇", page_icon="📚")
st.title("📚 상명대 공과대학 챗봇")
st.caption("과목/시간표, 절차/졸업요건뿐 아니라 자격증·취업분야·진출직업 같은 진로 질문까지 무엇이든 물어보세요.")

if "ready" not in st.session_state:
    ensure_data_ready()
    st.session_state.ready = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "my_timetable" not in st.session_state:
    st.session_state.my_timetable = load_timetable()

TYPE_LABEL = {
    "data": "🗂️ DB 조회", "document": "📄 안내문 검색", "general": "💬 일반 대화",
    "my_timetable": "📅 내 시간표 계산",
}

render_sidebar_timetable()

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_answer_extras(msg, key_prefix=f"hist_{idx}")

question = st.chat_input("예: 2학년 전공심화 과목 알려줘 / 졸업하려면 전공 학점 몇 점이야?")

if question:
    # 직전까지의 대화를 History로 넘겨서, "2학년 전공과목"처럼 학과가
    # 생략된 후속 질문도 이전 턴의 맥락(어느 학과 얘기였는지)을 이어받게 한다.
    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("조회 중..."):
            result = try_answer_from_my_timetable(question) or ask(
                question, history=history, my_timetable=st.session_state.my_timetable
            )

        st.markdown(result["answer"])
        # 이 답변이 history 루프에 들어갈 때의 인덱스와 같은 key_prefix를
        # 써서, 다음 rerun에서도 같은 위젯 상태로 이어지게 한다.
        render_answer_extras(result, key_prefix=f"hist_{len(st.session_state.messages)}")

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
            "courses": result.get("courses", []),
        }
    )
