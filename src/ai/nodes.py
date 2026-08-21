"""수강신청 챗봇 LangGraph의 노드/라우팅 함수.

rag-system의 src/ai/nodes.py와 같은 역할 — 실제 노드 로직을 한 파일에
모아두고, graph.py는 이 노드들을 엣지로 연결하는 역할만 한다.

1. classify_query 노드 — with_structured_output으로 질문을
   "general"(일반 대화) / "document"(안내문 조회) / "data"(DB 조회)로 분류
2. general 노드는 rag-system 베이스라인의 general_answer처럼 검색 없이
   바로 END로 간다. document 노드 → ai/retriever.py, data 노드 →
   ai/text2sql.py 호출

각 경로는 먼저 "같은 도메인 안에서" 재시도해보고, 그래도 안 되면 그때
"다른 도메인으로" 폴백한다 (rag-system 베이스라인의 vector_search ↔
rewrite_query 루프, database_query 자기 자신 재시도 루프와 같은 구조):
- document 노드가 불충분하면 → rewrite_document_query로 질문을 재작성해
  document 노드를 다시 시도 (최대 2회 재시도, 총 3번 검색)
- data 노드가 불충분하면 → data 노드 자신을 다시 시도 (이전 SQL을 피드백으로
  줘서 다른 방식으로 재생성, 최대 2회 재시도, 총 2번 실행)
- 그래도 안 되면 그제서야 반대 도메인(document ↔ data)으로 폴백한다.
  폴백 노드는 항상 END로 가서 왕복(핑퐁)이 일어나지 않고 1회만 시도한다
- 폴백으로 찾은 쪽의 답을 채택하고, query_type도 실제 답을 준 경로로
  갱신한다 (화면 배지가 실제 경로를 보여주도록).
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai.retriever import get_rag_engine
from ai.state import State
from ai.text2sql import get_text2sql_engine

load_dotenv(Path(__file__).parent.parent.parent / ".env")


class QueryType(str, Enum):
    GENERAL = "general"  # 인사, 잡담, 챗봇 자체에 대한 질문 — 검색 없이 바로 답변
    DOCUMENT = "document"  # 수강신청 절차/유의사항/졸업요건 등 안내문 기반 질문
    DATA = "data"  # 특정 과목/시간표/경쟁률 등 DB 조회가 필요한 질문


class QueryClassification(BaseModel):
    query_type: QueryType = Field(description="조회 유형")
    reason: str = Field(description="분류 이유")


_llm = init_chat_model("gpt-5.4-mini")
# 분류는 "정답이 하나로 정해진" 작업이라 temperature=0으로 고정한다.
# 온도가 있으면 같은 질문도 호출마다 document/data가 뒤바뀔 수 있어서
# (실제로 "이흥주 교수님 담당 과목 개수" 질문이 이 문제로 흔들린 적이 있다),
# 첫 분류의 일관성을 최대한 확보하기 위한 조치다.
_classifier_llm = init_chat_model("gpt-5.4-mini", temperature=0)

MAX_REWRITE_RETRIES = 2  # document 경로: 질문 재작성 재시도 횟수
MAX_SQL_RETRIES = 2  # data 경로: SQL 재생성 재시도 횟수

INSUFFICIENT_PHRASES = [
    "정보가 없습니다", "확인할 수 없습니다", "찾을 수 없습니다",
    "해당 정보가 없습니다", "결과가 없", "알 수 없습니다",
]


def _is_insufficient(answer: str, has_evidence: bool) -> bool:
    """RAG 답변이 사실상 '못 찾았다'는 뜻인지 판단한다.
    RAG 프롬프트(ai/retriever.py)가 이 문구들을 쓰도록 고정 지시돼 있어서
    문구 매칭으로도 안정적으로 잡힌다."""
    if not has_evidence:
        return True
    return any(phrase in answer for phrase in INSUFFICIENT_PHRASES)


class SqlSummary(BaseModel):
    """SQL 실행 결과를 요약하면서, 그 결과가 질문에 실제로 답이 되는지도
    같이 판단한다. 자유 텍스트에서 '모르겠다' 문구를 찾는 것보다,
    구조화된 필드(sufficient)로 직접 판단시키는 게 훨씬 안정적이다."""

    answer: str = Field(description="질문에 대한 한국어 답변 (한두 문장)")
    sufficient: bool = Field(
        description="이 SQL 실행 결과가 질문에 실제로 답이 되면 true. "
        "결과가 비어있거나, 질문과 관련 없는 값만 나와서 질문에 답할 수 "
        "없으면 false"
    )


def _summarize_sql_result(question: str, sql: str, result) -> SqlSummary:
    structured_llm = _llm.with_structured_output(SqlSummary)
    prompt = f"""다음 SQL 실행 결과를 보고 질문에 답변하세요.
결과에 없는 내용은 추측하지 마세요.

질문: {question}
SQL: {sql}
실행 결과: {result}"""
    return structured_llm.invoke(prompt)


def _get_question(state: State) -> str:
    """대화 기록에서 실제 사용자 질문을 찾는다.
    폴백 노드는 이전 노드가 붙인 AIMessage 뒤에서 실행되므로,
    단순히 마지막 메시지를 읽으면 그 AI 답변을 질문으로 오인하게 된다."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return state["messages"][-1].content


def classify_query(state: State) -> dict:
    question = _get_question(state)
    structured_llm = _classifier_llm.with_structured_output(QueryClassification)

    prompt = f"""다음 사용자 질문의 조회 유형을 분류하세요.
세 카테고리를 동시에 비교하지 말고, 아래 순서대로 하나씩 확인해서
처음으로 해당하는 카테고리를 고르세요.

[1단계] 질문에 실질적인 정보 요청이 전혀 없이 인사/감사/잡담/챗봇
자체(뭐 할 수 있어?)에 대한 것뿐인가? → 그렇다면 general.
(인사말이 섞여 있어도 뒤에 실제 질문이 붙어 있으면 general이 아니라
그 질문의 내용으로 2·3단계를 계속 판단하세요.
예: "안녕! 2학년 전공심화 과목 알려줘" → 인사는 무시하고 data)
예(general): "안녕", "고마워", "너는 뭐 하는 챗봇이야?", "잘 지내?"

[2단계] 학사 안내문(규정/설명)에 있는 내용인가? → 그렇다면 document.
- 수강신청 절차, 정정기간, 유의사항, 졸업요건, 복수전공, 재수강 규정
- 교수님 **본인 신상**(세부전공, 연구실 위치, 학위, 개인 연락처)
- **교양 과목**(기초교양/균형교양 등 "교양"이 붙은 과목)의 시간표·담당교수·추천
  — 교양 과목 시간표는 DB가 아니라 안내 문서에만 들어있다.
예(document): "졸업하려면 전공 몇 학점이야?", "재수강 규정이 어떻게 돼?",
    "이흥주 교수님 연구실이 어디야?", "조준희 교수님 세부전공이 뭐야?",
    "예술 영역 교양 과목 추천해줘", "영어 교양 과목 뭐 있어?"

[3단계] 여기까지 왔으면 data. 강좌 DB(과목/시간표/경쟁률/담당교수)를
조회해야 답할 수 있는 질문이다. 교수님의 **신상이 아니라 그 교수가
맡은 과목**(개수·목록·시간표)을 묻는 것이면 신상 질문처럼 보여도 항상
data다.
예(data): "2학년 전공심화 과목 알려줘", "화요일에 열리는 강의는?",
    "정민철 교수님이 담당하는 과목이 뭐야?",
    "이흥주 교수님이 담당하는 과목 개수는?"

혼동하기 쉬운 경우 — "챗봇이 뭘 할 수 있는지"를 추상적으로 묻는 것은
general이지만, "실제 과목/정보가 뭐가 있는지" 구체적으로 묻는 것은
document/data다.
예: "너 과목 정보도 알려줄 수 있어?"(추상적 능력 질문) → general
    "2학년 과목 뭐 있어?"(구체적 정보 요청) → data

사용자 질문: {question}"""

    result = structured_llm.invoke(prompt)
    return {"query_type": result.query_type.value}


def route_by_query_type(state: State) -> Literal["general", "document", "data"]:
    return state["query_type"]


def general_answer(state: State) -> dict:
    """rag-system 베이스라인의 general_answer와 같은 역할 —
    검색 없이 LLM이 바로 답하고 END로 간다 (폴백 대상이 아니다)."""
    question = _get_question(state)
    system_prompt = """당신은 상명대학교 전자공학과 수강신청 챗봇입니다.
인사나 잡담, 챗봇 자체에 대한 질문에 친절하고 자연스럽게 답변하세요.
필요하면 이 챗봇으로 수강신청 절차, 졸업요건, 과목 시간표·경쟁률,
교수님 정보 등을 물어볼 수 있다고 안내하세요."""
    response = _llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    )
    return {"messages": [AIMessage(content=response.content)]}


def _document_node(state: State, *, is_fallback: bool) -> dict:
    question = _get_question(state)
    # rewrite_document_query가 재작성한 검색어가 있으면 검색에만 쓰고,
    # 최종 답변은 항상 사용자의 원래 질문에 맞춰 생성한다.
    search_query = state.get("rewritten_query") if not is_fallback else None
    result = get_rag_engine().answer(question, search_query=search_query)

    return {
        "messages": [AIMessage(content=result["answer"])],
        "query_type": "document",
        # data 노드가 먼저 시도했던 SQL이 있으면 참고용으로 그대로 남겨둔다
        "sql": state.get("sql", "") if is_fallback else "",
        "pages": result["pages"],
        "category": result.get("category") or "",
        "dataframe": None,
        "from_cache": False,
        "fallback_used": is_fallback,
        "insufficient": _is_insufficient(result["answer"], bool(result["pages"])),
    }


def rewrite_document_query(state: State) -> dict:
    """안내문 검색 결과가 부족할 때 질문을 재작성해서 document 노드를
    다시 태운다 (rag-system 베이스라인의 rewrite_query와 같은 역할)."""
    question = _get_question(state)
    prompt = f"""다음 질문으로 안내 문서를 검색했지만 관련 내용을 찾지 못했습니다.
검색에 더 유리하도록 동의어나 다른 표현을 사용해 질문을 재작성하세요.
재작성된 질문만 반환하고 설명은 포함하지 마세요.

원래 질문: {question}"""
    response = _llm.invoke(prompt)
    return {
        "rewritten_query": response.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _data_node(state: State, *, is_fallback: bool) -> dict:
    question = _get_question(state)
    # 재시도(자기 자신으로 루프백)일 때는 이전에 실행됐지만 질문에 답이
    # 안 됐던 SQL을 피드백으로 줘서 다른 방식으로 재생성하게 한다.
    retry_count = 0 if is_fallback else state.get("retry_count", 0)
    extra_feedback = None
    if retry_count > 0 and state.get("sql"):
        extra_feedback = (
            f"이전 SQL은 정상 실행됐지만 질문에 실제로 답하지 못했습니다: {state['sql']}\n"
            "다른 조건/테이블/집계 방식으로 다시 시도하세요."
        )

    outcome = get_text2sql_engine().query(
        question, use_cache=(retry_count == 0), extra_feedback=extra_feedback
    )

    has_rows = not outcome["error"] and outcome["result"] not in (None, "", "[]")
    if outcome["error"]:
        answer = f"조회에 실패했어요: {outcome['error']}"
        sufficient = False
    elif has_rows:
        summary = _summarize_sql_result(question, outcome["sql"], outcome["result"])
        answer = summary.answer
        sufficient = summary.sufficient
    else:
        answer = "조회된 결과가 없습니다."
        sufficient = False

    result_dict = {
        "messages": [AIMessage(content=answer)],
        "query_type": "data",
        "sql": outcome["sql"] or "",
        "pages": [],
        "category": "",
        "dataframe": outcome.get("dataframe"),
        "from_cache": outcome.get("from_cache", False),
        "fallback_used": is_fallback,
        "insufficient": not sufficient,
    }
    if not is_fallback:
        result_dict["retry_count"] = retry_count + 1
    return result_dict


def answer_from_document(state: State) -> dict:
    return _document_node(state, is_fallback=False)


def document_fallback(state: State) -> dict:
    return _document_node(state, is_fallback=True)


def answer_from_data(state: State) -> dict:
    return _data_node(state, is_fallback=False)


def data_fallback(state: State) -> dict:
    return _data_node(state, is_fallback=True)


def route_after_document(state: State) -> Literal["rewrite", "fallback", "end"]:
    if not state["insufficient"]:
        return "end"
    if state.get("retry_count", 0) < MAX_REWRITE_RETRIES:
        return "rewrite"
    return "fallback"


def route_after_data(state: State) -> Literal["retry", "fallback", "end"]:
    if not state["insufficient"]:
        return "end"
    if state.get("retry_count", 0) < MAX_SQL_RETRIES:
        return "retry"
    return "fallback"
