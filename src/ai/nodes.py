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

import pandas as pd
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

class DocumentSummary(BaseModel):
    """RAG 답변이 질문에 실제로 쓸모 있는 정보를 줬는지 판단한다.

    예전엔 "확인되지 않았습니다" 같은 고정 문구를 답변에서 찾는 방식이었는데,
    복합 질문에서 일부만 답할 때 그 답변이 (안내 프롬프트 지시대로) "OOO는
    확인하지 못했습니다"라는 캐비엇을 정직하게 붙이는 바람에, 실제로는
    쓸모 있는 정보(예: 교양 과목 2개 추천)가 들어있는 답변까지 문구 하나
    때문에 전부 불충분으로 오판되는 문제가 있었다. SQL 쪽(SqlSummary)과
    같은 방식으로 구조화된 판단으로 바꿔서 이 문제를 없앤다."""

    sufficient: bool = Field(
        description="답변에 질문과 관련된 구체적이고 실질적인 정보(과목명, "
        "시간, 규정 등 실제 내용)가 하나라도 포함돼 있으면 true — 질문의 "
        "일부만 답했고 나머지는 확인 못 했다는 캐비엇이 섞여 있어도, 답한 "
        "부분에 실제 정보가 있으면 true다. 답변이 사실상 '못 찾았다'는 "
        "내용뿐이면 false."
    )


def _is_insufficient(question: str, answer: str, has_evidence: bool) -> bool:
    """RAG 답변이 사실상 '못 찾았다'는 뜻인지 판단한다."""
    if not has_evidence:
        return True
    structured_llm = _llm.with_structured_output(DocumentSummary)
    prompt = f"""다음은 안내 문서를 검색해서 생성한 답변입니다. 질문에 실질적으로
도움이 되는 구체적인 정보가 들어있는지 판단하세요.

질문: {question}
답변: {answer}"""
    result = structured_llm.invoke(prompt)
    return not result.sufficient


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
    """SQL 결과를 자연어 답변으로 요약한다.

    진로정보 테이블을 조회한 경우에만 학교 CSV보다 LLM이 더 잘 아는
    영역(자격증 활용법, 업계 동향 등)이라 일반 지식 보충을 허용한다.
    이 허용 여부는 "진로정보" 문자열이 SQL에 들어있는지로 코드가 직접
    판단한다(또 다른 LLM 분류를 추가하면 그 분류 자체가 흔들려서 강좌/
    교수진 같은 다른 데이터에도 새어 들어갈 위험이 있기 때문). 강좌/
    교수진/수강신청현황 조회는 지금처럼 SQL 결과 밖으로 못 나가게 그대로
    막아둔다 — 시간표나 연락처 같은 사실 정보에 LLM이 확인 안 된 내용을
    섞으면 안 되니까."""
    structured_llm = _llm.with_structured_output(SqlSummary)
    if "진로정보" in sql:
        prompt = f"""다음 SQL 실행 결과를 보고 질문에 답변하세요.
학교 자료(SQL 결과)가 우선이지만, 자격증 활용처럼 학교 자료에 없는
일반적인 지식으로 보충 설명을 덧붙여도 됩니다. 다만 학교 자료와 당신의
일반 지식을 명확히 구분해서 답변하세요 — 예: "학교 자료 기준으로는
~이고, 일반적으로는 ~" 처럼. 학교 자료인 것처럼 섞어서 말하지 마세요.

질문: {question}
SQL: {sql}
실행 결과: {result}"""
    else:
        prompt = f"""다음 SQL 실행 결과를 보고 질문에 답변하세요.
결과에 없는 내용은 추측하지 마세요.

질문: {question}
SQL: {sql}
실행 결과: {result}"""
    return structured_llm.invoke(prompt)


def _get_question(state: State) -> str:
    """대화 기록에서 실제 사용자 질문을 찾는다.
    resolve_question 노드가 이전 대화 맥락을 반영해 다시 쓴 질문이 있으면
    그걸 쓰고(예: "2학년 전공과목" → "휴먼지능로봇공학과 2학년 전공과목"),
    없으면(첫 턴 등) 원래 질문으로 폴백한다.
    폴백 노드는 이전 노드가 붙인 AIMessage 뒤에서 실행되므로,
    단순히 마지막 메시지를 읽으면 그 AI 답변을 질문으로 오인하게 된다."""
    if state.get("resolved_question"):
        return state["resolved_question"]
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return state["messages"][-1].content


class ResolvedQuestion(BaseModel):
    standalone_question: str = Field(
        description="이전 대화 맥락까지 반영해 그 자체로 완전한 하나의 질문으로 "
        "다시 쓴 것. 새 질문이 이미 그 자체로 완전하거나 이전 대화와 무관하면 "
        "새 질문을 그대로 반환한다."
    )


def resolve_question(state: State) -> dict:
    """후속 질문이 이전 대화의 주어(학과/학년/교수님 등)를 생략하고 있으면
    (예: "휴먼지능로봇공학과 전공과목" 다음에 "2학년 전공과목"만 물어보는
    경우) 이전 맥락을 채워 그 자체로 완전한 질문으로 다시 쓴다. classify_query
    를 포함한 이후 모든 노드가 _get_question()을 통해 이 결과를 쓰게 되므로,
    라우팅·검색·SQL 생성이 전부 맥락을 이어받는다."""
    messages = state["messages"]
    current = messages[-1].content

    if len(messages) <= 1:
        # 첫 턴 — 참고할 이전 대화가 없다.
        return {"resolved_question": current}

    history_text = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else '챗봇'}: {m.content}"
        for m in messages[:-1]
    )
    structured_llm = _classifier_llm.with_structured_output(ResolvedQuestion)
    prompt = f"""아래는 챗봇과 사용자의 이전 대화 기록과 사용자의 새 질문입니다.
새 질문이 이전 대화에서 언급된 학과/학년/교수님 등 주어를 생략한 후속
질문이면, 그 맥락을 채워 넣어 그 자체로 완전한 하나의 질문으로 다시
쓰세요. 새 질문이 이미 완전하거나 이전 대화와 무관한 새로운 주제면
새 질문을 그대로 반환하세요.

<이전 대화>
{history_text}
</이전 대화>

새 질문: {current}"""
    result = structured_llm.invoke(prompt)
    return {"resolved_question": result.standalone_question}


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
- **교양 과목**(기초교양/균형교양 등 "교양"이 붙은 과목)의 시간표·담당교수·추천
  — 교양 과목 시간표는 DB가 아니라 안내 문서에만 들어있다.
예(document): "졸업하려면 전공 몇 학점이야?", "재수강 규정이 어떻게 돼?",
    "예술 영역 교양 과목 추천해줘", "영어 교양 과목 뭐 있어?"

[3단계] 여기까지 왔으면 data. 강좌 DB(과목/시간표/경쟁률/담당교수),
교수진 DB(학위/세부전공/연구실/연락처), 진로정보 DB(학과별 취업분야/
진출직업/관련자격증)를 조회해야 답할 수 있는 질문이다.
교수님에 대한 질문은 담당 과목(개수·목록·시간표)이든 본인 신상(세부전공,
연구실, 연락처)이든 항상 data다 — 공과대학 11개 학과 교수님 정보가
교수진 DB에 있다. 자격증·취업·진로·회사 관련 질문도 항상 data다 —
공과대학 11개 학과 전체의 진로정보가 DB에 있다.
예(data): "2학년 전공심화 과목 알려줘", "화요일에 열리는 강의는?",
    "정민철 교수님이 담당하는 과목이 뭐야?",
    "이흥주 교수님이 담당하는 과목 개수는?",
    "이흥주 교수님 연구실이 어디야?", "조준희 교수님 세부전공이 뭐야?",
    "정보보안공학과 관련 자격증 뭐 있어?", "전자공학과 졸업하면 어떤 회사 가?",
    "지능형로봇학과 진출 직업이 뭐야?"

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
    system_prompt = """당신은 상명대학교 공과대학 학사·진로 안내 챗봇입니다.
인사나 잡담, 챗봇 자체에 대한 질문에 친절하고 자연스럽게 답변하세요.
필요하면 이 챗봇으로 수강신청 절차, 졸업요건, 과목 시간표·경쟁률,
교수님 정보뿐 아니라 학과별 취업분야·진출직업·관련 자격증 같은 진로
정보도 물어볼 수 있다고 안내하세요."""
    response = _llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    )
    return {"messages": [AIMessage(content=response.content)]}


def _document_node(state: State, *, is_fallback: bool) -> dict:
    question = _get_question(state)
    # rewrite_document_query가 재작성한 검색어가 있으면 검색에만 쓰고,
    # 최종 답변은 항상 사용자의 원래 질문에 맞춰 생성한다.
    search_query = state.get("rewritten_query") if not is_fallback else None
    result = get_rag_engine().answer(
        question, search_query=search_query, my_timetable=state.get("my_timetable")
    )
    insufficient = _is_insufficient(question, result["answer"], bool(result["pages"]))

    if is_fallback and insufficient and state.get("primary_result"):
        # 폴백도 부족하면, 원래 있던 (비록 불완전해도 더 나은) 답을
        # "조회된 결과가 없습니다" 같은 걸로 덮어쓰지 않고 그대로 유지한다.
        return {**state["primary_result"], "fallback_used": False, "insufficient": False}

    output = {
        "messages": [AIMessage(content=result["answer"])],
        "query_type": "document",
        # data 노드가 먼저 시도했던 SQL이 있으면 참고용으로 그대로 남겨둔다
        "sql": state.get("sql", "") if is_fallback else "",
        "pages": result["pages"],
        "category": result.get("category") or "",
        "dataframe": None,
        "from_cache": False,
        "fallback_used": is_fallback,
        "insufficient": insufficient,
        "courses": result.get("courses", []),
    }
    if not is_fallback:
        output["primary_result"] = dict(output)
    return output


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


def _courses_from_dataframe(df: pd.DataFrame | None) -> list[dict]:
    """SQL 조회 결과가 과목 목록처럼 보이면(과목명+요일 컬럼이 있으면) "내
    시간표에 담기" UI가 쓸 수 있는 구조로 변환한다. 질문마다 SELECT하는
    컬럼이 달라서 학점/시작시간처럼 없는 컬럼도 있을 수 있어, 있는 것만
    최대한 채운다."""
    if df is None or df.empty or not {"과목명", "요일"}.issubset(df.columns):
        return []
    courses = []
    for _, row in df.iterrows():
        schedule = str(row["요일"])
        if "시작시간" in df.columns and pd.notna(row["시작시간"]):
            schedule += f" {row['시작시간']}"
            if "종료시간" in df.columns and pd.notna(row["종료시간"]):
                schedule += f"~{row['종료시간']}"
        courses.append({
            "과목코드": row["과목코드"] if "과목코드" in df.columns else row["과목명"],
            "과목명": row["과목명"],
            "학점": int(row["학점"]) if "학점" in df.columns and pd.notna(row["학점"]) else None,
            "요일교시": schedule,
            "학과": row["학과"] if "학과" in df.columns else None,
            "출처": "강좌 DB",
        })
    return courses


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

    if is_fallback and not sufficient and state.get("primary_result"):
        # 폴백(SQL)도 부족하면, 원래 있던 (비록 불완전해도 더 나은) 답을
        # "조회된 결과가 없습니다" 같은 걸로 덮어쓰지 않고 그대로 유지한다.
        return {**state["primary_result"], "fallback_used": False, "insufficient": False}

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
        "courses": _courses_from_dataframe(outcome.get("dataframe")),
    }
    if not is_fallback:
        result_dict["retry_count"] = retry_count + 1
        result_dict["primary_result"] = dict(result_dict)
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
