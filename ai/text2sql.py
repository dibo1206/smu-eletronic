"""
수강신청 챗봇의 Text2SQL 엔진.

LangChain의 SQLDatabase로 SQLite에 연결하고, LLM으로 SQL을 생성한다.
LLM이 생성한 SQL은 실행 전에 validate_sql()로 한 번 더 검증한다
(SELECT만 허용, 테이블 화이트리스트, 위험 키워드 차단) — LLM이 실수해도
코드가 마지막 방어선 역할을 하도록.
"""

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).parent.parent / ".env")

DB_PATH = Path(__file__).parent / "sugang.db"

ALLOWED_TABLES = {"강좌", "수강신청현황"}

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "CREATE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA",
}

MAX_RETRIES = 2


def validate_sql(sql: str) -> tuple[bool, str]:
    """실행 전 SQL 검증: SELECT 단일 문장만, 테이블은 화이트리스트만."""
    if not sql or not sql.strip():
        return False, "SQL이 생성되지 않았습니다."

    sql = sql.strip().rstrip(";")
    sql_upper = sql.upper()

    if ";" in sql:
        return False, "여러 SQL 문장은 실행할 수 없습니다."

    if not re.match(r"^\s*SELECT\b", sql_upper):
        return False, "SELECT 문만 사용할 수 있습니다."

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", sql_upper):
            return False, f"{kw} 명령은 사용할 수 없습니다."

    tables = re.findall(r"\b(?:FROM|JOIN)\s+([가-힣a-zA-Z_][가-힣a-zA-Z0-9_]*)", sql)
    for table in tables:
        if table not in ALLOWED_TABLES:
            return False, f"허용되지 않은 테이블입니다: {table}"

    return True, ""


class Text2SQLEngine:
    def __init__(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"DB가 없습니다: {DB_PATH}\n먼저 `python ai/create_db.py`를 실행하세요."
            )
        self.db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        self.llm = init_chat_model("gpt-5.4-mini")
        self.schema_info = self.db.get_table_info()

    def _build_system_prompt(self, feedback: str | None) -> str:
        prompt = f"""당신은 SQLite 전문가입니다.
사용자의 질문을 정확한 SQL 쿼리로 변환하세요.

<database_schema>
{self.schema_info}
</database_schema>

<table_descriptions>
- 강좌: 상명대학교 전자공학과(천안) 개설 과목 정보 (과목코드, 과목명, 학과, 학년, 학기, 이수구분, 학점, 요일, 시작시간, 종료시간, 담당교수, 강의실, 정원). 이수구분은 "전공선택" 또는 "전공심화"만 존재한다.
- 수강신청현황: 연도-학기별 과목 신청 인원 (과목코드, 학기, 신청인원). 학기 값은 "2025-1학기"처럼 "연도-학기" 형식이다.
</table_descriptions>

<rules>
- SQLite 문법을 사용하세요
- SELECT 쿼리만 생성하세요 (INSERT, UPDATE, DELETE 금지)
- 결과는 최대 20개로 제한하세요 (LIMIT 20)
- SQL 쿼리만 반환하고, 설명은 포함하지 마세요
- 코드 블록(```)이나 'sql' 키워드 없이 순수 쿼리만 반환하세요
- 경쟁률을 물으면 신청인원 / 정원으로 계산하세요 (강좌와 수강신청현황을 JOIN)
- 존재하지 않는 컬럼을 사용하지 마세요
</rules>"""
        if feedback:
            prompt += f"\n\n이전 시도의 오류:\n{feedback}\n\n위 오류를 고려하여 쿼리를 수정하세요."
        return prompt

    def generate_sql(self, question: str, feedback: str | None = None) -> str:
        messages = [
            SystemMessage(content=self._build_system_prompt(feedback)),
            HumanMessage(content=question),
        ]
        response = self.llm.invoke(messages)
        sql = response.content.strip()

        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1]) if len(lines) > 2 else sql
            sql = sql.replace("sql", "", 1).strip()

        return sql

    def query(self, question: str) -> dict:
        """질문 → SQL 생성 → 검증 → 실행. 검증/실행 실패 시 오류를 피드백으로 재시도."""
        feedback = None

        for attempt in range(MAX_RETRIES + 1):
            sql = self.generate_sql(question, feedback=feedback)

            is_valid, validation_error = validate_sql(sql)
            if not is_valid:
                feedback = f"검증 실패: {validation_error}"
                if attempt == MAX_RETRIES:
                    return {"sql": sql, "result": None, "error": feedback}
                continue

            try:
                result = self.db.run(sql)
                return {"sql": sql, "result": result, "error": None}
            except Exception as e:
                feedback = f"실행 오류: {e}"
                if attempt == MAX_RETRIES:
                    return {"sql": sql, "result": None, "error": feedback}

        return {"sql": sql, "result": None, "error": "알 수 없는 오류"}


def get_text2sql_engine() -> Text2SQLEngine:
    return Text2SQLEngine()
