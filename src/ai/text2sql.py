"""수강신청 챗봇의 Text2SQL 엔진.

LangChain의 SQLDatabase로 SQLite에 연결하고, LLM으로 SQL을 생성한다.
LLM이 생성한 SQL은 실행 전에 validate_sql()로 한 번 더 검증한다
(SELECT/CTE만 허용, 테이블 화이트리스트, 위험 키워드 차단) — LLM이
실수해도 코드가 마지막 방어선 역할을 하도록.

쿼리 히스토리(쿼리기록 테이블)에 질문과 생성된 SQL을 저장해두고, 같은
질문이 다시 들어오면 LLM 호출 없이 캐시된 SQL을 그대로 재사용한다.
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_PATH = ROOT_DIR / "sugang.db"

ALLOWED_TABLES = {"강좌", "수강신청현황", "교수진", "진로정보"}

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "CREATE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA",
}

MAX_RETRIES = 2

FEW_SHOT_EXAMPLES = """
<examples>
질문: 2학년 전공심화 과목 알려줘
SQL: SELECT 과목명, 담당교수, 요일, 시작시간 FROM 강좌 WHERE 학년 = 2 AND 이수구분 = '전공심화' LIMIT 20;

질문: 이흥주 교수님 세부전공이랑 담당 과목 알려줘
SQL: SELECT p.세부전공, c.과목명 FROM 교수진 p JOIN 강좌 c ON c.담당교수 = p.이름 WHERE p.이름 = '이흥주' LIMIT 20;

질문: 2026-2학기 경쟁률 제일 높은 과목 3개는?
SQL: SELECT c.과목명, ROUND(CAST(e.신청인원 AS FLOAT) / c.정원, 2) AS 경쟁률 FROM 강좌 c JOIN 수강신청현황 e ON c.과목코드 = e.과목코드 WHERE e.학기 = '2026-2학기' ORDER BY 경쟁률 DESC LIMIT 3;

질문: 담당 과목이 제일 많은 교수는 누구야?
SQL: WITH 과목수 AS (SELECT 담당교수, COUNT(*) AS 건수 FROM 강좌 GROUP BY 담당교수) SELECT 담당교수, 건수 FROM 과목수 ORDER BY 건수 DESC LIMIT 1;

질문: 정보보안공학과 관련 자격증 뭐 있어?
SQL: SELECT 항목 FROM 진로정보 WHERE 학과명 = '정보보안공학과' AND 구분 = '관련자격증' LIMIT 20;

질문: 전자공학과 졸업하면 어떤 회사 가?
SQL: SELECT 항목 FROM 진로정보 WHERE 학과명 = '전자공학과' AND 구분 = '취업분야' LIMIT 20;
</examples>
"""


def validate_sql(sql: str) -> tuple[bool, str]:
    """실행 전 SQL 검증: SELECT 또는 CTE(WITH)로 시작하는 단일 문장만,
    테이블은 화이트리스트만."""
    if not sql or not sql.strip():
        return False, "SQL이 생성되지 않았습니다."

    sql = sql.strip().rstrip(";")
    sql_upper = sql.upper()

    if ";" in sql:
        return False, "여러 SQL 문장은 실행할 수 없습니다."

    if not re.match(r"^\s*(SELECT|WITH)\b", sql_upper):
        return False, "SELECT 문(또는 WITH로 시작하는 CTE)만 사용할 수 있습니다."

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", sql_upper):
            return False, f"{kw} 명령은 사용할 수 없습니다."

    # CTE(WITH 절)로 정의한 임시 이름은 실제 테이블이 아니므로 화이트리스트에
    # 임시로 포함시킨다 (예: "WITH 과목수 AS (...)"의 "과목수").
    cte_names = set(
        re.findall(r"(?:WITH|,)\s*([가-힣a-zA-Z_][가-힣a-zA-Z0-9_]*)\s+AS\s*\(", sql, re.IGNORECASE)
    )

    tables = re.findall(r"\b(?:FROM|JOIN)\s+([가-힣a-zA-Z_][가-힣a-zA-Z0-9_]*)", sql)
    for table in tables:
        if table not in ALLOWED_TABLES and table not in cte_names:
            return False, f"허용되지 않은 테이블입니다: {table}"

    return True, ""


def _ensure_history_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS 쿼리기록 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            질문 TEXT NOT NULL,
            sql TEXT NOT NULL,
            실행시각 TEXT NOT NULL
        )
        """
    )
    con.commit()


class Text2SQLEngine:
    def __init__(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"DB가 없습니다: {DB_PATH}\n먼저 `python scripts/create_db.py`를 실행하세요."
            )
        self.db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        self.llm = init_chat_model("gpt-5.4-mini")
        self.schema_info = self.db.get_table_info()

        self._history_con = sqlite3.connect(DB_PATH, check_same_thread=False)
        _ensure_history_table(self._history_con)

    def _build_system_prompt(self, feedback: str | None) -> str:
        prompt = f"""당신은 SQLite 전문가입니다.
사용자의 질문을 정확한 SQL 쿼리로 변환하세요.

<database_schema>
{self.schema_info}
</database_schema>

<table_descriptions>
- 강좌: 상명대학교 공과대학(천안) 11개 학과/전공 개설 과목 정보 (과목코드, 과목명, 학과, 학년, 학기, 이수구분, 학점, 요일, 시작시간, 종료시간, 담당교수, 강의실, 정원). 이수구분은 "전공선택" 또는 "전공심화"만 존재한다. 정원은 일부 과목(전자공학과)만 값이 있고 나머지는 NULL이다.
- 수강신청현황: 연도-학기별 과목 신청 인원 (과목코드, 학기, 신청인원). 학기 값은 "2025-1학기"처럼 "연도-학기" 형식이다. 정원이 있는 과목(전자공학과)만 들어있다.
- 교수진: 공과대학 11개 학과 교수 개인 정보 (이름, 학위, 학과, 세부전공, 연락처, 연구실). 강좌 테이블의 담당교수와 이름으로 JOIN할 수 있다. 세부전공/연락처가 원본 자료에 없는 교수는 NULL이다.
- 진로정보: 공과대학 11개 학과별 취업분야/진출직업/관련자격증/비고 (학과명, 구분, 순번, 항목). 구분 값은 "취업분야", "진출직업", "관련자격증", "비고" 중 하나이고, 항목이 실제 값이다. 자격증·취업·진로·회사 관련 질문은 이 테이블을 조회하세요.
</table_descriptions>

<rules>
- SQLite 문법을 사용하세요
- SELECT 쿼리, 또는 WITH로 시작하는 CTE(SELECT로 끝나는)를 생성하세요 (INSERT, UPDATE, DELETE 금지)
- 결과는 최대 20개로 제한하세요 (LIMIT 20)
- SQL 쿼리만 반환하고, 설명은 포함하지 마세요
- 코드 블록(```)이나 'sql' 키워드 없이 순수 쿼리만 반환하세요
- 경쟁률을 물으면 신청인원 / 정원으로 계산하세요 (강좌와 수강신청현황을 JOIN)
- 교수님의 세부전공/연락처/연구실을 물으면 교수진 테이블을 조회하세요.
  담당 과목과 함께 묻는 경우에만 강좌.담당교수 = 교수진.이름 으로 JOIN하세요.
- "제일 많은/적은 N은?"처럼 집계 후 순위를 매기는 질문은 서브쿼리나 CTE(WITH)를 활용하세요.
- 존재하지 않는 컬럼을 사용하지 마세요
</rules>

{FEW_SHOT_EXAMPLES}"""
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

    def _lookup_history(self, question: str) -> str | None:
        row = self._history_con.execute(
            "SELECT sql FROM 쿼리기록 WHERE 질문 = ? ORDER BY id DESC LIMIT 1",
            (question.strip(),),
        ).fetchone()
        return row[0] if row else None

    def _save_history(self, question: str, sql: str) -> None:
        self._history_con.execute(
            "INSERT INTO 쿼리기록 (질문, sql, 실행시각) VALUES (?, ?, ?)",
            (question.strip(), sql, datetime.now().isoformat(timespec="seconds")),
        )
        self._history_con.commit()

    def _fetch_dataframe(self, sql: str) -> pd.DataFrame:
        """결과를 표(DataFrame) 형태로도 볼 수 있게 직접 실행한다."""
        con = sqlite3.connect(DB_PATH)
        try:
            return pd.read_sql_query(sql, con)
        finally:
            con.close()

    def query(self, question: str, use_cache: bool = True, extra_feedback: str | None = None) -> dict:
        """질문 → SQL 생성(또는 캐시 재사용) → 검증 → 실행.
        검증/실행 실패 시 오류를 피드백으로 재시도.

        extra_feedback: 그래프 레벨에서 "이전 SQL은 실행됐지만 질문에
        실제로 답하지 못했다"처럼 의미적으로 재시도해야 할 때 넘겨주는
        힌트. 검증/실행 오류 피드백과 같은 프롬프트 자리에 얹혀서
        LLM이 처음부터 다른 접근을 시도하게 만든다."""
        cached_sql = self._lookup_history(question) if use_cache else None
        feedback = extra_feedback

        if cached_sql:
            is_valid, validation_error = validate_sql(cached_sql)
            if is_valid:
                try:
                    result = self.db.run(cached_sql)
                    df = self._fetch_dataframe(cached_sql)
                    return {
                        "sql": cached_sql, "result": result, "error": None,
                        "dataframe": df, "from_cache": True,
                    }
                except Exception:
                    pass  # 캐시된 SQL이 더는 안 맞으면 새로 생성

        for attempt in range(MAX_RETRIES + 1):
            sql = self.generate_sql(question, feedback=feedback)

            is_valid, validation_error = validate_sql(sql)
            if not is_valid:
                feedback = f"검증 실패: {validation_error}"
                if attempt == MAX_RETRIES:
                    return {"sql": sql, "result": None, "error": feedback, "dataframe": None, "from_cache": False}
                continue

            try:
                result = self.db.run(sql)
                df = self._fetch_dataframe(sql)
                self._save_history(question, sql)
                return {
                    "sql": sql, "result": result, "error": None,
                    "dataframe": df, "from_cache": False,
                }
            except Exception as e:
                feedback = f"실행 오류: {e}"
                if attempt == MAX_RETRIES:
                    return {"sql": sql, "result": None, "error": feedback, "dataframe": None, "from_cache": False}

        return {"sql": sql, "result": None, "error": "알 수 없는 오류", "dataframe": None, "from_cache": False}


def get_text2sql_engine() -> Text2SQLEngine:
    return Text2SQLEngine()
