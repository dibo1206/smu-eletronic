"""
SQL 쿼리 직접 테스트 (day3 팀 프로젝트 템플릿 6단계).

Text2SQL(LLM)을 거치지 않고, 우리가 직접 작성한 SQL로 기본 조회/JOIN/집계를
테스트한다. LLM이 만드는 SQL이 맞는지 비교할 기준이 되기도 한다.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sugang.db"


def run(con: sqlite3.Connection, title: str, query: str) -> None:
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(query.strip())
    print("\n결과:")
    try:
        for row in con.execute(query):
            print(row)
    except Exception as e:
        print(f"쿼리 실행 오류: {e}")
    print()


def main():
    con = sqlite3.connect(DB_PATH)

    # 1. 기본 조회
    run(
        con,
        "[1] 기본 조회 — 3학년 전공심화 과목",
        """
        SELECT 과목명, 담당교수, 요일, 시작시간, 종료시간
        FROM 강좌
        WHERE 학년 = 3 AND 이수구분 = '전공심화'
        ORDER BY 요일, 시작시간;
        """,
    )

    # 2. JOIN — 강좌 + 교수진
    run(
        con,
        "[2] JOIN — 담당교수의 세부전공까지 같이 조회",
        """
        SELECT c.과목명, c.학년, c.학기, p.이름 AS 담당교수, p.세부전공, p.연구실
        FROM 강좌 c
        JOIN 교수진 p ON c.담당교수 = p.이름
        WHERE c.학기 = '2학기'
        ORDER BY c.학년;
        """,
    )

    # 3. JOIN — 강좌 + 수강신청현황 (경쟁률)
    run(
        con,
        "[3] JOIN — 2026-2학기 경쟁률 상위 5개 과목",
        """
        SELECT c.과목명, c.정원, e.신청인원,
               ROUND(CAST(e.신청인원 AS FLOAT) / c.정원, 2) AS 경쟁률
        FROM 강좌 c
        JOIN 수강신청현황 e ON c.과목코드 = e.과목코드
        WHERE e.학기 = '2026-2학기'
        ORDER BY 경쟁률 DESC
        LIMIT 5;
        """,
    )

    # 4. 집계 — 학년별/이수구분별 과목 수, 학점 합
    run(
        con,
        "[4] 집계 — 학년별 이수구분별 과목 수와 학점 합",
        """
        SELECT 학년, 이수구분, COUNT(*) AS 과목수, SUM(학점) AS 학점합
        FROM 강좌
        GROUP BY 학년, 이수구분
        ORDER BY 학년, 이수구분;
        """,
    )

    # 5. 집계 — 교수별 담당 과목 수 (HAVING)
    run(
        con,
        "[5] 집계 — 담당 과목이 2개 이상인 교수",
        """
        SELECT 담당교수, COUNT(*) AS 담당과목수
        FROM 강좌
        GROUP BY 담당교수
        HAVING COUNT(*) >= 2
        ORDER BY 담당과목수 DESC;
        """,
    )

    con.close()


if __name__ == "__main__":
    main()
