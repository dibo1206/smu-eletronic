"""
수강신청 챗봇용 데이터 생성 + SQLite DB 구축 스크립트.

실행하면:
1. 강좌/수강신청현황/교수진 데이터를 CSV로 생성 (data/ 폴더)
2. 세 CSV를 SQLite DB(sugang.db)에 테이블로 적재

강좌 데이터는 학교 공식 문서(datasets/2026-2학기_수강신청_공식자료_통합.pdf,
"5. 2026-2학기 개설학과별 시간표" 144페이지 "공과대학 전자공학과" 표)에
실제로 나오는 과목만 담았다. 전자공학과 커리큘럼에는 더 많은 과목이 있지만,
이번 학기(2026-2학기)에 실제로 개설된다고 공식 문서로 확인되는 과목만
남기고 나머지는 뺐다 — 담당교수를 임의로 지어내지 않기 위해서다.
분반이 여러 개인 과목(컴퓨터프로그래밍II, 공학수학Ⅱ, 전자신호와시스템,
반도체공정)은 1분반 정보만 담았다. 수강 정원은 공식 문서에 없는 값이라
기존에 조사해둔 실제 수강신청 화면 기준 값을 그대로 썼다.
"""

import random
import sqlite3
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "sugang.db"

random.seed(42)

DEPARTMENT = "전자공학과"

# 전자공학과 교수진 정보 — 학교 교수소개 페이지 그대로.
# scripts/make_faculty_pdf.py의 FACULTY와 같은 원본 데이터를 SQL로도 조회할 수 있게
# 별도 테이블로 만든다 (팀원 노트북의 professors 테이블 구조 참고).
PROFESSOR_INFO = [
    # (이름, 학위, 세부전공, 연락처, 연구실)
    ("이흥주", "박사", "전자공학", "041-550-5360", "한누리관 (I608)"),
    ("정민철", "박사", "컴퓨터비전, 인공지능", "041-550-5361", "한누리관 (I611)"),
    ("이준하", "박사", "반도체재료 및 공정", "041-550-5362", "한누리관 (I607)"),
    ("이유진", "박사", "전자파적합성", "041-550-5413", "한누리관 (I606)"),
    ("조준희", "박사", "Optoelectronics / Energy Nanomaterials", "041-550-5134", "한누리관 (I610)"),
]

# 2026학년도 2학기 전자공학과 실제 개설강좌 — 공식 시간표 PDF 그대로.
# (과목명, 학년, 이수구분, 학점, 요일, 시작시간, 종료시간, 강의실, 담당교수, 정원)
REAL_COURSES = [
    ("회로망론Ⅰ", 1, "전공선택", 3, "월", "13:00", "16:00", "I107", "이흥주", 60),
    ("회로망론II", 1, "전공선택", 3, "화", "16:00", "19:00", "I211", "이흥주", 126),
    ("컴퓨터프로그래밍Ⅰ(SW)", 1, "전공선택", 3, "월", "16:00", "17:00", "C317", "정민철", 20),
    ("공학수학Ⅰ(PBL)", 1, "전공선택", 3, "수", "09:00", "12:00", "I208", "조준희", 60),
    ("컴퓨터프로그래밍II", 1, "전공선택", 3, "월", "17:00", "18:00", "C317", "정민철", 20),
    ("공학수학Ⅱ(PBL)", 1, "전공선택", 3, "월", "09:00", "12:00", "I109", "조준희", 60),
    ("VerilogHDL과디지털시스템설계", 2, "전공선택", 3, "화", "09:00", "12:00", "C317", "김선희", 24),
    ("전자신호와시스템", 2, "전공심화", 3, "월", "14:00", "15:00", "C317", "정민철", 24),
    ("자동제어(PBL)", 3, "전공심화", 3, "수", "09:00", "12:00", "C317", "이유진", 48),
    ("반도체공정(PBL)", 3, "전공심화", 3, "목", "09:00", "12:00", "I211", "조준희", 60),
    ("응용전자회로실험", 3, "전공심화", 3, "화", "13:00", "16:00", "C409", "이흥주", 20),
    ("전자공학세미나", 4, "전공선택", 3, "수", "13:00", "16:00", "I104", "이유진", 60),
]


def generate_courses() -> pd.DataFrame:
    rows = []
    code_seq = 1000
    for name, grade, category, credit, day, start, end, room, professor, capacity in REAL_COURSES:
        code_seq += 1
        rows.append(
            {
                "과목코드": f"EE{code_seq}",
                "과목명": name,
                "학과": DEPARTMENT,
                "학년": grade,
                "학기": "2학기",
                "이수구분": category,
                "학점": credit,
                "요일": day,
                "시작시간": start,
                "종료시간": end,
                "담당교수": professor,
                "강의실": room,
                "정원": capacity,
            }
        )
    return pd.DataFrame(rows)


def generate_professors() -> pd.DataFrame:
    rows = [
        {
            "이름": name,
            "학위": degree,
            "학과": DEPARTMENT,
            "세부전공": major,
            "연락처": phone,
            "연구실": office,
        }
        for name, degree, major, phone, office in PROFESSOR_INFO
    ]
    return pd.DataFrame(rows)


def generate_enrollment(courses: pd.DataFrame) -> pd.DataFrame:
    """신청 인원(경쟁률 계산용)은 학교에서 공개하는 자료가 없어 데모용으로
    생성한다 — 담당교수와 달리 특정인의 신원과 무관한 통계값이라 실제
    데이터가 없어도 임의 생성이 문제되지 않는다. 정원 기준으로 최근 두
    학년도치를 만든다."""
    rows = []
    for _, course in courses.iterrows():
        capacity = course["정원"]
        for year in [2025, 2026]:
            semester_label = f"{year}-{course['학기']}"
            # 정원보다 적을 수도, 초과(경쟁률>1)일 수도 있게 랜덤 분포
            applied = int(capacity * random.uniform(0.4, 1.6))
            rows.append(
                {
                    "과목코드": course["과목코드"],
                    "학기": semester_label,
                    "신청인원": applied,
                }
            )
    return pd.DataFrame(rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    courses = generate_courses()
    enrollment = generate_enrollment(courses)
    professors = generate_professors()

    courses.to_csv(DATA_DIR / "강좌.csv", index=False, encoding="utf-8-sig")
    enrollment.to_csv(DATA_DIR / "수강신청현황.csv", index=False, encoding="utf-8-sig")
    professors.to_csv(DATA_DIR / "교수진.csv", index=False, encoding="utf-8-sig")
    print(
        f"CSV 생성 완료: 강좌 {len(courses)}건, 수강신청현황 {len(enrollment)}건, "
        f"교수진 {len(professors)}건"
    )

    con = sqlite3.connect(DB_PATH)
    courses.to_sql("강좌", con, if_exists="replace", index=False)
    enrollment.to_sql("수강신청현황", con, if_exists="replace", index=False)
    professors.to_sql("교수진", con, if_exists="replace", index=False)
    con.close()
    print(f"SQLite DB 생성 완료: {DB_PATH}")


if __name__ == "__main__":
    main()
