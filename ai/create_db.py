"""
수강신청 챗봇용 데이터 생성 + SQLite DB 구축 스크립트.

실행하면:
1. 강좌/수강신청현황 데이터를 CSV로 생성 (data/ 폴더)
2. 두 CSV를 SQLite DB(sugang.db)에 테이블로 적재

과목 목록(과목명/학년/학기/이수구분/학점)은 상명대학교 전자공학과(천안)
2026학년도 공식 교육과정(https://www.smu.ac.kr/primeee/admission/curriculum.do)을
그대로 반영했다. 2026학년도 2학기 개설강좌 중 시간이 명확한 과목은
REAL_2ND_SEMESTER_OFFERINGS에 담긴 실제 교수/시간/강의실/정원 값을 쓰고,
나머지(1학기 과목, 분반이 여러 개라 시간이 불명확한 과목)는 데모용으로
임의 생성한 값을 쓴다.
"""

import random
import sqlite3
from pathlib import Path

import pandas as pd

AI_DIR = Path(__file__).parent
DATA_DIR = AI_DIR / "data"
DB_PATH = AI_DIR / "sugang.db"

random.seed(42)

DEPARTMENT = "전자공학과"

PROFESSORS = [
    "김민준", "이서연", "박도윤", "최지우", "정하은",
    "강시우", "조은서", "윤지호", "임수아", "한예준",
]

DAYS = ["월", "화", "수", "목", "금"]

# (과목명, 학년, 학기, 이수구분, 학점) — 상명대 전자공학과 2026학년도 교육과정 원문 그대로
COURSES = [
    # 1학년 — 1학기/2학기 동일하게 개설
    ("회로망론Ⅰ", 1, "1학기", "전공선택", 3),
    ("회로망론II", 1, "1학기", "전공선택", 3),
    ("컴퓨터프로그래밍Ⅰ(SW)", 1, "1학기", "전공선택", 3),
    ("공학수학Ⅰ(PBL)", 1, "1학기", "전공선택", 3),
    ("컴퓨터프로그래밍II", 1, "1학기", "전공선택", 3),
    ("공학수학Ⅱ(PBL)", 1, "1학기", "전공선택", 3),
    ("전공체험(전자공학과)", 1, "1학기", "전공선택", 2),
    ("회로망론Ⅰ", 1, "2학기", "전공선택", 3),
    ("회로망론II", 1, "2학기", "전공선택", 3),
    ("컴퓨터프로그래밍Ⅰ(SW)", 1, "2학기", "전공선택", 3),
    ("공학수학Ⅰ(PBL)", 1, "2학기", "전공선택", 3),
    ("컴퓨터프로그래밍II", 1, "2학기", "전공선택", 3),
    ("공학수학Ⅱ(PBL)", 1, "2학기", "전공선택", 3),
    ("전공체험(전자공학과)", 1, "2학기", "전공선택", 2),
    # 2학년
    ("디지털공학(PBL)", 2, "1학기", "전공심화", 3),
    ("전자기학", 2, "1학기", "전공심화", 3),
    ("기초회로망실험", 2, "1학기", "전공심화", 3),
    ("물리전자개론", 2, "1학기", "전공선택", 3),
    ("VerilogHDL과디지털시스템설계", 2, "2학기", "전공선택", 3),
    ("전자신호와시스템", 2, "2학기", "전공심화", 3),
    ("전자회로Ⅰ", 2, "2학기", "전공심화", 3),
    ("컴퓨터구조(전자공학과)", 2, "2학기", "전공선택", 3),
    # 3학년
    ("임베디드리눅스시스템", 3, "1학기", "전공선택", 3),
    ("마이크로프로세서(전자공학과)", 3, "1학기", "전공선택", 3),
    ("전자디지털영상처리", 3, "1학기", "전공선택", 3),
    ("반도체소자", 3, "1학기", "전공심화", 3),
    ("전자회로II", 3, "1학기", "전공심화", 3),
    ("전자컴퓨터비전", 3, "2학기", "전공선택", 3),
    ("자동제어(PBL)", 3, "2학기", "전공심화", 3),
    ("반도체공정(PBL)", 3, "2학기", "전공심화", 3),
    ("응용전자회로실험", 3, "2학기", "전공심화", 3),
    # 4학년
    ("캡스톤디자인I(전자공학과)", 4, "1학기", "전공심화", 3),
    ("반도체집적회로설계", 4, "1학기", "전공심화", 3),
    ("전자공학세미나", 4, "2학기", "전공선택", 3),
    ("캡스톤디자인II(전자공학과)", 4, "2학기", "전공선택", 3),
    ("반도체시뮬레이션", 4, "2학기", "전공심화", 3),
]


def make_time_slot():
    day = random.choice(DAYS)
    start_hour = random.choice([9, 10, 11, 13, 14, 15, 16])
    end_hour = start_hour + random.choice([1, 2])
    return day, f"{start_hour:02d}:00", f"{end_hour:02d}:00"


# 2026학년도 2학기 실제 개설강좌 정보(학과별강좌조회 화면 기준).
# (담당교수, 요일, 시작시간, 종료시간, 강의실, 수강제한인원) — 과목명 기준으로
# 매칭되는 "2학기" 행에만 덮어쓴다. 여러 분반/시간이 섞여 정확히 알 수 없는
# 과목(컴퓨터프로그래밍Ⅰ/II, 전자신호와시스템)은 임의 생성값을 그대로 둔다.
REAL_2ND_SEMESTER_OFFERINGS = {
    "회로망론Ⅰ": ("이흥주", "화", "13:00", "16:00", "I107", 60),
    "회로망론II": ("이흥주", "화", "16:00", "19:00", "I211", 126),
    "공학수학Ⅰ(PBL)": ("조준희", "수", "09:00", "12:00", "I208", 60),
    "공학수학Ⅱ(PBL)": ("조준희", "월", "09:00", "12:00", "I109", 60),
    "VerilogHDL과디지털시스템설계": ("김선희", "화", "09:00", "12:00", "C317", 24),
    "자동제어(PBL)": ("이유진", "수", "09:00", "12:00", "C317", 48),
    "반도체공정(PBL)": ("조준희", "목", "09:00", "12:00", "I104", 60),
    "응용전자회로실험": ("이흥주", "화", "13:00", "16:00", "C409", 20),
    "전자공학세미나": ("이유진", "수", "13:00", "16:00", "I104", 60),
}


def generate_courses() -> pd.DataFrame:
    rows = []
    code_seq = 1000
    for name, grade, semester, category, credit in COURSES:
        code_seq += 1

        real = REAL_2ND_SEMESTER_OFFERINGS.get(name) if semester == "2학기" else None
        if real:
            professor, day, start, end, room, capacity = real
        else:
            day, start, end = make_time_slot()
            professor = random.choice(PROFESSORS)
            room = f"공학관 {random.randint(101, 509)}호"
            capacity = random.choice([20, 30, 40, 50, 60])

        rows.append(
            {
                "과목코드": f"EE{code_seq}",
                "과목명": name,
                "학과": DEPARTMENT,
                "학년": grade,
                "학기": semester,
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


def generate_enrollment(courses: pd.DataFrame) -> pd.DataFrame:
    """과목이 실제 개설되는 학기(1학기/2학기)에 맞춰 최근 두 개 학년도의
    신청 인원을 생성한다."""
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

    courses.to_csv(DATA_DIR / "강좌.csv", index=False, encoding="utf-8-sig")
    enrollment.to_csv(DATA_DIR / "수강신청현황.csv", index=False, encoding="utf-8-sig")
    print(f"CSV 생성 완료: 강좌 {len(courses)}건, 수강신청현황 {len(enrollment)}건")

    con = sqlite3.connect(DB_PATH)
    courses.to_sql("강좌", con, if_exists="replace", index=False)
    enrollment.to_sql("수강신청현황", con, if_exists="replace", index=False)
    con.close()
    print(f"SQLite DB 생성 완료: {DB_PATH}")


if __name__ == "__main__":
    main()
