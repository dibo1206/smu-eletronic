"""
CSV 데이터를 Supabase PostgreSQL에 업로드 (day3 팀 프로젝트 템플릿 4단계).

scripts/create_db.py가 만든 CSV(data/*.csv)를 그대로 읽어서 Supabase에
테이블로 올린다. 강좌/교수진(부모 격) 먼저, 수강신청현황(강좌를 참조)
나중에 업로드한다.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"

# 부모 -> 자식 순서 (수강신청현황이 강좌.과목코드를 참조하는 구조)
UPLOAD_ORDER = [
    ("강좌", "강좌.csv"),
    ("교수진", "교수진.csv"),
    ("수강신청현황", "수강신청현황.csv"),
]


def main():
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit("SUPABASE_DB_URL이 .env에 없습니다.")

    engine = create_engine(db_url)

    with engine.connect() as conn:
        for table_name, filename in UPLOAD_ORDER:
            csv_path = DATA_DIR / filename
            if not csv_path.exists():
                print(f"✗ {filename}이 없습니다. 먼저 `python scripts/create_db.py`를 실행하세요.")
                continue

            df = pd.read_csv(csv_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"✓ {table_name} 업로드 완료 ({len(df)}행)")

    print("\n모든 테이블 업로드 완료.")


if __name__ == "__main__":
    main()
