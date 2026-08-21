"""
CSV 데이터 탐색 및 통계 (day3 팀 프로젝트 템플릿 2단계).

scripts/create_db.py로 만든 강좌/수강신청현황/교수진 CSV를 읽어서
행 수, 컬럼, 결측치, 카테고리 분포를 확인한다.
"""

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

CSV_FILES = {
    "강좌": DATA_DIR / "강좌.csv",
    "수강신청현황": DATA_DIR / "수강신청현황.csv",
    "교수진": DATA_DIR / "교수진.csv",
}


def load_all() -> dict[str, pd.DataFrame]:
    dataframes = {}
    for table_name, path in CSV_FILES.items():
        if not path.exists():
            print(f"✗ {table_name} 로드 실패: 파일이 없습니다 ({path})")
            print("  먼저 `python scripts/create_db.py`를 실행하세요.")
            continue
        df = pd.read_csv(path)
        dataframes[table_name] = df

        print("=" * 80)
        print(f"📋 {table_name} 테이블")
        print("=" * 80)
        print(f"\n행 수: {len(df)}")
        print(f"컬럼: {list(df.columns)}")
        print("\n첫 5개 행:")
        print(df.head())
        print("\n데이터 타입:")
        print(df.dtypes)
        print()
    print(f"✓ 총 {len(dataframes)}개의 테이블 로드 완료")
    return dataframes


def explore_stats(dataframes: dict[str, pd.DataFrame]) -> None:
    for table_name, df in dataframes.items():
        print(f"\n{'=' * 80}")
        print(f"📊 {table_name} 통계")
        print("=" * 80)

        print("\n[결측치]")
        null_counts = df.isnull().sum()
        print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "결측치 없음")

        print("\n[카테고리 분포]")
        if table_name == "강좌":
            print("\n1. 학년별 과목 수:")
            print(df["학년"].value_counts().sort_index())
            print("\n2. 이수구분별 과목 수:")
            print(df["이수구분"].value_counts())
            print("\n3. 학기별 과목 수:")
            print(df["학기"].value_counts())
            print("\n4. 담당교수별 과목 수:")
            print(df["담당교수"].value_counts())
        elif table_name == "교수진":
            print("\n1. 세부전공 목록:")
            print(df["세부전공"].value_counts())
            print("\n2. 연구실 위치 분포:")
            print(df["연구실"].value_counts())
        elif table_name == "수강신청현황":
            print("\n1. 학기별 신청 건수:")
            print(df["학기"].value_counts())
            print("\n2. 신청인원 기초 통계:")
            print(df["신청인원"].describe())


if __name__ == "__main__":
    dfs = load_all()
    explore_stats(dfs)
