"""
검색 테스트: Child Chunk 검색 vs Parent Document 검색 비교.

rag-system 04번 노트북(섹션 4) 패턴 그대로:
1. Child chunk 직접 검색 — 작은 조각 그대로 반환
2. Parent Document 검색 — 그 조각이 속한 원본(페이지 또는 교양 과목 행) 전체 반환
3. 두 결과의 길이/개수를 비교해서 "정확한 검색 + 풍부한 컨텍스트"라는
   Parent Document Retriever의 효과를 확인한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.rag import get_rag_engine  # noqa: E402


def compare(question: str, category: str | None = None):
    engine = get_rag_engine()
    retriever = engine.retriever

    print("=" * 80)
    print(f"검색 쿼리: {question}")
    if category:
        print(f"카테고리 필터: {category}")
    print()

    # 1. Child Chunk 직접 검색
    child_results = retriever.get_child_chunks(question, category=category, k=3)
    print(f"[1] Child Chunk 검색 결과 — {len(child_results)}개")
    for i, doc in enumerate(child_results, start=1):
        print(f"  {i}. (페이지 {doc.metadata.get('page')}) {doc.page_content[:80]}...")
    print()

    # 2. Parent Document 검색
    parent_results = retriever.invoke(question, category=category)
    print(f"[2] Parent Document 검색 결과 — {len(parent_results)}개")
    for i, doc in enumerate(parent_results, start=1):
        print(f"  {i}. (페이지 {doc.metadata.get('page')}) {doc.page_content[:80]}...")
    print()

    # 3. 비교 분석
    child_lengths = [len(d.page_content) for d in child_results]
    parent_lengths = [len(d.page_content) for d in parent_results]
    print("[3] 비교 분석")
    print(f"  Child  : {len(child_results)}개, 평균 {sum(child_lengths)/len(child_lengths):.0f}자, 총 {sum(child_lengths)}자")
    print(f"  Parent : {len(parent_results)}개, 평균 {sum(parent_lengths)/len(parent_lengths):.0f}자, 총 {sum(parent_lengths)}자")
    if sum(child_lengths):
        print(f"  컨텍스트 크기 비율: {sum(parent_lengths) / sum(child_lengths):.1f}배")
    print()


if __name__ == "__main__":
    compare("졸업하려면 전공 학점 몇 점이야?", category="학사안내")
    compare("전자디지털영상처리 담당교수랑 강의시간 알려줘", category="개설학과별_시간표")
    compare("예술 영역 교양 과목 추천해줘", category="교양_균형(예술)")
