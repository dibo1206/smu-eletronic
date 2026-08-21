"""
학교에서 받은 실제 공식 PDF 5개를 하나의 PDF로 합치는 스크립트.

각 원본 PDF 앞에 "어떤 자료인지" 표지 페이지를 한 장씩 넣어서,
Parent Document Retriever가 페이지 단위로 청킹/검색할 때 어느 문서에서
온 내용인지 맥락을 알 수 있게 했다. 원본 페이지는 그대로 복사만 하고
내용은 손대지 않는다(직접 작성한 게 아니라 학교 원본 그대로).
"""

from pathlib import Path

import fitz

DOWNLOADS = Path("C:/Users/dibo1/Downloads")
ROOT_DIR = Path(__file__).parent.parent
OUT_PATH = ROOT_DIR / "datasets" / "2026-2학기_수강신청_공식자료_통합.pdf"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")

# (원본 파일명, 표지에 쓸 제목)
SOURCE_FILES = [
    ("1._2026-2학기_수강신청_안내자료 (1).pdf", "1. 2026-2학기 수강신청 안내자료"),
    ("2026-2학기_학사안내_자료 (3).pdf", "2. 2026-2학기 학사안내 자료"),
    (
        "2._2026-2학기_수강신청_1일차_주전공_학생_외_수강제한_강좌_목록.pdf",
        "3. 2026-2학기 수강신청 1일차 주전공 학생 외 수강제한 강좌 목록",
    ),
    (
        "[부록1]_2026학년도_제2학기_타학과_전공선택_인정_교과목_20260803.pdf",
        "4. [부록1] 2026학년도 제2학기 타학과 전공선택 인정 교과목",
    ),
    (
        "3._2026-2학기_개설학과별_시간표_2026.8.13._기준.pdf",
        "5. 2026-2학기 개설학과별 시간표 (2026.8.13. 기준)",
    ),
]


def add_cover_page(doc: "fitz.Document", title: str) -> None:
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="malgun", fontfile=str(FONT_PATH))
    page.insert_textbox(
        fitz.Rect(72, 350, 523, 500),
        title,
        fontname="malgun",
        fontsize=18,
        lineheight=1.6,
        align=fitz.TEXT_ALIGN_CENTER,
    )


def merge():
    out_doc = fitz.open()

    for filename, title in SOURCE_FILES:
        src_path = DOWNLOADS / filename
        if not src_path.exists():
            print(f"⚠ 파일을 찾지 못했습니다: {src_path}")
            continue

        add_cover_page(out_doc, title)

        src_doc = fitz.open(src_path)
        page_count = src_doc.page_count
        out_doc.insert_pdf(src_doc)
        src_doc.close()
        print(f"병합 완료: {filename} ({page_count}페이지)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    out_doc.save(OUT_PATH)
    print(f"\n최종 PDF 생성 완료: {OUT_PATH} (총 {out_doc.page_count}페이지)")
    out_doc.close()


if __name__ == "__main__":
    merge()
