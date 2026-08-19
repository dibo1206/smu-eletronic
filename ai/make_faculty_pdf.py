"""
RAG용 문서(전자공학과 교수진 안내) PDF 생성 스크립트.

상명대학교 전자공학과 교수소개 페이지(https://primeee.smu.ac.kr) 내용을
그대로 옮겨 별도 PDF로 만든다. 교수 1명 = 1페이지로 구성해서
Parent Document Retriever가 "OO 교수님 연구실이 어디야?" 같은 질문에
그 교수 페이지만 정확히 반환하도록 했다.
"""

from pathlib import Path

import fitz

AI_DIR = Path(__file__).parent
OUT_PATH = AI_DIR / "datasets" / "교수진_안내.pdf"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")

# (이름, 학위, 세부전공, 연구실, 연락처)
FACULTY = [
    ("이흥주", "박사", "전자공학", "한누리관 (I608)", "041-550-5360"),
    ("정민철", "박사", "컴퓨터비전, 인공지능", "한누리관 (I611)", "041-550-5361"),
    ("이준하", "박사", "반도체재료 및 공정", "한누리관 (I607)", "041-550-5362"),
    ("이유진", "박사", "전자파적합성", "한누리관 (I606)", "041-550-5413"),
    ("조준희", "박사", "Optoelectronics / Energy Nanomaterials", "한누리관 (I610)", "041-550-5134"),
]


def build_pdf():
    OUT_PATH.parent.mkdir(exist_ok=True)
    doc = fitz.open()

    for name, degree, major, office, phone in FACULTY:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_font(fontname="malgun", fontfile=str(FONT_PATH))

        page.insert_text((72, 80), f"{name} 교수", fontname="malgun", fontsize=20)
        page.draw_line((72, 95), (523, 95), width=1)

        body = f"""소속: 상명대학교 전자공학과(천안)
학위: {degree}
세부전공: {major}
연구실: {office}
연락처: {phone}
"""
        page.insert_textbox(
            fitz.Rect(72, 120, 523, 780),
            body.strip(),
            fontname="malgun",
            fontsize=13,
            lineheight=1.8,
        )

    doc.save(OUT_PATH)
    doc.close()
    print(f"PDF 생성 완료: {OUT_PATH} ({len(FACULTY)}페이지)")


if __name__ == "__main__":
    build_pdf()
