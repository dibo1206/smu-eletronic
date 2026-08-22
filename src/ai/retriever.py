"""수강신청 챗봇의 RAG 엔진.

rag-system의 04번(Parent Document Retriever) + 05번(메타데이터 필터링)
패턴을 함께 따른다:
1. PDF를 페이지 단위(Parent Document)로 로드, 원본 문서 섹션별로
   category 메타데이터를 붙임
2. 페이지를 작은 child chunk로 분할해 Qdrant Cloud에 저장
   (category는 payload index로 등록해 필터링 가능하게 함)
3. 질문이 들어오면 LLM이 어느 category에 해당하는지 먼저 판단
   (동적 필터링, determine_category)
4. category 필터를 적용해 child chunk 검색 → parent_id로 원본 페이지 반환
5. 반환된 페이지를 컨텍스트로 LLM 답변 생성
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams
from qdrant_client import models

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

DATASETS_DIR = ROOT_DIR / "datasets"
PDF_PATHS = [
    DATASETS_DIR / "2026-2학기_수강신청_공식자료_통합.pdf",  # 학교 공식 자료 5종 병합본
    DATASETS_DIR / "교수진_안내.pdf",
]
COLLECTION_NAME = "sugang_docs"

# 통합본 PDF 안에서 원본 문서(섹션)별 페이지 범위 → 카테고리.
# scripts/merge_official_pdfs.py가 각 원본 앞에 표지 1장을 넣고 이어붙인 순서와 같다.
MERGED_PDF_CATEGORIES = [
    (1, 10, "수강신청_안내"),
    (11, 56, "학사안내"),
    (57, 60, "수강제한_강좌목록"),
    (61, 96, "타학과_인정과목"),
    (97, 157, "개설학과별_시간표"),
    (158, 172, "교양"),  # 아래 GYOYANG_PAGE_RANGE로 과목 단위까지 더 세분화됨
    (173, 200, "개설학과별_시간표"),  # 마이크로전공 등 나머지 시간표
]

# "계당교양교육원"·"전체학과" 표제로 된 교양 시간표 페이지 범위.
# 이 범위는 페이지 단위가 아니라 "교양영역"별로 과목 하나하나를 쪼개서 카테고리를 매긴다.
GYOYANG_PAGE_RANGE = (158, 172)
COURSE_CODE_RE = re.compile(r"HB[A-Z]{2}\d{3,4}")
GYOYANG_AREA_RE = re.compile(r"(기초|균형)\(([^)]*)\)")

# 교양/전공 시간표 행 텍스트에 공통으로 나오는 "월2,3,4(I604)" 같은
# 요일+교시 패턴. 시간 겹침 여부는 LLM의 자연어 추론에 맡기지 않고 이걸로
# 직접 계산한다 — "월 2,3교시"와 "월 2,3,4교시"처럼 숫자가 여러 개 겹치는
# 경우를 LLM이 텍스트만 보고 비교하다가 안 겹친다고 잘못 판단한 적이 있다.
DAY_PERIODS_RE = re.compile(r"([월화수목금토일])((?:\d+,)*\d+)\(")


def _extract_time_slots(text: str) -> set[tuple[str, int]]:
    """텍스트에서 요일+교시 패턴을 모두 찾아 {(요일, 교시), ...} 집합으로
    반환한다. 한 과목이 여러 요일에 열리면(분할 수업 등) 매치가 여러 번
    나오는데 전부 합쳐서 반환한다."""
    slots = set()
    for day, periods in DAY_PERIODS_RE.findall(text):
        for p in periods.split(","):
            slots.add((day, int(p)))
    return slots


TIME_RANGE_RE = re.compile(r"([월화수목금토일])\s*(\d{1,2}):\d{2}(?:~(\d{1,2}):\d{2})?")
# "요일교시" 필드는 우리가 직접 만든 문자열이라 "화6,7,8(I407)"처럼 강의실
# 표시가 안 붙는다 — DAY_PERIODS_RE(뒤에 "(" 필요)와 달리 괄호 없이 매치한다.
BARE_DAY_PERIODS_RE = re.compile(r"([월화수목금토일])((?:\d+,)*\d+)")


def parse_schedule_to_slots(schedule: str) -> set[tuple[str, int]]:
    """"내 시간표" 같은 UI에 저장된 "요일교시" 문자열에서 겹침 판단용
    (요일, 교시) 집합을 뽑는다. 이 문자열은 출처에 따라 두 형식이
    섞여있다 — RAG(PDF)에서 온 건 "화6,7,8"처럼 교시 번호, SQL(강좌 DB)
    에서 온 건 "월 09:00" 또는 "월 09:00~12:00"처럼 실제 시각이라, 둘 다
    처리한다."""
    slots = set()
    time_matches = list(TIME_RANGE_RE.finditer(schedule))
    if time_matches:
        for day, start_h, end_h in (m.groups() for m in time_matches):
            start_period = int(start_h) - 8
            end_period = int(end_h) - 8 - 1 if end_h else start_period
            for p in range(start_period, max(end_period, start_period) + 1):
                slots.add((day, p))
        return slots
    for day, periods in BARE_DAY_PERIODS_RE.findall(schedule):
        for p in periods.split(","):
            slots.add((day, int(p)))
    return slots


# 과목 행 텍스트는 "학점\n이론시간\n실습시간\n" 세 줄이 "균형(...)"/"기초(...)"
# 또는 "요일+교시" 바로 앞에 온다(예: "3\n3\n0\n균형(인문)\n화6,7,8..."). 이
# 숫자만 보고 LLM이 몇 번째 줄이 학점인지 헷갈려서(예: 2학점 과목을 3학점
# 이라고 잘못 답한 적이 있다) 학점을 코드로 직접 뽑아서 명시적으로 알려준다.
CREDIT_RE = re.compile(r"\n(\d+)\n\d+\n\d+\n(?:균형|기초|[월화수목금토일]\d)")


def _extract_credit(text: str) -> Optional[int]:
    """텍스트에서 학점 숫자를 뽑는다. 못 찾으면 None."""
    m = CREDIT_RE.search(text)
    return int(m.group(1)) if m else None


def _extract_course_candidates(docs: List[Document]) -> list[dict]:
    """검색된 시간표 문서(교양/개설학과별)에서 "내 시간표에 담기" UI가
    쓸 수 있는 구조화된 과목 정보를 뽑는다. RAG 답변은 자유 텍스트라
    Streamlit이 체크박스로 보여줄 방법이 없어서, 답변 생성과 별개로
    같은 원문에서 코드로 직접 파싱해 리스트로 반환한다."""
    seen_codes = set()
    candidates = []
    for doc in docs:
        category = str(doc.metadata.get("category", ""))
        if category != "개설학과별_시간표" and not category.startswith("교양_"):
            continue
        text = doc.page_content
        code_match = COURSE_CODE_RE.search(text)
        if not code_match or code_match.group() in seen_codes:
            continue
        seen_codes.add(code_match.group())

        name = text[code_match.end():].split("\n", 1)[0].strip()
        schedule = ", ".join(f"{day}{periods}" for day, periods in DAY_PERIODS_RE.findall(text))
        if not schedule:
            continue

        candidates.append({
            "과목코드": code_match.group(),
            "과목명": name,
            "학점": _extract_credit(text),
            "요일교시": schedule,
            "학과": doc.metadata.get("department"),
            "출처": f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}",
        })
    return candidates


# 상명대 공과대학 학과별 시간표(통합본 143~157페이지, 11개 학과/전공)도
# 페이지 통째가 아니라 과목 단위로 쪼갠다 — 안 그러면 "2학년만" 같은 질문에서
# LLM이 뒤죽박죽인 원본 표 텍스트를 직접 읽고 학년을 스스로 걸러내야 해서
# 틀리기 쉽다 (실제로 1학년 과목을 2학년으로 잘못 답한 적이 있다).
DEPARTMENT_TIMETABLE_PAGES = range(143, 158)

GYOYANG_AREA_DESCRIPTIONS = {
    "교양_기초(사고와표현)": "글쓰기/의사표현 관련 기초교양 과목",
    "교양_기초(영어)": "영어 관련 기초교양 과목",
    "교양_기초(기초수학)": "수학 관련 기초교양 과목",
    "교양_기초(교양과인성)": "인성 관련 기초교양 과목",
    "교양_기초(알고리즘과게임콘텐츠)": "알고리즘/게임콘텐츠 관련 기초교양 과목",
    "교양_균형(인문)": "인문 영역 균형교양 과목",
    "교양_균형(사회)": "사회 영역 균형교양 과목",
    "교양_균형(자연)": "자연 영역 균형교양 과목",
    "교양_균형(예술)": "예술 영역 균형교양 과목",
    "교양_균형(공학)": "공학 영역 균형교양 과목",
    "교양_균형(브리지)": "융합/브리지 영역 균형교양 과목",
}

CATEGORY_DESCRIPTIONS = {
    "수강신청_안내": "수강신청 기간, 절차, 방법, 정정기간 등 신청 방법 자체에 대한 안내",
    "학사안내": "학사일정, 등록, 성적, 졸업, 복수전공/부전공 등 전반적인 학사 규정",
    "수강제한_강좌목록": "특정 강좌를 주전공 학생 외에는 수강신청 1일차에 못 듣는 제한 규정/목록",
    "타학과_인정과목": "공과대학 학생이 다른 학과 과목을 전공선택으로 인정받을 수 있는 교과목 목록",
    "개설학과별_시간표": "상명대 공과대학 11개 학과/전공의 2026-2학기 개설강좌 담당교수, 강의시간, 강의실 등 시간표 정보",
    "교수진_안내": "전자공학과 교수님 개인 정보(세부전공, 연구실, 연락처) — 다른 학과 교수님은 이 정보가 없음",
    **GYOYANG_AREA_DESCRIPTIONS,
}

# 공과대학 개설학과별 시간표(143~157페이지)에 실제로 있는 학과/전공 목록.
# classify_query가 department를 뽑을 때 이 목록과 맞춰본다.
DEPARTMENT_NAMES = [
    "전자공학과", "소프트웨어학과", "스마트정보통신공학과", "경영공학과",
    "그린화학공학과", "건설시스템공학과", "정보보안공학과", "시스템반도체공학과",
    "휴먼지능로봇공학과", "지능형로봇학과", "AI모빌리티공학과",
]

# 학생들이 정식 학과명 대신 쓰는 줄임말 — DEPARTMENT_NAMES(정식 학과명)와
# 매핑해줘야 department 필터가 정확히 걸린다.
DEPARTMENT_ALIASES = {
    "휴먼지능로봇공학과": ["휴지로"],
    "소프트웨어학과": ["솦웨"],
    "시스템반도체공학과": ["시반공"],
    "건설시스템공학과": ["건시공"],
    "그린화학공학과": ["그화공"],
    "경영공학과": ["경공"],
    "스마트정보통신공학과": ["스정통"],
}
DEPARTMENT_ALIASES_TEXT = ", ".join(
    f"{full}({'/'.join(aliases)})" for full, aliases in DEPARTMENT_ALIASES.items()
)


def _category_for_page(pdf_stem: str, page_num: int, page_text: str = "") -> Optional[str]:
    """페이지 하나의 카테고리를 정한다.

    "개설학과별_시간표" 페이지 범위(97~157, 173~200)는 공과대학뿐 아니라
    학교 전체 학과 시간표가 섞여 있다(예술대학, 디자인대학 등). 이 챗봇은
    공과대학 전용이라 다른 단과대학 페이지까지 색인해두면, 검색이 엉뚱한
    학과 시간표를 반환할 위험만 생긴다 (실제로 "연극전공" 시간표가 뽑힌 적이
    있었다). 그래서 이 범위에서는 페이지 표제가 "공과대학"으로 시작하는
    페이지만 색인하고, 나머지는 None을 반환해 아예 건너뛴다.
    """
    if pdf_stem == "교수진_안내":
        return "교수진_안내"
    for start, end, category in MERGED_PDF_CATEGORIES:
        if start <= page_num <= end:
            if category == "개설학과별_시간표" and _department_from_header(page_text) is None:
                return None
            return category
    return "기타"


def _split_gyoyang_rows(page_text: str) -> list[tuple[str, str]]:
    """교양 시간표 페이지 텍스트를 과목(행) 단위로 잘라서
    [(카테고리, 행 텍스트), ...]로 반환한다. 학수번호(HB로 시작하는 코드)가
    한 행에 하나씩만 나오는 걸 이용해서, 코드가 나온 지점부터 다음 코드
    직전까지를 한 과목의 정보로 취급한다."""
    matches = list(COURSE_CODE_RE.finditer(page_text))
    rows = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        block = page_text[start:end].strip()

        area_match = GYOYANG_AREA_RE.search(block)
        if not area_match:
            continue
        area = f"{area_match.group(1)}({area_match.group(2).replace(chr(10), '').strip()})"
        category = f"교양_{area}"
        if category not in GYOYANG_AREA_DESCRIPTIONS:
            continue

        rows.append((category, block))
    return rows


def _department_from_header(page_text: str) -> Optional[str]:
    """페이지 2번째 줄("공과대학XXX")에서 학과명을 뽑는다.
    학과명 없이 "공과대학"만 있으면(143페이지 취업과창업 같은 공통 과목)
    "공과대학 공통"으로 취급한다."""
    lines = [line for line in page_text.split("\n") if line.strip()]
    if len(lines) < 2 or not lines[1].strip().startswith("공과대학"):
        return None
    name = lines[1].strip()[len("공과대학"):].strip()
    return name or "공과대학 공통"


def _split_department_timetable_rows(page_text: str) -> list[tuple[int, str, str]]:
    """공과대학 개설학과별 시간표(143~157페이지)를 과목 단위로 잘라서
    [(학년, 과목명, 행 텍스트), ...]로 반환한다.

    PyMuPDF가 셀 단위로 뽑아내는 이 표는 "No / 학년 / 이수구분 / 학수번호
    +교과목명 / ..." 순서인데, 두 가지 변형이 섞여 있다:
    - 이수구분이 "1전선"처럼 따로 줄이 되는 행: 학년은 코드 줄의 2줄 앞
    - 이수구분이 코드와 한 줄에 붙어버리는 행(예: "1전선HBMA1008..."):
      학년은 코드 줄의 1줄 앞
    그래서 1줄 앞부터 먼저 확인하고, 아니면 2줄 앞을 확인한다.

    교과목명이 길면 줄바꿈되기도 해서(예: "Verilog기반디지털시스템설계(P" /
    "BL)"), 다음 줄이 숫자(학점)가 아닌 동안은 이름의 연속으로 보고 이어붙인다."""
    lines = [line for line in page_text.split("\n") if line.strip()]
    code_line_indices = [i for i, line in enumerate(lines) if COURSE_CODE_RE.search(line)]

    def _grade_and_row_start(code_idx: int) -> tuple[Optional[int], Optional[int]]:
        for back in (1, 2):
            idx = code_idx - back
            if idx < 0:
                continue
            m = re.match(r"^\s*([1-4])\s*$", lines[idx])
            if m:
                return int(m.group(1)), idx - 1  # No는 학년 한 줄 앞
        return None, None

    rows = []
    for pos, i in enumerate(code_line_indices):
        grade, _ = _grade_and_row_start(i)
        if grade is None:
            continue

        code_match = COURSE_CODE_RE.search(lines[i])
        course_name = lines[i][code_match.end():].strip()
        j = i + 1
        while j < len(lines) and not re.match(r"^\d+(\.\d+)?$", lines[j].strip()):
            course_name += lines[j].strip()
            j += 1

        if pos + 1 < len(code_line_indices):
            _, next_row_start = _grade_and_row_start(code_line_indices[pos + 1])
            end = next_row_start if next_row_start is not None else code_line_indices[pos + 1]
        else:
            end = len(lines)
        block = "\n".join(lines[i:max(end, i + 1)])

        rows.append((grade, course_name, block))
    return rows


class CategoryClassification(BaseModel):
    """질문에 가장 적합한 카테고리 분류 결과.

    rag-system 05번 노트북의 "복합 필터링(OR 조건)" 패턴 — 카테고리를
    하나만 강제로 고르게 하면, "전공 시간표 피해서 교양 추천해줘"처럼
    두 주제가 섞인 질문에서 한쪽 정보가 아예 검색조차 안 되는 문제가
    있었다. 리스트로 받아서 Qdrant의 should(OR) 필터로 여러 카테고리를
    한 번에 검색한다."""

    categories: Optional[List[str]] = Field(
        description="선택된 카테고리 이름 목록 (1~3개). 적합한 카테고리가 전혀 없으면 빈 리스트"
    )
    grade: Optional[int] = Field(
        default=None,
        description="질문이 특정 학년(1~4)의 전공 시간표만 콕 집어 물어보면 "
        "그 학년 숫자(예: '2학년 전공 시간표' → 2). 학년을 지정하지 않았거나 "
        "교양처럼 학년과 무관한 질문이면 null.",
    )
    department: Optional[str] = Field(
        default=None,
        description="질문이 공과대학 특정 학과(전자공학과, 소프트웨어학과 등)의 "
        "전공 시간표를 콕 집어 물어보면 그 학과 이름. 학과를 지정하지 않았거나 "
        "교양처럼 학과와 무관한 질문이면 null.",
    )

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 상명대학교 공과대학 학사 안내 전문가입니다.
주어진 정보를 바탕으로 사용자의 질문에 정확하고 친절하게 답변하세요.
답변에 참고한 문서의 이름과 페이지 번호를 함께 밝히세요.

<교시_시각_대응표>
시간표 문서는 "월2,3,4"처럼 요일+교시로만 표기되어 있고 실제 시각은
안 적혀 있습니다. 질문이 "9시", "오전 10시"처럼 실제 시각으로 물어보면
아래 표로 교시와 대응시켜서 답하세요 (1교시당 1시간, N교시 = (8+N)시~(9+N)시):
1교시=9~10시, 2교시=10~11시, 3교시=11~12시, 4교시=12~13시,
5교시=13~14시, 6교시=14~15시, 7교시=15~16시, 8교시=16~17시,
9교시=17~18시, 10교시=18~19시, 11교시=19~20시, 12교시=20~21시,
13교시=21~22시, 14교시=22~23시, 15교시=23~24시.
예: "월2,3,4"는 월요일 10시~13시 수업입니다. 이 대응표를 벗어나는
교시(16교시 이상, 8시 이전)는 문서에 없는 시간대이니 추측하지 마세요.
</교시_시각_대응표>

각 문서 앞에 "[학점: N]"이 붙어 있으면 그 과목의 학점은 반드시 그 숫자를
쓰세요. 본문에 있는 "3\n3\n0"처럼 줄바꿈된 숫자들을 직접 세어 학점을
추측하지 마세요 — 이론/실습 시간과 헷갈려서 틀리기 쉽습니다.

<겹침_판단_주의>
전공 시간표나 "사용자가 이미 담아둔 내 시간표"와 겹치는지 스스로 다시
계산하지 마세요 — 여러 요일/교시가 섞인 문서에서 숫자를 눈으로 비교하다
안 겹치는데 겹친다고(또는 그 반대로) 틀리기 쉽습니다. 컨텍스트에 있는
교양 후보 목록은 이미 코드로 겹침 계산을 끝내고 걸러진 것이므로, 그
목록에 있는 과목은 겹치지 않는다고 그대로 신뢰해서 답하세요. "[참고: ...
겹칩니다]"라는 문구가 명시적으로 있을 때만 겹친다고 판단하세요.
</겹침_판단_주의>

질문이 여러 요청을 한 번에 담고 있어서 그 중 일부만 아래 정보로 답할 수
있다면, 답할 수 있는 부분만 답하고 나머지는 "이 답변에서는 OOO 정보는
확인하지 못했습니다. OOO만 따로 다시 물어봐 주세요"처럼 어떤 부분이
빠졌는지 구체적으로 밝히세요.

주어진 정보에 없는 내용은 추측하지 마세요. 다만 "안내 문서에 해당 정보가
없습니다"처럼 문서 전체에 아예 없다고 단정하지 말고, "이번 검색 결과에서는
확인되지 않았습니다"처럼 이번 조회 범위에서 못 찾았다는 뜻으로 표현하세요
(실제로는 다른 페이지에 있을 수 있습니다).

<context>
{context}
</context>

<question>
{question}
</question>
""",
)


class ParentDocumentRetriever:
    """child chunk로 검색하고 parent(전체 페이지)를 반환한다."""

    def __init__(self, vectorstore, parent_docstore: dict, k: int = 3):
        self.vectorstore = vectorstore
        self.parent_docstore = parent_docstore
        self.k = k

    def _build_filter(
        self,
        categories: Optional[List[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
        grade: Optional[int] = None,
        department: Optional[str] = None,
    ) -> Optional["models.Filter"]:
        """categories가 여러 개면 should(OR)로 묶는다 — rag-system 05번의
        "4-3. 복합 필터링(OR 조건)" 패턴. page_range/grade/department는 각
        카테고리 안에서 must(AND)로 겹쳐 건다 — "4-6. 복합 조건(AND + Range)" 패턴.

        grade/department는 카테고리별 서브 필터 안에만 넣는다 — "개설학과별_시간표"만
        학년(metadata.grade)·학과(metadata.department) 메타데이터가 있고
        교양/교수진 등은 그 필드가 아예 없어서, 전체에 그냥 AND로 걸면 그
        필드가 없는 다른 카테고리 문서가 전부 걸러져버린다 (실제로 이 문제로
        교양 검색 결과가 통째로 사라진 적이 있다)."""
        if not categories:
            return None

        def _category_filter(category: str) -> "models.Filter":
            conditions = [
                models.FieldCondition(
                    key="metadata.category", match=models.MatchValue(value=category)
                )
            ]
            if page_range:
                gte, lte = page_range
                conditions.append(
                    models.FieldCondition(key="metadata.page", range=models.Range(gte=gte, lte=lte))
                )
            if category == "개설학과별_시간표":
                if grade:
                    conditions.append(
                        models.FieldCondition(key="metadata.grade", match=models.MatchValue(value=grade))
                    )
                if department:
                    conditions.append(
                        models.FieldCondition(
                            key="metadata.department", match=models.MatchValue(value=department)
                        )
                    )
            return models.Filter(must=conditions)

        return models.Filter(should=[_category_filter(c) for c in categories])

    def get_child_chunks(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
        grade: Optional[int] = None,
        department: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        """비교용: parent로 확장하지 않고 검색된 child chunk 자체를 반환한다."""
        return self.vectorstore.similarity_search(
            query, k=k or self.k, filter=self._build_filter(categories, page_range, grade, department)
        )

    def invoke(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
        grade: Optional[int] = None,
        department: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        child_results = self.get_child_chunks(
            query, categories=categories, page_range=page_range, grade=grade, department=department, k=k
        )

        parent_ids = []
        for doc in child_results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        return [
            self.parent_docstore[pid]
            for pid in parent_ids
            if pid in self.parent_docstore
        ]


class RagEngine:
    def __init__(self):
        missing = [p for p in PDF_PATHS if not p.exists()]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"안내 문서가 없습니다: {names}\n"
                "먼저 `python scripts/make_pdf.py`와 `python scripts/make_faculty_pdf.py`를 실행하세요."
            )

        self.llm = init_chat_model("gpt-5.4-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60
        )

        docs, child_docs = self._load_and_split()
        self.parent_docstore = {d.metadata["parent_id"]: d for d in docs}
        self._ensure_collection(child_docs)

        vectorstore = QdrantVectorStore(
            client=self.client, collection_name=COLLECTION_NAME, embedding=self.embeddings
        )
        # k를 3→5로 늘림: should(OR) 필터로 카테고리 여러 개를 한 번에 검색할 때
        # 각 카테고리에서 최소한 한두 개씩은 뽑힐 여지를 주기 위해서다.
        self.retriever = ParentDocumentRetriever(vectorstore, self.parent_docstore, k=5)

    def _load_and_split(self):
        import fitz

        docs = []
        for pdf_path in PDF_PATHS:
            doc = fitz.open(pdf_path)
            is_merged_pdf = pdf_path.stem == "2026-2학기_수강신청_공식자료_통합"

            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text", sort=True)
                if len(text.strip()) < 10:
                    continue

                page = page_num + 1
                is_gyoyang_page = is_merged_pdf and GYOYANG_PAGE_RANGE[0] <= page <= GYOYANG_PAGE_RANGE[1]

                if is_gyoyang_page:
                    # sort=True로 정렬하면 이 표는 일부 과목(교양영역이 줄바꿈되는
                    # 경우)의 텍스트 순서가 깨져서 못 찾는 경우가 있어, 원본 순서
                    # 그대로 별도로 다시 읽어서 행을 나눈다.
                    unsorted_text = doc[page_num].get_text("text")
                    rows = _split_gyoyang_rows(unsorted_text)
                    for row_idx, (category, row_text) in enumerate(rows):
                        docs.append(
                            Document(
                                page_content=row_text,
                                metadata={
                                    "source": pdf_path.name,
                                    "page": page,
                                    "parent_id": f"{pdf_path.stem}_page_{page}_row_{row_idx}",
                                    "category": category,
                                },
                            )
                        )
                    if rows:
                        continue
                    # 이 페이지에서 과목을 하나도 못 뽑았으면 페이지 통째로 폴백

                is_department_page = is_merged_pdf and page in DEPARTMENT_TIMETABLE_PAGES
                if is_department_page:
                    unsorted_text = doc[page_num].get_text("text")
                    department = _department_from_header(unsorted_text)
                    dept_rows = _split_department_timetable_rows(unsorted_text) if department else []
                    for row_idx, (grade, course_name, row_text) in enumerate(dept_rows):
                        docs.append(
                            Document(
                                page_content=f"[{department} {grade}학년] {course_name}\n{row_text}",
                                metadata={
                                    "source": pdf_path.name,
                                    "page": page,
                                    "parent_id": f"{pdf_path.stem}_page_{page}_row_{row_idx}",
                                    "department": department,
                                    "category": "개설학과별_시간표",
                                    "grade": grade,
                                },
                            )
                        )
                    if dept_rows:
                        continue
                    # 과목을 하나도 못 뽑았으면 페이지 통째로 폴백

                category = _category_for_page(pdf_path.stem, page, text)
                if category is None:
                    # 공과대학과 무관한 다른 단과대학 시간표 페이지 — 색인하지 않는다
                    continue

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page,
                            "parent_id": f"{pdf_path.stem}_page_{page}",
                            "category": category,
                        },
                    )
                )
            doc.close()

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        child_docs = []
        for parent in docs:
            for chunk in splitter.split_text(parent.page_content):
                child_docs.append(
                    Document(page_content=chunk, metadata=dict(parent.metadata))
                )
        return docs, child_docs

    def _ensure_collection(self, child_docs: list[Document]) -> None:
        """컬렉션이 없으면 새로 만들고 child chunk를 채운다.
        이미 있으면(재실행 시) 그대로 재사용한다 — input() 없이 자동 처리."""
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        # 메타데이터 필터링(category)을 빠르게 하기 위한 페이로드 인덱스
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.category",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        # 페이지 범위(range) 필터용 인덱스 — 카테고리 AND 페이지 범위 복합 조건에 쓴다
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.page",
            field_schema=PayloadSchemaType.INTEGER,
        )
        # 학년(grade) 필터용 인덱스 — 개설학과별_시간표 카테고리 AND 학년 복합 조건에 쓴다
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.grade",
            field_schema=PayloadSchemaType.INTEGER,
        )
        # 학과(department) 필터용 인덱스 — 개설학과별_시간표 카테고리 AND 학과 복합 조건에 쓴다
        self.client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.department",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        vectorstore = QdrantVectorStore(
            client=self.client, collection_name=COLLECTION_NAME, embedding=self.embeddings
        )
        # 청크가 많을 때 한 번에 다 올리면 타임아웃 나기 쉬워서 배치로 나눠 업로드
        batch_size = 100
        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i : i + batch_size]
            vectorstore.add_documents(documents=batch)
            print(f"  업로드 중... {min(i + batch_size, len(child_docs))}/{len(child_docs)}")

    def determine_category(
        self, question: str
    ) -> tuple[Optional[List[str]], Optional[int], Optional[str]]:
        """질문을 보고 어느 카테고리(들)를 검색할지, 그리고 특정 학년/학과로
        더 좁혀야 하는지 LLM으로 동적으로 판단한다. 질문이 한 주제만
        다루면 카테고리 하나, 여러 주제가 섞여 있으면 여러 개를 함께
        돌려준다 (rag-system 05번의 복합 필터링 패턴)."""
        category_list = "\n".join(
            f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()
        )
        department_list = ", ".join(DEPARTMENT_NAMES)
        prompt = f"""다음 질문을 분석하여 검색에 사용할 카테고리를 선택하세요.

<available_categories>
{category_list}
</available_categories>

<engineering_departments>
{department_list}
</engineering_departments>

<department_aliases>
학생들이 정식 학과명 대신 줄임말을 쓰기도 합니다: {DEPARTMENT_ALIASES_TEXT}.
질문에 줄임말이 나오면 department에는 반드시 정식 학과명(위 목록에 있는
이름 그대로)을 넣으세요.
</department_aliases>

<question>
{question}
</question>

<rules>
1. 질문이 한 가지 주제만 다루면 가장 관련 있는 카테고리 하나만 선택하세요.
2. 질문이 서로 다른 주제를 동시에 묻고 있으면(예: "전공 시간표를 피해서
   교양 추천해줘"는 시간표 주제 + 교양 주제) 관련된 카테고리를 모두,
   최대 3개까지 함께 선택하세요.
3. "교양 추천해줘"처럼 구체적인 교양 영역(예술/인문/사회 등)을 지정하지
   않았으면, 관련성 있어 보이는 교양_* 카테고리를 2~3개 함께 선택하세요.
4. 전혀 애매하면 categories를 빈 리스트로 두세요.
5. 질문이 특정 학년(1~4학년)의 전공 시간표를 콕 집어 물어보면 grade에
   그 숫자를 넣으세요. 학년을 지정하지 않았거나 교양 관련 질문이면 grade는
   null로 두세요.
6. 질문이 <engineering_departments> 중 특정 학과의 전공 시간표를 콕 집어
   물어보면 department에 그 학과 이름을 정확히 넣으세요. 학과를 지정하지
   않았거나 교양 관련 질문이면 department는 null로 두세요.
</rules>"""
        structured_llm = self.llm.with_structured_output(CategoryClassification)
        result = structured_llm.invoke(prompt)
        valid = [c for c in (result.categories or []) if c in CATEGORY_DESCRIPTIONS]
        # LLM이 지침(최대 3개)을 안 지키고 더 많이 고를 때가 있어서 코드로 한 번 더 자른다 —
        # 카테고리가 많을수록 공유 k(검색 개수)가 잘게 쪼개져 정작 중요한
        # 카테고리가 밀려날 수 있다. "개설학과별_시간표"는 잘리지 않도록 맨 앞으로 옮겨둔다
        # (뒤에서 별도로 학년/학과 보장 검색도 이 카테고리가 남아있어야 작동한다).
        if "개설학과별_시간표" in valid:
            valid = ["개설학과별_시간표"] + [c for c in valid if c != "개설학과별_시간표"]
        valid = valid[:3]
        department = result.department if result.department in DEPARTMENT_NAMES else None
        return (valid or None), result.grade, department

    def answer(
        self,
        question: str,
        search_query: Optional[str] = None,
        my_timetable: Optional[list[dict]] = None,
    ) -> dict:
        """search_query를 따로 주면 검색(카테고리 판단·벡터 검색)에는 그걸 쓰고,
        답변 생성에는 원래 question을 쓴다 — rewrite_query로 검색어만
        재작성했을 때 최종 답변이 재작성된 문장이 아니라 사용자의 원래
        질문에 답하도록 하기 위해서다.

        my_timetable: Streamlit "내 시간표"에 담아둔 과목 목록. "담은 과목과
        안 겹치게 추천해줘"처럼 전공 시간표가 아니라 사용자가 이미 골라둔
        과목 기준으로 겹침을 피해야 하는 질문에 쓴다."""
        query_for_search = search_query or question
        categories, grade, department = self.determine_category(query_for_search)
        retrieved = self.retriever.invoke(
            query_for_search, categories=categories, grade=grade, department=department
        )

        # 카테고리 분류가 잘못돼서 결과가 하나도 없으면, 필터 없이 한 번 더 검색
        if not retrieved and categories:
            categories = None
            retrieved = self.retriever.invoke(query_for_search, categories=None)

        # 학년/학과까지 지정된 전공 시간표는 다른 카테고리들과 검색 결과
        # 개수(k)를 나눠 갖다 보면 밀려날 수 있어서 (교양 카테고리가 여러 개
        # 섞이면 특히 그렇다), 항상 별도로 한 번 더 가져와서 빠지지 않게 보장한다.
        if (grade or department) and categories and "개설학과별_시간표" in categories:
            narrowed_docs = self.retriever.invoke(
                query_for_search, categories=["개설학과별_시간표"], grade=grade, department=department
            )
            existing_ids = {d.metadata.get("parent_id") for d in retrieved}
            for doc in narrowed_docs:
                if doc.metadata.get("parent_id") not in existing_ids:
                    retrieved.append(doc)
                    existing_ids.add(doc.metadata.get("parent_id"))

        # "전공 시간표 피해서 교양 추천해줘"처럼 전공 시간표와 교양을 함께
        # 찾는 질문은, 실제로 PDF에 있는 교양 과목 중 "겹치지 않는 걸 찾아야"
        # 하는데 공유 k=5로는 교양 후보가 1~2개만 뽑혀서 정말로 존재하는
        # 다른 후보들을 못 보고 "없다"고 오판할 수 있다(실제로 PDF엔 더
        # 있는데 검색에 안 걸린 경우). 이 조합일 때는 교양 카테고리만 훨씬
        # 넉넉하게(k=30) 한 번 더 가져와서 후보 풀을 넓힌다.
        gyoyang_categories = [c for c in (categories or []) if c.startswith("교양_")]
        wants_conflict_check = ("개설학과별_시간표" in (categories or [])) or bool(my_timetable)
        if gyoyang_categories and wants_conflict_check:
            broader_gyoyang_docs = self.retriever.invoke(
                query_for_search, categories=gyoyang_categories, k=30
            )
            existing_ids = {d.metadata.get("parent_id") for d in retrieved}
            for doc in broader_gyoyang_docs:
                if doc.metadata.get("parent_id") not in existing_ids:
                    retrieved.append(doc)
                    existing_ids.add(doc.metadata.get("parent_id"))

        # "전공 시간표 피해서 교양 추천해줘" 같은 질문은 교양 후보와 전공
        # 시간표가 겹치는지를 LLM이 텍스트 보고 스스로 판단하게 두면 틀리기
        # 쉽다("월2,3"과 "월2,3,4"가 겹치는데 안 겹친다고 결론 낸 사례가
        # 있었다). grade/department가 정해져 있어 정확히 "이 학생의 전공
        # 시간표"를 알 수 있을 때는, 겹치는 교시를 코드로 직접 계산해서
        # 겹치는 교양 후보는 아예 컨텍스트에서 빼버린다.
        def _is_dept_timetable(doc: Document) -> bool:
            return doc.metadata.get("category") == "개설학과별_시간표"

        def _is_gyoyang(doc: Document) -> bool:
            return str(doc.metadata.get("category", "")).startswith("교양_")

        dept_docs = [d for d in retrieved if _is_dept_timetable(d)]
        gyoyang_docs = [d for d in retrieved if _is_gyoyang(d)]
        other_docs = [d for d in retrieved if not _is_dept_timetable(d) and not _is_gyoyang(d)]

        conflict_note = None
        occupied = set()
        if grade and department:
            for d in dept_docs:
                occupied |= _extract_time_slots(d.page_content)
        conflict_source = "위 전공 시간표"
        if my_timetable:
            timetable_slots = set()
            for course in my_timetable:
                timetable_slots |= parse_schedule_to_slots(course.get("요일교시", ""))
            if timetable_slots:
                occupied |= timetable_slots
                conflict_source = "이미 담아둔 내 시간표" if not (grade and department) else "전공 시간표/내 시간표"

        if occupied and gyoyang_docs:
            free_gyoyang = [d for d in gyoyang_docs if not (_extract_time_slots(d.page_content) & occupied)]
            if free_gyoyang:
                gyoyang_docs = free_gyoyang
            else:
                # 후보 전부 겹치는 드문 경우 — 걸러내면 후보가 하나도 안
                # 남으므로, 그대로 두되 전부 겹친다는 사실을 프롬프트에
                # 명시해서 LLM이 "안 겹친다"고 오판하지 않게 한다.
                conflict_note = (
                    f"아래 교양 후보 과목은 모두 {conflict_source}와 요일·교시가 "
                    "겹칩니다. 겹치지 않는 대안을 찾지 못했다고 답변하세요."
                )
            retrieved = dept_docs + gyoyang_docs + other_docs

        def _format_doc(doc: Document) -> str:
            header = f"[{doc.metadata.get('source', '?')} - 페이지 {doc.metadata.get('page', '?')}]"
            credit = _extract_credit(doc.page_content)
            if credit is not None:
                header += f" [학점: {credit}]"
            return f"{header}\n{doc.page_content}"

        context_parts = [_format_doc(doc) for doc in retrieved]
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(관련 문서를 찾지 못했습니다)"
        if conflict_note:
            context += f"\n\n---\n\n[참고: {conflict_note}]"
        if my_timetable:
            # 겹치는 후보를 코드로 이미 걸러냈어도, "담은 과목이 뭔지 몰라서
            # 답 못하겠다"고 하지 않도록 실제 담긴 과목 목록을 컨텍스트에
            # 명시한다.
            timetable_text = ", ".join(
                f"{c['과목명']}({c.get('요일교시', '?')})" for c in my_timetable
            )
            context += f"\n\n---\n\n[사용자가 이미 담아둔 내 시간표: {timetable_text}]"

        formatted_prompt = PROMPT.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "category": ", ".join(categories) if categories else None,
            "pages": [
                f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
                for doc in retrieved
            ],
            "courses": _extract_course_candidates(retrieved),
        }


_engine: RagEngine | None = None


def get_rag_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine
