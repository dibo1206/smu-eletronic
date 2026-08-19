"""
수강신청 챗봇의 RAG 엔진.

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

load_dotenv(Path(__file__).parent.parent / ".env")

DATASETS_DIR = Path(__file__).parent / "datasets"
PDF_PATHS = [
    DATASETS_DIR / "2026-2학기_수강신청_공식자료_통합.pdf",  # 학교 공식 자료 5종 병합본
    DATASETS_DIR / "교수진_안내.pdf",
]
COLLECTION_NAME = "sugang_docs"

# 통합본 PDF 안에서 원본 문서(섹션)별 페이지 범위 → 카테고리.
# merge_official_pdfs.py가 각 원본 앞에 표지 1장을 넣고 이어붙인 순서와 같다.
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
    "타학과_인정과목": "전자공학과 학생이 타 학과 과목을 전공선택으로 인정받을 수 있는 교과목 목록",
    "개설학과별_시간표": "특정 전공 과목의 담당교수, 강의시간, 강의실, 수강정원 등 시간표 정보",
    "교수진_안내": "전자공학과 교수님 개인 정보(세부전공, 연구실, 연락처)",
    **GYOYANG_AREA_DESCRIPTIONS,
}


def _category_for_page(pdf_stem: str, page_num: int) -> str:
    if pdf_stem == "교수진_안내":
        return "교수진_안내"
    for start, end, category in MERGED_PDF_CATEGORIES:
        if start <= page_num <= end:
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


class CategoryClassification(BaseModel):
    """질문에 가장 적합한 카테고리 분류 결과."""

    category: Optional[str] = Field(
        description="선택된 카테고리 이름. 여러 카테고리에 걸치거나 애매하면 None"
    )

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 상명대학교 전자공학과 학사 안내 전문가입니다.
주어진 정보를 바탕으로 사용자의 질문에 정확하고 친절하게 답변하세요.
답변에 참고한 문서의 이름과 페이지 번호를 함께 밝히세요.
주어진 정보에 없는 내용은 추측하지 말고 "안내 문서에 해당 정보가 없습니다"라고 답하세요.

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

    def _build_filter(self, category: Optional[str]) -> Optional["models.Filter"]:
        if not category:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.category", match=models.MatchValue(value=category)
                )
            ]
        )

    def get_child_chunks(
        self, query: str, category: Optional[str] = None, k: Optional[int] = None
    ) -> List[Document]:
        """비교용: parent로 확장하지 않고 검색된 child chunk 자체를 반환한다."""
        return self.vectorstore.similarity_search(
            query, k=k or self.k, filter=self._build_filter(category)
        )

    def invoke(self, query: str, category: Optional[str] = None) -> List[Document]:
        child_results = self.get_child_chunks(query, category=category)

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
                "먼저 `python ai/make_pdf.py`와 `python ai/make_faculty_pdf.py`를 실행하세요."
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
        self.retriever = ParentDocumentRetriever(vectorstore, self.parent_docstore, k=3)

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

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page,
                            "parent_id": f"{pdf_path.stem}_page_{page}",
                            "category": _category_for_page(pdf_path.stem, page),
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
        vectorstore = QdrantVectorStore(
            client=self.client, collection_name=COLLECTION_NAME, embedding=self.embeddings
        )
        # 청크가 많을 때 한 번에 다 올리면 타임아웃 나기 쉬워서 배치로 나눠 업로드
        batch_size = 100
        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i : i + batch_size]
            vectorstore.add_documents(documents=batch)
            print(f"  업로드 중... {min(i + batch_size, len(child_docs))}/{len(child_docs)}")

    def determine_category(self, question: str) -> Optional[str]:
        """질문을 보고 어느 카테고리를 검색할지 LLM으로 동적으로 판단한다.
        애매하거나 여러 카테고리에 걸치면 None(필터 없음)을 반환한다."""
        category_list = "\n".join(
            f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()
        )
        prompt = f"""다음 질문을 분석하여 가장 적합한 카테고리를 선택하세요.

<available_categories>
{category_list}
</available_categories>

<question>
{question}
</question>

<rules>
1. 질문과 가장 관련 있는 카테고리 하나만 선택하세요.
2. 여러 카테고리에 걸치거나 애매하면 category를 null로 두세요.
</rules>"""
        structured_llm = self.llm.with_structured_output(CategoryClassification)
        result = structured_llm.invoke(prompt)
        if result.category in CATEGORY_DESCRIPTIONS:
            return result.category
        return None

    def answer(self, question: str) -> dict:
        category = self.determine_category(question)
        retrieved = self.retriever.invoke(question, category=category)

        # 카테고리 분류가 잘못돼서 결과가 하나도 없으면, 필터 없이 한 번 더 검색
        if not retrieved and category:
            category = None
            retrieved = self.retriever.invoke(question, category=None)

        context_parts = [
            f"[{doc.metadata.get('source', '?')} - 페이지 {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in retrieved
        ]
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(관련 문서를 찾지 못했습니다)"

        formatted_prompt = PROMPT.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "category": category,
            "pages": [
                f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
                for doc in retrieved
            ],
        }


_engine: RagEngine | None = None


def get_rag_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine
