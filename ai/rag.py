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
    (97, 200, "개설학과별_시간표"),
]

CATEGORY_DESCRIPTIONS = {
    "수강신청_안내": "수강신청 기간, 절차, 방법, 정정기간 등 신청 방법 자체에 대한 안내",
    "학사안내": "학사일정, 등록, 성적, 졸업, 복수전공/부전공 등 전반적인 학사 규정",
    "수강제한_강좌목록": "특정 강좌를 주전공 학생 외에는 수강신청 1일차에 못 듣는 제한 규정/목록",
    "타학과_인정과목": "전자공학과 학생이 타 학과 과목을 전공선택으로 인정받을 수 있는 교과목 목록",
    "개설학과별_시간표": "특정 과목의 담당교수, 강의시간, 강의실, 수강정원 등 시간표 정보",
    "교수진_안내": "전자공학과 교수님 개인 정보(세부전공, 연구실, 연락처)",
}


def _category_for_page(pdf_stem: str, page_num: int) -> str:
    if pdf_stem == "교수진_안내":
        return "교수진_안내"
    for start, end, category in MERGED_PDF_CATEGORIES:
        if start <= page_num <= end:
            return category
    return "기타"


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

    def invoke(self, query: str, category: Optional[str] = None) -> List[Document]:
        search_filter = None
        if category:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.category", match=models.MatchValue(value=category)
                    )
                ]
            )

        child_results = self.vectorstore.similarity_search(
            query, k=self.k, filter=search_filter
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
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text", sort=True)
                if len(text.strip()) < 10:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num + 1,
                            "parent_id": f"{pdf_path.stem}_page_{page_num + 1}",
                            "category": _category_for_page(pdf_path.stem, page_num + 1),
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
