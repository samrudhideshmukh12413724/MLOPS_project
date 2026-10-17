"""
FastAPI Request & Response Data Validation Schemas for Enterprise RAG Assistant.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the data retention policy for engineering logs?", description="Enterprise knowledge query")
    user_role: str = Field(default="employee", description="User security role: intern, employee, manager, executive, security_admin")
    top_k: int = Field(default=3, ge=1, le=10, description="Top-K context chunks to retrieve")
    retrieval_method: str = Field(default="semantic", description="Retrieval method: tfidf, bm25, or semantic")
    language: str = Field(default="en", description="Language code: en, hi, es, fr, de")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or blank")
        return v.strip()

    @field_validator("user_role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        valid_roles = {"intern", "employee", "manager", "executive", "security_admin", "admin"}
        if v.lower().strip() not in valid_roles:
            raise ValueError(f"Invalid user_role '{v}'. Allowed roles: {sorted(list(valid_roles))}")
        return v.lower().strip()

    @field_validator("retrieval_method")
    @classmethod
    def validate_retrieval_method(cls, v: str) -> str:
        valid_methods = {"tfidf", "bm25", "semantic", "vector"}
        if v.lower().strip() not in valid_methods:
            raise ValueError(f"Invalid retrieval_method '{v}'. Allowed methods: {sorted(list(valid_methods))}")
        return v.lower().strip()


class RetrievedDocumentSchema(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str
    access_level: str


class QueryResponse(BaseModel):
    query: str
    user_role: str
    answer: str
    citations: List[str]
    retrieved_documents: List[RetrievedDocumentSchema]
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    prompt_version: str
    retrieval_version: str
    document_version: str
    application_version: str
    llm_provider: str
    language: str = "en"


class HealthResponse(BaseModel):
    status: str
    chroma_db_status: str
    chunk_count: int
    document_count: int
    version: str


class VersionResponse(BaseModel):
    application_version: str
    retrieval_version: str
    prompt_version: str
    dataset_version: str
    embedding_model: str


class DocumentItemSchema(BaseModel):
    document_id: str
    title: str
    department: str
    category: str
    access_level: str


class DocumentsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentItemSchema]


class AdminActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
