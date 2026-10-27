from pydantic import BaseModel, Field, HttpUrl
from typing import Literal
class VerifyRequest(BaseModel):
    text: str = Field(min_length=3, max_length=12000)
    url: HttpUrl | None = None
    source_text: str = Field(default="", max_length=20000)
class StudentProfile(BaseModel):
    degree: str = Field(min_length=1, max_length=100)
    study_level: str = Field(min_length=1, max_length=60)
    year_of_study: int | None = Field(default=None, ge=1, le=10)
    branch: str = Field(default="", max_length=100)
    graduation_year: int | None = Field(default=None, ge=2020, le=2040)
    location: str = Field(default="India", max_length=100)
    opportunity_type: Literal["scholarship", "internship", "both"] = "both"
class SearchRequest(StudentProfile):
    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=8, ge=1, le=10)
class Source(BaseModel):
    url: str
    title: str | None = None
    retrieved_at: str
    status: str
    text: str = ""
class Claim(BaseModel):
    claim: str
    status: Literal["supports", "conflicts", "insufficient"]
    source_url: str | None = None
    evidence_quote: str | None = None
    uncertainty: str = ""
class VerifyResponse(BaseModel):
    report_id: str
    evidence_status: Literal["supports", "conflicts", "insufficient"]
    summary: str
    claims: list[Claim]
    sources: list[Source]
    risk_signals: list[str]
    model_signal: dict | None = None
    limitations: list[str]
    retrieved_at: str
    live: bool

