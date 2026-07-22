from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    timestamp: float
    method: str
    path: str
    request: dict
    response: dict
    status_code: int


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)


class LineageDump(BaseModel):
    """One lineage in the fork-mode ``GET /sessions/{id}`` dump."""

    records: list[SessionRecord]
    truncated: bool
    metadata: dict = Field(default_factory=dict)


class ForkedGetSessionResponse(BaseModel):
    """Fork-mode session dump: per-lineage records plus session-level metadata.

    Single-lineage modes keep the flat ``GetSessionResponse`` shape untouched.
    """

    session_id: str
    lineages: list[LineageDump]
    metadata: dict = Field(default_factory=dict)
