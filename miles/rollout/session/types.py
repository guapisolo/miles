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


class SegmentDump(BaseModel):
    """One segment in the fork-mode ``GET /sessions/{id}`` dump."""

    records: list[SessionRecord]
    truncated: bool
    metadata: dict = Field(default_factory=dict)


class ForkedGetSessionResponse(BaseModel):
    """Fork-mode session dump: per-segment records plus session-level metadata.

    Single-segment modes keep the flat ``GetSessionResponse`` shape untouched.
    """

    session_id: str
    segments: list[SegmentDump]
    metadata: dict = Field(default_factory=dict)
