from pydantic import BaseModel


class SelectRequest(BaseModel):
    user_query: str


class SelectedTool(BaseModel):
    tool: str
    score: float


class SelectResponse(BaseModel):
    tools_selected: list[SelectedTool]
