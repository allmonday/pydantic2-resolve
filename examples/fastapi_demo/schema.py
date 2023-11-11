from asyncio import Future
from pydantic import BaseModel, ConfigDict
from typing import List
from pydantic2_resolve import LoaderDepend
import fastapi_demo.loader as ld


class FeedbackSchema(BaseModel):
    id: int
    comment_id: int
    content: str
    private: bool

    model_config = ConfigDict(from_attributes=True)

class CommentSchema(BaseModel):
    id: int
    task_id: int
    content: str

    feedbacks: List[FeedbackSchema] = [] 
    def resolve_feedbacks(self, feedback_loader=LoaderDepend(ld.FeedbackLoader)) -> Future[List[FeedbackSchema]]:
        return feedback_loader.load(self.id)

    model_config = ConfigDict(from_attributes=True)

class TaskSchema(BaseModel):
    id: int
    name: str

    comments: List[CommentSchema] = [] 
    def resolve_comments(self, comment_loader=LoaderDepend(ld.CommentLoader)) -> Future[List[CommentSchema]]:
        return comment_loader.load(self.id)

    model_config = ConfigDict(from_attributes=True)
