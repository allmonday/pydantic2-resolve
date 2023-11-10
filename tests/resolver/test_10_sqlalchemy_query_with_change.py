import pytest
from typing import List
from collections import Counter, defaultdict
from aiodataloader import DataLoader
from pydantic import ConfigDict, BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from pydantic2_resolve import Resolver, LoaderDepend

@pytest.mark.asyncio
async def test_sqlite_and_dataloader():
    counter = Counter()
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        echo=False,
    )
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    class Base(DeclarativeBase):
        pass

    class Task(Base):
        __tablename__ = "task"

        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str]

    class Comment(Base):
        __tablename__ = "comment"
        id: Mapped[int] = mapped_column(primary_key=True)
        task_id: Mapped[int] = mapped_column()
        content: Mapped[str]

    class Feedback(Base):
        __tablename__ = "feedback"
        id: Mapped[int] = mapped_column(primary_key=True)
        comment_id: Mapped[int] = mapped_column()
        content: Mapped[str]

    async def insert_objects() -> None:
        async with async_session() as session:
            async with session.begin():
                session.add_all(
                    [
                        Task(id=1, name="task-1"),

                        Comment(id=1, task_id=1, content="comment-1 for task 1"),

                        Feedback(id=1, comment_id=1, content="feedback-1 for comment-1"),
                        Feedback(id=2, comment_id=1, content="feedback-2 for comment-1"),
                        Feedback(id=3, comment_id=1, content="feedback-3 for comment-1"),
                    ]
                )

    async def insert_and_update_objects() -> None:
        async with async_session() as session:
            async with session.begin():
                task_1 = (await session.execute(select(Task).filter_by(id=1))).scalar_one()
                task_1.name = 'task-1 xyz'

                comment_1 = (await session.execute(select(Comment).filter_by(id=1))).scalar_one()
                comment_1.content = 'comment-1 for task 1 (changes)'

                feedback_1 = (await session.execute(select(Feedback).filter_by(id=1))).scalar_one()
                feedback_1.content = 'feedback-1 for comment-1 (changes)'

                session.add(task_1)
                session.add(comment_1)
                session.add(feedback_1)
                session.add_all(
                    [
                        Comment(id=2, task_id=1, content="comment-2 for task 1"),

                        Feedback(id=4, comment_id=2, content="test"),
                    ]
                )


    # =========================== Pydantic Schema layer =========================
    class FeedbackLoader(DataLoader):
        async def batch_load_fn(self, comment_ids):
            counter['load-feedback'] += 1
            async with async_session() as session:
                res = await session.execute(select(Feedback).where(Feedback.comment_id.in_(comment_ids)))
                rows = res.scalars().all()
                dct = defaultdict(list)
                for row in rows:
                    dct[row.comment_id].append(FeedbackSchema.model_validate(row))
                return [dct.get(k, []) for k in comment_ids]

    class CommentLoader(DataLoader):
        async def batch_load_fn(self, task_ids):
            counter['load-comment'] += 1
            async with async_session() as session:
                res = await session.execute(select(Comment).where(Comment.task_id.in_(task_ids)))
                rows = res.scalars().all()

                dct = defaultdict(list)
                for row in rows:
                    dct[row.task_id].append(CommentSchema.model_validate(row))
                return [dct.get(k, []) for k in task_ids]

    class FeedbackSchema(BaseModel):
        id: int
        comment_id: int
        content: str
        model_config = ConfigDict(from_attributes=True)

    class CommentSchema(BaseModel):
        id: int
        task_id: int
        content: str
        feedbacks: List[FeedbackSchema]  = []

        def resolve_feedbacks(self, loader=LoaderDepend(FeedbackLoader)):
            return loader.load(self.id)
        model_config = ConfigDict(from_attributes=True)

    class TaskSchema(BaseModel):
        id: int
        name: str
        comments: List[CommentSchema]  = []
        
        def resolve_comments(self, loader=LoaderDepend(CommentLoader)):
            return loader.load(self.id)
        model_config = ConfigDict(from_attributes=True)

    async def init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await insert_objects()

    async def task_query(task_id = 1):
        async with async_session() as session:
            tasks = (await session.execute(select(Task).where(Task.id == task_id))).scalars().all()
            task_objs = [TaskSchema.model_validate(t) for t in tasks]
            resolved_results = await Resolver().resolve(task_objs)
            to_dict_arr = [r.model_dump() for r in resolved_results]
            return to_dict_arr

    await init()
    result_1 = await task_query()
    await insert_and_update_objects()
    result_2 = await task_query()

    expected_1 = [
        {
            'id': 1,
            'name': 'task-1',
            'comments': [
                {
                    'content': 'comment-1 for task 1',
                    'feedbacks': [
                        {'comment_id': 1, 'content': 'feedback-1 for comment-1', 'id': 1},
                        {'comment_id': 1, 'content': 'feedback-2 for comment-1', 'id': 2},
                        {'comment_id': 1, 'content': 'feedback-3 for comment-1', 'id': 3}
                        ],
                    'id': 1,
                    'task_id': 1},
                ],
            }
        ]

    expected_2 = [
        {
            'id': 1,
            'name': 'task-1 xyz',
            'comments': [
                {
                    'content': 'comment-1 for task 1 (changes)',
                    'feedbacks': [
                        {'comment_id': 1, 'content': 'feedback-1 for comment-1 (changes)', 'id': 1},
                        {'comment_id': 1, 'content': 'feedback-2 for comment-1', 'id': 2},
                        {'comment_id': 1, 'content': 'feedback-3 for comment-1', 'id': 3}
                        ],
                    'id': 1,
                    'task_id': 1},
                {
                    'content': 'comment-2 for task 1',
                    'feedbacks': [
                        {'comment_id': 2, 'content': 'test', 'id': 4},
                    ],
                    'id': 2,
                    'task_id': 1}
                ],
            }
        ]

    assert result_1 == expected_1
    assert result_2 == expected_2
    
    assert counter['load-comment'] == 2  # Resolver + LoaderDepend can fix cache issue
    assert counter['load-feedback'] == 2
