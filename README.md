# Pydantic-resolve

```python
import asyncio
from pydantic import BaseModel
from pydantic_resolve import resolve

class Student(BaseModel):
    name: str
    greet: str = ''

    async def resolve_greet(self):
        await asyncio.sleep(1)
        return f'hello {self.name}'

async def main():
    students = [Student(name='john' )]
    results = await resolve(students)
    print(results)

# [Student(name='john', greet='hello john')]
```

- `Pydantic-resolve` helps you asynchoronously, resursively resolve a pydantic object (or dataclass object)

- `Pydantic-resolve`, when used in conjunction with aiodataloader, allows you to easily generate nested data structures without worrying about generating N+1 queries.

- Inspired by [GraphQL](https://graphql.org/) and [graphene](https://graphene-python.org/)

[![CI](https://github.com/allmonday/pydantic_resolve/actions/workflows/ci.yml/badge.svg)](https://github.com/allmonday/pydantic_resolve/actions/workflows/ci.yml)
![Python Versions](https://img.shields.io/pypi/pyversions/pydantic-resolve)
![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/allmonday/6f1661c6310e1b31c9a10b0d09d52d11/raw/covbadge.json)

## Install

```shell
pip install pydantic-resolve
```

## Feature-1, Resolve asynchoronously

```python
class NodeB(BaseModel):  # concurrently resolve fields
    value_1: int = 0
    async def resolve_value_1(self):
        await asyncio.sleep(1)  # sleep 1
        return random()
    
    value_2: int = 0
    async def resolve_value_2(self):
        await asyncio.sleep(1)  # sleep 1
        return 12

    value_3: int = 0
    async def resolve_value_3(self):
        await asyncio.sleep(1)  # sleep 1
        return 12

class NodeA(BaseModel):
    node_b_1: int = 0
    def resolve_node_b_1(self):
        return NodeB()

    node_b_2: int = 0
    def resolve_node_b_2(self):
        return NodeB()

class Root(BaseModel):
    node_a_1: int = 0
    def resolve_node_a_1(self):
        return NodeA()

    node_a_2: int = 0
    def resolve_node_a_2(self):
        return NodeA()

async def main():
    t = time()
    root = Root()
    result = await resolve(root)
    print(result.json())
    print(time() - t)

# output
{
    "node_a_1": {
        "node_b_1": {
            "value": 0.7815090210172618,
            "value_2": 12, "value_3": 12
        },
        "node_b_2": {
            "value": 0.22252007296099774,
            "value_2": 12,
            "value_3": 12
        }}, 
    "node_a_2": {
        "node_b_1": {
            "value": 0.30685697832345826,
            "value_2": 12,
            "value_3": 12
        }, 
        "node_b_2": {
            "value": 0.7664967562984117,
            "value_2": 12,
            "value_3": 12
        }
    }
}
# 1.0116631984710693
```

### Feature-2: Integrated with aiodataloader:

`pydantic_resolve.Resolver` will handle the lifecycle and injection of loader instance, you don't need to manage it with contextvars any more.

1. Define loaders

```python
class FeedbackLoader(DataLoader):
    async def batch_load_fn(self, comment_ids):
        async with async_session() as session:
            res = await session.execute(select(Feedback).where(Feedback.comment_id.in_(comment_ids)))
            rows = res.scalars().all()
            dct = defaultdict(list)
            for row in rows:
                dct[row.comment_id].append(FeedbackSchema.from_orm(row))
            return [dct.get(k, []) for k in comment_ids]


class CommentLoader(DataLoader):
    async def batch_load_fn(self, task_ids):
        async with async_session() as session:
            res = await session.execute(select(Comment).where(Comment.task_id.in_(task_ids)))
            rows = res.scalars().all()

            dct = defaultdict(list)
            for row in rows:
                dct[row.task_id].append(CommentSchema.from_orm(row))
            return [dct.get(k, []) for k in task_ids]

```

2. Define schemas

```python
class FeedbackSchema(BaseModel):
    id: int
    comment_id: int
    content: str

    class Config:
        orm_mode = True

class CommentSchema(BaseModel):
    id: int
    task_id: int
    content: str
    feedbacks: Tuple[FeedbackSchema, ...]  = tuple()
    def resolve_feedbacks(self, feedback_loader = LoaderDepend(FeedbackLoader)):  
        # LoaderDepend will manage contextvars for you
        return feedback_loader.load(self.id)

    class Config:
        orm_mode = True

class TaskSchema(BaseModel):
    id: int
    name: str
    comments: Tuple[CommentSchema, ...]  = tuple()
    def resolve_comments(self, comment_loader = LoaderDepend(CommentLoader)):
        return comment_loader.load(self.id)

    class Config:
        orm_mode = True
```

3. Resolve it

```python
tasks = (await session.execute(select(Task))).scalars().all()
tasks = [TaskSchema.from_orm(t) for t in tasks]
results = await Resolver().resolve(tasks)


# output
[
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
                'task_id': 1
            },
            {
                'content': 'comment-2 for task 1',
                'feedbacks': [
                    {'comment_id': 2, 'content': 'test', 'id': 4},
                ],
                'id': 2,
                'task_id': 1
            }
        ]
    }
]

```

For more examples, please explore `examples` folder.

## Unittest

```shell
poetry run python -m unittest  # or
poetry run pytest  # or
poetry run tox
```

## Coverage 

```shell
poetry run coverage run -m pytest
poetry run coverage report -m
```
