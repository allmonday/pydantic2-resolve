from typing import List, Optional
from pydantic import BaseModel
from pydantic2_resolve import Resolver, mapper, LoaderDepend, ResolverTargetAttrNotFound
import pytest

# define loader functions
async def friends_batch_load_fn(names):
    mock_db = {
        'tangkikodo': ['tom', 'jerry'],
        'john': ['mike', 'wallace'],
        'trump': ['sam', 'jim'],
        'sally': ['sindy', 'lydia'],
    }
    return [mock_db.get(name, []) for name in names]

async def cash_batch_load_fn(names):
    mock_db = {
        'tom': 100, 'jerry':200, 'mike': 3000, 'wallace': 400, 'sam': 500,
        'jim': 600, 'sindy': 700, 'lydia': 800, 'tangkikodo': 900, 'john': 1000,
        'trump': 1200, 'sally': 1300,
    }
    result = []
    for name in names:
        n = mock_db.get(name, None)
        result.append({'number': n} if n else None)  # conver to auto mapping compatible style
    return result

# define schemas
class Cash(BaseModel):
    number: Optional[int] = None

class Friend(BaseModel):
    name: str

    cash: Optional[Cash] = None
    @mapper(Cash)  # auto mapping
    def resolve_cash(self, contact_loader=LoaderDepend(cash_batch_load_fn)):
        return contact_loader.load(self.name)
    
    has_cash: bool = False
    def post_has_cashx(self):
        self.has_cash = self.cash is not None
    

class User(BaseModel):
    name: str
    age: int

    friends: List[Friend] = []
    @mapper(lambda names: [Friend(name=name) for name in names])
    def resolve_friends(self, friend_loader=LoaderDepend(friends_batch_load_fn)):
        return friend_loader.load(self.name)
    
    has_cash: bool = False
    def post_has_cash(self):
        self.has_cash = any([f.has_cash for f in self.friends])

class Root(BaseModel):
    users: List[User] = []
    @mapper(lambda items: [User(**item) for item in items])
    def resolve_users(self):
        return [
            {"name": "tangkikodo", "age": 19}, 
            {"name": "noone", "age": 19}, 
        ]

@pytest.mark.asyncio
async def test_post_methods():
    root = Root()
    with pytest.raises(ResolverTargetAttrNotFound):
        await Resolver().resolve(root)