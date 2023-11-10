from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel
from pydantic_resolve.util import update_forward_refs

class ClassRoom(BaseModel):
    students: List[Student]

class Student(BaseModel):
    id: int
    name: str
    books: List[Book]

class Book(BaseModel):
    name: str


ClassRoom.update_forward_refs()
Student.update_forward_refs()

update_forward_refs(ClassRoom)


print('------------')

print(ClassRoom.model_fields['students'].annotation)
print(ClassRoom.model_fields['students'].type_)
print(ClassRoom.model_fields['students'].outer_type_)

print('------------')

print(Student.model_fields['books'].annotation)
print(Student.model_fields['books'].type_)
print(Student.model_fields['books'].outer_type_)