# from __future__ import annotations
from pydantic import BaseModel
from pydantic2_resolve.core import _get_class

def test_get_class():
    class Student(BaseModel):
        name: str = 'kikodo'

    stu = Student()
    stus = [Student(), Student()]

    assert _get_class(stu) == Student
    assert _get_class(stus) == Student
