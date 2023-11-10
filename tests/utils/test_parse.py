from pydantic2_resolve import util
from typing import Optional
from pydantic import ConfigDict, BaseModel, ValidationError
import pytest
from dataclasses import dataclass

def test_pydantic():
    class A(BaseModel):
        name: Optional[str] = None
    
    a = A(name=None)
    value = util.try_parse_data_to_target_field_type(a, 'name', '123')
    assert value == '123'

    value = util.try_parse_data_to_target_field_type(a, 'name', None)
    assert value is None

    with pytest.raises(ValidationError):
        util.try_parse_data_to_target_field_type(a, 'name', [1,2,3])


def test_dataclass():
    @dataclass
    class B:
        age: int

    @dataclass
    class A:
        b: Optional[B] 
    
    a = A(b=None)
    value = util.try_parse_data_to_target_field_type(a, 'b', None)
    assert value is None 

    value = util.try_parse_data_to_target_field_type(a, 'b', {'age': 21})
    assert value == B(age=21)

    with pytest.raises(ValidationError):
        util.try_parse_data_to_target_field_type(a, 'b', [1,2,3])

def test_mix():
    class B(BaseModel):
        age: int

    @dataclass
    class A:
        b: Optional[B] 
    
    a = A(b=None)
    value = util.try_parse_data_to_target_field_type(a, 'b', None)
    assert value is None 

    value = util.try_parse_data_to_target_field_type(a, 'b', {'age': 21})
    assert value == B(age=21)

    with pytest.raises(ValidationError):
        util.try_parse_data_to_target_field_type(a, 'b', [1,2,3])

def test_orm():
    class B(BaseModel):
        age: int
        model_config = ConfigDict(from_attributes=True)

    @dataclass
    class A:
        b: Optional[B] 

    class BB():
        def __init__(self, age: int):
            self.age = age

    
    a = A(b=None)
    value = util.try_parse_data_to_target_field_type(a, 'b', None)
    assert value is None 

    value = util.try_parse_data_to_target_field_type(a, 'b', BB(age=21))
    assert value == B(age=21)

    with pytest.raises(ValidationError):
        util.try_parse_data_to_target_field_type(a, 'b', [1,2,3])
