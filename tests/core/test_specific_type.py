from __future__ import annotations
from typing import Tuple
from pydantic import BaseModel
from pydantic_resolve.core import scan_and_store_metadata, convert_metadata_key_as_kls
from pydantic_resolve import LoaderDepend

async def loader_fn(keys):
    return keys

class Student(BaseModel):
    zones: Tuple[int, int] = (0, 0)

def test_get_all_fields():
    # https://github.com/allmonday/pydantic2-resolve/issues/7
    result = scan_and_store_metadata(Student)
    expect = {
        'test_specific_type.Student': {
            'resolve': [],
            'post': [],
            'attribute': [],
            'expose_dict': {},
            'collect_dict': {}
            # ... others
        },
    }
    for k, v in result.items():
        assert expect[k].items() <= v.items()
