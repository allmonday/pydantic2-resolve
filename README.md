[![pypi](https://img.shields.io/pypi/v/pydantic2-resolve.svg)](https://pypi.python.org/pypi/pydantic2-resolve)
[![Downloads](https://static.pepy.tech/personalized-badge/pydantic2-resolve?period=month&units=abbreviation&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pydantic2-resolve)
![Python Versions](https://img.shields.io/pypi/pyversions/pydantic2-resolve)
[![CI](https://github.com/allmonday/pydantic2-resolve/actions/workflows/ci.yml/badge.svg)](https://github.com/allmonday/pydantic2-resolve/actions/workflows/ci.yml)
![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/allmonday/372580ad111c92340dac39987c0c4e9a/raw/covbadge.json)

![img](doc/resolver.png)

[Change Log](./changelog.md)

> A small yet powerful tool to extend your pydantic schema with no pain.
>
> It is the key to the realm of composition-oriented-development-pattern (wip)
>
> [Attention] This package supports pydantic v2 only, it's incompatible with pydantic v1, if you want to play with pydantic v1, please use pydantic-resolve instead.

What is composable pattarn? https://github.com/allmonday/composable-development-pattern

## Install

```shell
pip install pydantic-resolve
```

## Code snippets

1. basic usage, resolve your fields.

```python
import asyncio
from pydantic import BaseModel
from pydantic_resolve import Resolver

async def query_age(name):
    print(f'query {name}')
    await asyncio.sleep(1)
    _map = {
        'kikodo': 21,
        'John': 14,
        '老王': 40,
    }
    return _map.get(name)

class Person(BaseModel):
    name: str

    age: int = 0
    async def resolve_age(self):
        return await query_age(self.name)

    is_adult: bool = False
    def post_is_adult(self):
        return self.age > 18

async def simple():
    p = Person(name='kikodo')
    p = await Resolver().resolve(p)
    print(p)
    # query kikodo
    # Person(name='kikodo', age=21, is_adult=True)

    people = [Person(name=n) for n in ['kikodo', 'John', '老王']]
    people = await Resolver().resolve(people)
    print(people)
    # Oops!! the issue of N+1 query happens
    # query kikodo
    # query John
    # query 老王
    # [Person(name='kikodo', age=21, is_adult=True), Person(name='John', age=14, is_adult=False), Person(name='老王', age=40, is_adult=True)]

asyncio.run(simple())
```

2. optimize `N+1` with dataloader

```python
import asyncio
from typing import List
from pydantic import BaseModel
from pydantic_resolve import Resolver, LoaderDepend as LD

async def batch_person_age_loader(names: List[str]):
    print(names)
    _map = {
        'kikodo': 21,
        'John': 14,
        '老王': 40,
    }
    return [_map.get(n) for n in names]

class Person(BaseModel):
    name: str

    age: int = 0
    def resolve_age(self, loader=LD(batch_person_age_loader)):
        return loader.load(self.name)

    is_adult: bool = False
    def post_is_adult(self):
        return self.age > 18

async def simple():
    people = [Person(name=n) for n in ['kikodo', 'John', '老王']]
    people = await Resolver().resolve(people)
    print(people)

    # query query kikodo,John,老王 (N+1 query fixed)
    # [Person(name='kikodo', age=21, is_adult=True), Person(name='John', age=14, is_adult=False), Person(name='老王', age=40, is_adult=True)]

asyncio.run(simple())
```

## More cases:

for more cases like:

- how to filter members
- how to make post calculation after resolved?
- and so on..

please read the following demos.

```shell
cd examples

python -m readme_demo.0_basic
python -m readme_demo.1_filter
python -m readme_demo.2_post_methods
python -m readme_demo.3_context
python -m readme_demo.4_loader_instance
python -m readme_demo.5_subset
python -m readme_demo.6_mapper
python -m readme_demo.7_single
```

## API

### Resolver(loader_filters, global_loader_filter, loader_instances, ensure_type, context)

- loader_filters: `dict`

  provide extra query filters along with loader key.

  reference: [6_sqlalchemy_loaderdepend_global_filter.py](examples/6_sqlalchemy_loaderdepend_global_filter.py) L55, L59

- global_loader_filter: `dict`

  provide global filter config for all dataloader instances

  it will raise exception if some fields are duplicated with specific loader filter config in `loader_filters`

  reference: [test_33_global_loader_filter.py](tests/resolver/test_33_global_loader_filter.py) L47, L49

- loader_instances: `dict`

  provide pre-created loader instance, with can `prime` data into loader cache.

  reference: [test_20_loader_instance.py](tests/resolver/test_20_loader_instance.py), L62, L63

- ensure_type: `bool`

  if `True`, resolve method is restricted to be annotated.

  reference: [test_13_check_wrong_type.py](tests/resolver/test_13_check_wrong_type.py)

- annotation_class: `class`

  if you have `from __future__ import annotation`, and pydantic raises error, use this config to update forward refs

  reference: [test_25_parse_to_obj_for_pydantic_with_annotation.py](tests/resolver/test_25_parse_to_obj_for_pydantic_with_annotation.py), L39

- context: `dict`

  context can carry setting into each single resolver methods.

  ```python

  class Earth(BaseModel):
      humans: List[Human] = []
      def resolve_humans(self, context):
          return [dict(name=f'man-{i}') for i in range(context['count'])]

  earth = await Resolver(context={'count': 10}).resolve(earth)
  ```

### LoaderDepend(loader_fn)

- loader_fn: `subclass of DataLoader or batch_load_fn`. [detail](https://github.com/syrusakbary/aiodataloader#dataloaderbatch_load_fn-options)

  declare dataloader dependency, `pydantic-resolve` will take the care of lifecycle of dataloader.

### build_list(rows, keys, fn), build_object(rows, keys, fn)

- rows: `list`, query result
- keys: `list`, batch_load_fn:keys
- fn: `lambda`, define the way to get primary key

  helper function to generate return value required by `batch_load_fn`. read the code for details.

  reference: [test_utils.py](tests/utils/test_utils.py), L32

### mapper(param)

- param: `class of pydantic or dataclass, or a lambda`

  `pydantic-resolve` will trigger the fn in `mapper` after inner future is resolved. it exposes an interface to change return schema even from the same dataloader.
  if param is a class, it will try to automatically transform it.

  reference: [test_16_mapper.py](tests/resolver/test_16_mapper.py)

### ensure_subset(base_class)

- base_class: `class`

  it will raise exception if fields of decorated class has field not existed in `base_class`.

  reference: [test_2_ensure_subset.py](tests/utils/test_2_ensure_subset.py)

### model_config(default_required: bool)

- default_required: if True, fields with default values will also in schema['required']
- use with `Field(exclude=True)` to hide fields in schema and dumped result

  reference: [test_schema_config.py](tests/utils/test_model_config.py)

## Run FastAPI example:

FastAPI example

```shell
poetry shell
poetry install
cd examples
uvicorn fastapi_demo.main:app
visit http://localhost:8000/docs#/default/get_tasks_tasks_get
```

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
