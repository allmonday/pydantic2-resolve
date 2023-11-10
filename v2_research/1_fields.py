from pydantic import BaseModel, TypeAdapter, ConfigDict


class People(BaseModel):
    name: str
    age: int = 10
    model_config = ConfigDict(from_attributes=True)

class Person(BaseModel):
    name: str
    pp: People
    model_config = ConfigDict(from_attributes=True)
    

p = Person(name='kkk', pp=People(name='xxx'))

for v in p.model_fields.values():
    print(v.annotation)