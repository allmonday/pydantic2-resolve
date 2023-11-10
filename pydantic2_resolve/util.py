import asyncio
import types
import functools
from collections import defaultdict
from dataclasses import is_dataclass
from pydantic import BaseModel, TypeAdapter, ValidationError
from inspect import iscoroutine, isfunction
from typing import Any, DefaultDict, Sequence, Type, TypeVar, List, Callable, Optional, Mapping, Union, Iterator, Dict, get_type_hints
import pydantic2_resolve.constant as const



def get_class_field_annotations(cls: Type):
    anno = cls.__dict__.get('__annotations__') or {}
    return anno.keys()


T = TypeVar("T")
V = TypeVar("V")

def build_object(items: Sequence[T], keys: List[V], get_pk: Callable[[T], V]) -> Iterator[Optional[T]]:
    """
    helper function to build return object data required by aiodataloader
    """
    dct: Mapping[V, T] = {}
    for item in items:
        _key = get_pk(item)
        dct[_key] = item
    results = (dct.get(k, None) for k in keys)
    return results


def build_list(items: Sequence[T], keys: List[V], get_pk: Callable[[T], V]) -> Iterator[List[T]]:
    """
    helper function to build return list data required by aiodataloader
    """
    dct: DefaultDict[V, List[T]] = defaultdict(list) 
    for item in items:
        _key = get_pk(item)
        dct[_key].append(item)
    results = (dct.get(k, []) for k in keys)
    return results


def replace_method(cls: Type, cls_name: str, func_name: str, func: Callable):
    KLS = type(cls_name, (cls,), {func_name: func})
    return KLS


def get_required_fields(kls: BaseModel):
    required_fields = []

    # 1. get required fields
    for fname, field in kls.model_fields.items():
        if field.is_required:
            required_fields.append(fname)

    # 2. get resolve_ and post_ target fields
    for f in dir(kls):
        if f.startswith(const.PREFIX):
            if isfunction(getattr(kls, f)):
                required_fields.append(f.replace(const.PREFIX, ''))

        if f.startswith(const.POST_PREFIX):
            if isfunction(getattr(kls, f)):
                required_fields.append(f.replace(const.POST_PREFIX, ''))
    
    return required_fields


def output(kls):
    """
    set required as True for all fields
    make typescript code gen result friendly to use
    """

    if issubclass(kls, BaseModel):

        def build():
            def schema_extra(schema: Dict[str, Any], model) -> None:
                fnames = get_required_fields(model)
                schema['required'] = fnames
            return schema_extra

        kls.Config.schema_extra = staticmethod(build())

    else:
        raise AttributeError(f'target class {kls.__name__} is not BaseModel')
    return kls


def mapper(func_or_class: Union[Callable, Type]):
    """
    execute post-transform function after the value is reolved
    func_or_class:
        is func: run func
        is class: call auto_mapping to have a try
    """
    def inner(inner_fn):

        # if mapper provided, auto map from target type will be disabled
        setattr(inner_fn, const.HAS_MAPPER_FUNCTION, True)

        @functools.wraps(inner_fn)
        async def wrap(*args, **kwargs):

            retVal = inner_fn(*args, **kwargs)
            while iscoroutine(retVal) or asyncio.isfuture(retVal):
                retVal = await retVal  # get final result
            
            if retVal is None:
                return None

            if isinstance(func_or_class, types.FunctionType):
                # manual mapping
                return func_or_class(retVal)
            else:
                # auto mapping
                if isinstance(retVal, list):
                    if retVal:
                        rule = get_mapping_rule(func_or_class, retVal[0])
                        return apply_rule(rule, func_or_class, retVal, True)
                    else:
                        return retVal  # return []
                else:
                    rule = get_mapping_rule(func_or_class, retVal)
                    return apply_rule(rule, func_or_class, retVal, False)
        return wrap
    return inner


def get_mapping_rule(target, source) -> Optional[Callable]:
    # do noting
    if isinstance(source, target):
        return None

    # pydantic
    if issubclass(target, BaseModel):
        if target.model_config.get('from_attributes'):
            if isinstance(source, dict):
                raise AttributeError(f"{type(source)} -> {target.__name__}: pydantic from_orm can't handle dict object")
            else:
                return lambda t, s: t.model_validate(s)

        if isinstance(source, dict):
            return lambda t, s: t.model_validate(s)

        if isinstance(source, BaseModel):
            if source.model_config.get('from_attributes'):
                return lambda t, s: t.model_validate(s) 
            else:
                return lambda t, s: t(**s.model_dump()) 

        else:
            raise AttributeError(f"{type(source)} -> {target.__name__}: pydantic can't handle non-dict data")
    
    # dataclass
    if is_dataclass(target):
        if isinstance(source, dict):
            return lambda t, s: t(**s)
    
    raise NotImplementedError(f"{type(source)} -> {target.__name__}: faild to get auto mapping rule and execut mapping, use your own rule instead.")


def apply_rule(rule: Optional[Callable], target, source: Any, is_list: bool):
    if not rule:  # no change
        return source

    if is_list:
        return [rule(target, s) for s in source]
    else:
        return rule(target, source)
    

def ensure_subset(base):
    """
    used with pydantic class to make sure a class's field is 
    subset of target class
    """
    def wrap(kls):
        assert issubclass(base, BaseModel), 'base should be pydantic class'
        assert issubclass(kls, BaseModel), 'class should be pydantic class'

        @functools.wraps(kls)
        def inner():
            for k, field in kls.model_fields.items():
                if field.is_required:
                    base_field = base.model_fields.get(k)
                    if not base_field:
                        raise AttributeError(f'{k} not existed in {base.__name__}.')
                    if base_field and base_field.type_ != field.annotation:
                        raise AttributeError(f'type of {k} not consistent with {base.__name__}'  )
            return  kls
        return inner()
    return wrap


def update_forward_refs(kls: Type[BaseModel]):
    """
    recursively update refs.
    """
    # kls.update_forward_refs()
    kls.model_rebuild()
    setattr(kls, const.PYDANTIC_FORWARD_REF_UPDATED, True)

def update_dataclass_forward_refs(kls):
    if not getattr(kls, const.DATACLASS_FORWARD_REF_UPDATED, False):
        anno = get_type_hints(kls)
        kls.__annotations__ = anno
        setattr(kls, const.DATACLASS_FORWARD_REF_UPDATED, True)

        for _, v in kls.__annotations__.items():
            t = shelling_type(v)
            if is_dataclass(t):
                update_dataclass_forward_refs(t)


def try_parse_data_to_target_field_type(target, field_name, data):
    """
    parse to pydantic or dataclass object
    1. get type of target field
    2. parse
    """
    field_type = None

    # 1. get type of target field
    if isinstance(target, BaseModel):
        _fields = target.__class__.model_fields
        field_type = _fields[field_name].annotation

        # handle optional logic
        if data is None and _fields[field_name].is_required == False:
            return data

    elif is_dataclass(target):
        field_type = target.__class__.__annotations__[field_name]

    # 2. parse
    if field_type:
        try:
            # result = parse_obj_as(field_type, data)
            result = TypeAdapter(field_type).validate_python(data)
            return result
        except ValidationError as e:
            print(f'Warning: type mismatch, pls check the return type for "{field_name}", expected: {field_type}')
            raise e
    
    else:
        return data  #noqa


def is_optional(annotation):
    annotation_origin = getattr(annotation, "__origin__", None)
    return annotation_origin == Union \
        and len(annotation.__args__) == 2 \
        and annotation.__args__[1] == type(None)  # noqa

def is_list(annotation):
    return getattr(annotation, "__origin__", None) == list

def shelling_type(type):
    while is_optional(type) or is_list(type):
        type = type.__args__[0]
    return type