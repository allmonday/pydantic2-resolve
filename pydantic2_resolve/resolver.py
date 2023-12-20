import asyncio
from collections import defaultdict
import contextvars
import inspect
from inspect import iscoroutine
from typing import TypeVar, Dict
from .exceptions import ResolverTargetAttrNotFound, LoaderFieldNotProvidedError, MissingAnnotationError
from typing import Any, Callable, Optional
from pydantic2_resolve import core
from aiodataloader import DataLoader
from inspect import isclass
from types import MappingProxyType
import pydantic2_resolve.constant as const
import pydantic2_resolve.util as util


def LoaderDepend(  # noqa: N802
    dependency: Optional[Callable[..., Any]] = None,
) -> Any:
    return Depends(dependency=dependency)

class Depends:
    def __init__(
        self, 
        dependency: Optional[Callable[..., Any]] = None,
    ):
        self.dependency = dependency

T = TypeVar("T")

class Resolver:
    """
    Entrypoint of a resolve action
    """
    def __init__(
            self, 
            loader_filters: Optional[Dict[Any, Dict[str, Any]]] = None, 
            loader_instances: Optional[Dict[Any, Any]] = None,
            ensure_type=False,
            context: Optional[Dict[str, Any]] = None
            ):
        self.loader_instance_cache = {}

        self.ancestor_vars = {}
        self.ancestor_vars_checker = defaultdict(set)  # expose_field_name: set(kls fullpath) if len > 1, raise error

        # for dataloader which has class attributes, you can assign the value at here
        self.loader_filters = loader_filters or {}

        # now you can pass your loader instance, Resolver will check `isinstance``
        if loader_instances and self._validate_loader_instance(loader_instances):
            self.loader_instances = loader_instances
        else:
            self.loader_instances = None

        self.ensure_type = ensure_type
        self.context = MappingProxyType(context) if context else None
        self.scan_data = {}


    def _add_expose_fields(self, target):
        """
        1. check whether expose to descendant existed
        2. add fields into contextvars (ancestor_vars_checker)
        2.1 check overwrite by another class(which is forbidden)
        2.2 check field exists
        """
        dct: Optional[dict] = getattr(target, const.EXPOSE_TO_DESCENDANT, None)
        # 1
        if dct:
            if type(dct) is not dict:
                raise AttributeError(f'{const.EXPOSE_TO_DESCENDANT} is not dict')

            # 2
            for field, alias in dct.items():  # eg: name, bar_name
                # 2.1
                self.ancestor_vars_checker[alias].add(util.get_kls_full_path(target.__class__))
                if len(self.ancestor_vars_checker[alias]) > 1:
                    conflict_modules = ', '.join(list(self.ancestor_vars_checker[alias]))
                    raise AttributeError(f'alias name conflicts, please check: {conflict_modules}')

                if not self.ancestor_vars.get(alias):
                    self.ancestor_vars[alias] = contextvars.ContextVar(alias)
                
                try:
                    val = getattr(target, field)
                except AttributeError:
                    raise AttributeError(f'{field} does not existed')

                self.ancestor_vars[alias].set(val)
    

    def _build_ancestor_context(self):
        """get values from contextvars and put into a dict"""
        return { k: v.get()  for k, v in self.ancestor_vars.items()}
    

    def _validate_loader_instance(self, loader_instances: Dict[Any, Any]):
        for cls, loader in loader_instances.items():
            if not issubclass(cls, DataLoader):
                raise AttributeError(f'{cls.__name__} must be subclass of DataLoader')
            if not isinstance(loader, cls):
                raise AttributeError(f'{loader.__name__} is not instance of {cls.__name__}')
        return True
    

    def _execute_resolver_method(self, method):
        """
        1. inspect method, atttach context if declared in method
        2. if params includes LoaderDepend, create instance and cache it.
            2.1 create from DataLoader class
                2.1.1 apply loader_filters into dataloader instance
            2.2 ceate from batch_load_fn
        3. execute method
        """

        # >>> 1
        signature = inspect.signature(method)
        params = {}

        if signature.parameters.get('context'):
            if self.context is None:
                raise AttributeError('Resolver.context is missing')
            params['context'] = self.context

        if signature.parameters.get('ancestor_context'):
            if self.ancestor_vars is None:
                raise AttributeError(f'there is not class has {const.EXPOSE_TO_DESCENDANT} configed')
            params['ancestor_context'] = self._build_ancestor_context()

        # manage the creation of loader instances
        for k, v in signature.parameters.items():
            # >>> 2
            if isinstance(v.default, Depends):
                # Base: DataLoader or batch_load_fn
                Loader = v.default.dependency

                # check loader_instance first, if already defined in Resolver param, just take it.
                if self.loader_instances and self.loader_instances.get(Loader):
                    loader = self.loader_instances.get(Loader)
                    params[k] = loader
                    continue

                # module.kls to avoid same kls name from different module
                cache_key = util.get_kls_full_path(v.default.dependency)
                hit = self.loader_instance_cache.get(cache_key)
                if hit:
                    loader = hit
                else:
                    # >>> 2.1
                    # create loader instance 
                    if isclass(Loader):
                        # if extra transform provides
                        loader = Loader()

                        filter_config = self.loader_filters.get(Loader, {})

                        for field in util.get_class_field_annotations(Loader):
                        # >>> 2.1.1
                        # class ExampleLoader(DataLoader):
                        #     filtar_x: bool  <--------------- set this field
                            try:
                                value = filter_config[field]
                                setattr(loader, field, value)
                            except KeyError:
                                raise LoaderFieldNotProvidedError(f'{cache_key}.{field} not found in Resolver()')

                    # >>> 2.2
                    # build loader from batch_load_fn, filters config is impossible
                    else:
                        loader = DataLoader(batch_load_fn=Loader) # type:ignore

                    self.loader_instance_cache[cache_key] = loader
                params[k] = loader

        # 3
        return method(**params)


    def _execute_post_method(self, method):
        signature = inspect.signature(method)
        params = {}

        if signature.parameters.get('context'):
            if self.context is None:
                raise AttributeError('Post.context is missing')
            params['context'] = self.context

        if signature.parameters.get('ancestor_context'):
            if self.ancestor_vars is None:
                raise AttributeError(f'there is not class has {const.EXPOSE_TO_DESCENDANT} configed')
            params['ancestor_context'] = self._build_ancestor_context()
        return method(**params)


    async def _resolve_obj_field(self, target, field, attr):
        """
        resolve each single object field

        1. validate the target field of resolver method existed.
        2. exec methods
        3. parse to target type and then continue resolve it
        4. set back value to field
        """

        # >>> 1
        target_attr_name = str(field).replace(const.PREFIX, '')

        if not hasattr(target, target_attr_name):
            raise ResolverTargetAttrNotFound(f"attribute {target_attr_name} not found")

        if self.ensure_type:
            if not attr.__annotations__:
                raise MissingAnnotationError(f'{field}: return annotation is required')

        # >>> 2
        val = self._execute_resolver_method(attr)
        while iscoroutine(val) or asyncio.isfuture(val):
            val = await val

        # >>> 3
        if not getattr(attr, const.HAS_MAPPER_FUNCTION, False):  # defined in util.mapper
            val = util.try_parse_data_to_target_field_type(target, target_attr_name, val)

        val = await self._resolve(val)

        # >>> 4
        setattr(target, target_attr_name, val)


    async def _resolve(self, target: T) -> T:
        """ 
        resolve object (pydantic, dataclass) or list.

        1. iterate over elements if list
        2. resolve object
            2.1 resolve each single resolver fn and object fields
            2.2 execute post fn
        """

        # >>> 1
        if isinstance(target, (list, tuple)):
            await asyncio.gather(*[self._resolve(t) for t in target])

        # >>> 2
        if core.is_acceptable_instance(target):
            self._add_expose_fields(target)
            tasks = []
            # >>> 2.1
            resolve_list, attribute_list = core.iter_over_object_resolvers_and_acceptable_fields(target, self.scan_data)
            for field, attr in resolve_list:
                tasks.append(self._resolve_obj_field(target, field, attr))
            for field, attr in attribute_list:
                tasks.append(self._resolve(attr))

            await asyncio.gather(*tasks)

            # >>> 2.2
            # execute post methods, if context declared, self.context will be injected into it. 
            for post_key in core.iter_over_object_post_methods(target, self.scan_data):
                post_attr_name = post_key.replace(const.POST_PREFIX, '')
                if not hasattr(target, post_attr_name):
                    raise ResolverTargetAttrNotFound(f"fail to run {post_key}(), attribute {post_attr_name} not found")

                post_method = getattr(target, post_key)
                calc_result = self._execute_post_method(post_method)
                setattr(target, post_attr_name, calc_result)
            
            # finally, if post_default_handler is declared, run it.
            default_post_method = getattr(target, const.POST_DEFAULT_HANDLER, None)
            if default_post_method:
                self._execute_post_method(default_post_method)

        return target


    async def resolve(self, target: T) -> T:
        if isinstance(target, list) and target == []:
            return target

        self.scan_data = core.scan_and_store_required_fields(target)

        await self._resolve(target)
        return target 