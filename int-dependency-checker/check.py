"""
_MorePy.py

Project started by JoBe
Github: 
    https://www.github.com/JoBeGaming/MorePy

Version intended for usage with Python 3.14.0a7
"""
#TODO: Add `@Validate` Decorator (gets annotations of self.func, and compares types), make all decorators copy __doc__ and co.! What if we have a = Union[a, Union[a, b], Union[b], Union[c]] or similar?????????????????

__author__: str = "JoBe" 
__github__: str = "https://www.github.com/JoBeGaming/"
__version__: str = "0.43a6"
__release__: bool = False
#TODO tvt with default, typeMApping, ...


__all__: list[str] = [
    "_",
    "_Decorator",
    "_Factory", #TODO depr???! -> UNused -> only export??
    "_Final",
    "_FinalClassVar",
    "_got_finalized",
    "_is_generic",
    "_is_type_or_ellipsis",
    "_isinstance", 
    "_Mixin", #TOD0 make Alias `Mixin` = Mixin
    "_no_arg_given", #TODO keep?
    "_Property", # Only if i do something with it, and add the args
    "_repr_obj",
    "_repr_type", #TODO use this
    "_SENTINEL",
    "_set",
    "all_isinstance",
    "all_isinstance_cache",
    "ALLOWED_TYPES",
    "Any",
    "Base", #TODO ?
    "CallableConstruct",  #TODO ?
    "catch",
    "descriptor", #Typing.Doc ??
    "empty",
    "gettable",
    "NewConstruct",
    "NewTypeBaseClass",
    "Not",
    "NotCallable", # TODO  ??
    "onError",
    "onException",
    "OwnedSet",
    "pack",
    "slots",
    "stub", #TODO maybe depr
    "supportsWeakref",
    "Type", # maybe not?! -> could get deprd
    "TypedSet",
    #"TypedUnion", #TODO
    "TypeVar",
    "unpack",
    "walk_type_tree", #TODO add the whole system!!
]

# For everyone using Cspell:
# Cspell:words getframe, DEPR, haskey, Typecheckers, kwdefaults, Pylance, redef

# -------
# Imports
# -------

# b_type is used to indicate the classes
# functionality, both to get the runtime
# type of an object, and to create a new
# type on the fly, although only the 
# first case is enforced.
# For type-annotations of types, `type` 
# should always be preferred. 
from builtins import type as b_type

from sys import implementation
from types import GenericAlias
from collections.abc import Callable

from typing import (
    _ProtocolMeta, # type: ignore[reportPrivateUsage]
    Any as t_Any,
    get_args as t_get_args,
    get_origin as t_get_origin,
    Literal,
    Never,
    #overload,
    #Protocol,
    #TYPE_CHECKING,
    TypeAlias,
    TypeAliasType,
    TypeVar as t_TypeVar,
    #TypeGuard,
    #cast
)

import typing
import annotationlib # type: ignore[import-not-found, reportMissingStub]

# Credits
__credits__: set[str] =  {
    "builtins.py",
    "dataclasses.py",
    "inspect.py",
    "pathlib.py",
    "sys.py",
    "time.py",
    "types.py",
    "typing.py"
}

def getframe(depth: int = 0, /, *fallback: Any):
    if implementation.name == "cpython":
        from sys import _getframe # type: ignore[reportPrivateUsage]
        try:
            return _getframe(depth)
        except KeyboardInterrupt:
            raise 
        except Exception:
            return fallback
        finally:
            del _getframe
    else:
        from inspect import stack
        try:
            return stack()[depth]
        except KeyboardInterrupt:
            raise
        except Exception:
            return fallback
        finally:
            del stack

del implementation

def warn(msg: str, tp: Literal["deprecation", "todo"] | str="", color: int=91, sep: str = "=") -> None:
    # Following separators work the best:
    #   "-", # Thinnest
    #   "~", # Thin
    #   "=", # Medium
    #   "#", # Bold

    from os import get_terminal_size
    if tp:
        tp = str(tp) + str("-")

    print(sep * (get_terminal_size().columns // len(sep)))
    print(f"\033[93m{str(tp).upper()}WARNING:\033[00m\033[{int(color)}m {msg}\033[00m")
    print(sep * (get_terminal_size().columns // len(sep)))

    del get_terminal_size


class DEPR:

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __call__(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
        warn(f"Function or Object {self.func} is getting deprecated", "deprecation")
        return self.func(*args, **kwargs)


class TODO:

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __call__(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
        warn(f"Function or Object {self.func} is still being worked on", "todo")
        return self.func(*args, **kwargs)

# -----------------
# Beginning of Code
# -----------------
type Any = t_Any
GenericCallable = Callable[..., Any]

# typing.Union is used here, 
# as the __or__ and __ror__
# of types return such Union
t_UnionType = typing.Union


#validType: TypeAlias = """type | str | annotationlib.ForwardRef | type[Ellipsis]
#| tuple[validType, ...] | TypeAlias | TypeAliasType | typing.Union[validType] | GenericAlias"""
validType = ...
ALLOWED_TYPES: tuple[type, ...] = (
    type, 
    TypeAliasType, 
    GenericAlias,
    t_UnionType
) #TYPE_VAR??
"""All type-forms to be passed instead of raw `type`"""

# Note that using conversions,
# a lot of the types specified
# in ALLOWED_TYPES can be used
# with either isinstance() or
# _isinstance().
INSTANCE_TYPES: tuple[type, ...] = (
    type, 
    TypeAliasType,
    t_UnionType
) #TYPE_VAR??
"""Similar to `ALLOWED_TYPES`, but inline with `isinstance()`"""

del GenericAlias

def _resolve_types(tp: validType, *, allow_duplicates: bool = False) -> type | tuple[type, ...]:
    if not _isinstance(tp, tuple):
        tp = (tp,)
    res: tuple[type, ...] = ()
    for t in tp:
        if tp is ...:
            ...
        elif _isinstance(t, type):
            ...
        elif _isinstance(t, (str, annotationlib.ForwardRef)):
            ...
        elif get_origin(t):
            origin = get_origin(t)
            args = get_args(t)
            
    if len(res) < 2:
        return res[0]
    return res
    
    #TODO when checking stuff use 'ellipsis' instead of '...' when checking for ... input
    #TODO use this

def _collect_type_params_repr(params: tuple[type, ...]) -> str: 
    """Collect and unionize all type-params given to a single string."""
    res = ""
    for index, tp in enumerate(params):
        if type(tp) == type:
            res += tp.__name__ #_repr_type(tp)
        else:
            res += str(tp)
        if not index == len(params) - 1:
            res += ", "
    return res


def _is_acceptable_type(tp):
    ...


slots = tuple[str, ...]
"""Type of the `__slots__` attribute of classes."""


# Globally used internal helpers
# ------------------------------
#TODO rename??!
def is_obj(obj: obj) -> bool:
    """Helper to check wether a given `object` type is a valid object."""
    return str(obj).startswith("<class") and str(obj).endswith(">")


def _is_class(obj: obj) -> bool:
    """Helper to check wether a given `object` type is a valid class."""
    return type(obj) == type


def _isinstance(obj: Any, tp: Any) -> bool:
    origin = get_origin(tp, allow_all=True, allow_none=True)
    args = get_args(tp, allow_all=True)
    if origin is None:
        origin = t_get_origin(tp)
    # Still is None
    if origin is None:
        if isinstance(tp, tuple) and Ellipsis in tp:
            return _isinstance(obj, tp[0])  # Validate all items as `tp[0]`
        if isinstance(tp, tuple):
            return any(_isinstance(obj, single_tp) for single_tp in tp)
        return isinstance(obj, tp)
    if origin is tuple:
        if not isinstance(obj, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_isinstance(o, args[0]) for o in obj)
        else:
            if not len(obj) == len(args):
                return False
        return all(_isinstance(o, t) for o, t in zip(obj, args))
    if origin is list:
        if not isinstance(obj, list):
            return False
        return all(_isinstance(o, args[0]) for o in obj)
    if origin is set:
        if not isinstance(obj, set):
            return False
        return all(_isinstance(o, args[0]) for o in obj)
    if origin is dict:
        if not isinstance(obj, dict):
            return False
        key_tp, val_tp = args
        return all(_isinstance(k, key_tp) and _isinstance(v, val_tp) for k, v in obj.items())
    return isinstance(obj, UnionType(origin, tp)) or type(obj) == tp # type: ignore[arg-type]


def _is_generic(obj):
    return (
        get_origin(type(obj), allow_none=True) is not None or 
        hasattr(obj, "_is_generic")
    ) #TODO use ABC-like subclasshook approach / check subclass of an abc


# Caches

def _copy(_from: Callable[..., Any], _to: Callable[..., Any]) -> None:
    attributes = {
        "__name__", 
        "__doc__", 
        "__annotations__", 
        "__module__",
        "__defaults__",
        "__kwdefaults__"
    }
    for attr in attributes:
        try:
            setattr(_to, attr, getattr(_from, attr, None))
        except NameError:
            pass


def _cache(obj: Callable) -> Callable:
    _cache: CacheType = {}

    def make_key(args: tuple[Any], kwargs: dict[Any, Any]) -> CacheKey:
        key = args
        if kwargs:
            key += tuple(sorted(kwargs.items()))
        return key

    def wrapper(*args, **kwargs):
        key = make_key(args, kwargs)
        if key in _cache:
            return _cache[key]
        result = obj(*args, **kwargs)
        _cache[key] = result
        return result

    _copy(obj, wrapper)
    return wrapper


# Be careful when using `@_id_cache`,
# as some things might be mutated at
# runtime, which makes the key
# invalid.
def _id_cache(obj: Callable) -> Callable:
    _cache: IdCacheType = {}

    def make_key(args, kwargs) -> IdCacheKey:
        return (
            id(obj),
            id(args),
            id(kwargs)
        )

    def wrapper(*args, **kwargs):
        key = make_key(args, kwargs)
        if key in _cache:
            return _cache[key]
        result = obj(*args, **kwargs)
        _cache[key] = result
        return result

    _copy(obj, wrapper)
    return wrapper


def get_origin(tp: type | obj, *, allow_none: bool = False, allow_all: bool = False) -> Any | None:
    if (
        not allow_all and
        not isinstance(tp, type) and 
        t_get_origin(tp) is None
    ):
        raise TypeError(f"get_origin only works for types, got '{tp}'")
    if t_get_origin(tp) is not None:
        return t_get_origin(tp)
    if getattr(tp, "_is_union", False):
        return UnionType #TODO TEST!!!!!
    if not allow_none:
        return tp
    return None


def get_args(tp: type | obj, *, allow_all: bool = False) -> tuple[Any, ...]:
    if tp is None:
        return ()
    if (
        not allow_all and
        not isinstance(tp, type) and 
        t_get_origin(tp) is None
    ):
        raise TypeError(f"get_args only works for types, got '{tp}'")
    if getattr(tp, "_is_union", False):
        return getattr(tp, "tp", ())
    if _is_generic(tp):
        return getattr(tp, "args")
    return t_get_args(tp)


def _get_multiple_dict_entries(dict_: dict[_KT, _VT], *keys: _KT) -> tuple[_VT, ...]:
    entries = ()
    for key in keys:
        entries += (dict_[key],)
    return entries


def _get_all_dict_entires_except(dict: dict[_KT, _VT], *keys: _KT) -> tuple[_VT, ...]:
    final_keys: tuple[_VT, ...] = ()
    for key in dict.keys():
        if not key in keys:
            final_keys += (key,)
    return _get_multiple_dict_entries(dict, final_keys)


# Helper for `@_Form` decorator
def _repr(_self: Self) -> str:
    return f"MorePy.{_self.__class__.__name__}"


# Similar to `typing._SpecialForm`
def _Form(cls: T) -> T:
    if not isinstance(cls, type):
        raise TypeError("Cannot use `@_Form` for non-class objects.")
    if '__repr__' not in cls.__dict__:
        cls.__repr__ = _repr
    return cls


def supportsWeakref(cls: T) -> T:
    """Internal Decorator to add `__weakref__` to the `__slots__` attribute of objects."""
    if not isinstance(cls, type):
        raise TypeError("Cannot use `@allowsWeakref` for non-class objects.")
    if hasattr(cls, "__slots__"):
        if "__weakref__" not in cls.__slots__:
            prev_slots = getattr(cls, "__slots__")
            cls.__slots__ = prev_slots + ("__weakref__",)
    return cls


def _tp_getitem(func: Callable[[Any, tuple[type, ...]], type | tuple[type, ...]]) -> Callable[[Any, type | tuple[type, ...]], type | tuple[type, ...]]:
    #assert len(func.__annotations__.items()) >= 2 
    #assert len(func.__annotations__.items()) <= 3
    resolved_params = (Any,)
    for param in func.__annotations__:
        if param not in ("cls", "return"):
            resolved_params = get_args(func.__annotations__[param], allow_all=True)
            if not resolved_params[1] is Ellipsis:
                raise TypeError
            
            resolved_params = get_args(resolved_params[0], allow_all=True)
            break
    if not func.__name__ in ("__class_getitem__", "__getitem__"):
        raise TypeError

    def wrapper(self, items: type | tuple[type, ...]) -> type | tuple[type, ...]:
        if not isinstance(items, tuple):
            items = (items,)
        for tp in items:
            if not type(tp) in resolved_params:
                raise TypeError(
                    f"{func.__qualname__.split(func.__name__)[0].rstrip(".")}["
                    f"{_collect_type_params_repr(items)}]: each arg must be a type, got {tp}"
                )
        return func(self, items)

    _copy(func, wrapper)
    return wrapper

class c[*T]:
    @_tp_getitem
    def __class_getitem__(cls, items: tuple[type | TypeAlias, ...]) -> ...:
        ...
print(c[int, str, int | float])

# This just helps to make code more readable, as we now only have a function we use instead of a class
# We use the `__type__` attribute, discussed in PEP /////// #TODO
@_id_cache #del?
def Type(obj: obj, /) -> type:
    """
    Function that returns the type of any object, using the newly added `__type__` attribute of the object,
    or, as a fallback the builtin `type`.
    There are edge cases, like::

        >>> type t = int | float
        >>> print(type(t)) 
        <class 'typing.TypeAliasType'>
        >>> print(Type(t)) 
        <class 'int'> | <class 'float'>

    In this case the type is an instance of `typing.TypeAliasType`, and we will return the types we initiated the object with.  
    Note that using the builtin `type` is often more reliable for code that checks for specific types, or for cases, where the returned
    arguments should be used with `eval()`.
    """  

    if isinstance(obj, TypeAliasType):
        return obj.__value__
    return getattr(obj, "__type__", type(obj))

exit()

MAX_REPR_SIZE: int = 100

def _trunc_repr(obj: obj) -> str:
    repr = f"{obj!r}"
    return _trunc(repr)


def _trunc(s: str) -> str:
    """Return the string, truncated to `MAX_REPR_SIZE` characters if needed"""
    if len(s) > MAX_REPR_SIZE:
        return f"{s[:MAX_REPR_SIZE]}..."
    return s


def _repr_obj(obj: obj, *, treat_generics_like_all: bool = True) -> str:
    tp = type(obj)
    if not _is_generic(obj):
        return _repr_type(tp)
    if treat_generics_like_all:
        return _trunc_repr(obj)
    
    return f"{_repr_type(tp)}[{' , '.join([_repr_obj(o) for o in get_args(obj)])}]" 
# type: ignore[arg-type] # We checked all cases, obj is a type


def _repr_type(tp: type) -> str:
    """Represent a type, **not an argument or its type**."""
    return getattr(tp, "__name__", repr(tp))


def _hl_arg(arg: Any) -> str:
    """
    Helper to convert the argument to be highlighted with '...' notation, e.g.:

        >>> print(_hl_arg(1))
        '1'
        >>> print(_hl_arg("A"))
        'A'
        >>> ...
    """
    return f"'{arg}'"


def _is_empty(obj: tuple | None, /) -> bool:
    """Helper to check wether a given tuple is empty."""
    return not obj or obj is None


# Cache to lookup the arguments that failed the test
all_isinstance_cache: dict[tuple[Any, ...], Any] = {}

# TODO del later or something, this is internal
def all_isinstance(obj: tuple[Any, ...], instance: type | tuple[type, ...]) -> bool:
    """
    Helper to check wether a collection of arguments matches the type for each item 
    in the collection, whilst caching the object that failed the test (first).
    """

    if (
        not isinstance(instance, type) and 
        not isinstance(instance, tuple) and
        not all_isinstance(instance, type) and
        not isinstance(get_args(instance)[0], type) #TODO does this actually pass Unions? -> t.Union + mp.UnionType
    ):
        raise TypeError(f"instance should be a type or Union of types, got {instance}")
    if not isinstance(obj, tuple):
        raise TypeError(f"expected tuple, got {Type(obj)}")
    for o in obj:
        try:
            if not isinstance(o, instance):
                all_isinstance_cache[obj] = o
                return False
        except TypeError:
            if not _isinstance(o, instance):
                all_isinstance_cache[obj] = o
                return False
    return True


@DEPR # unused, see Mixins to use it, and move to them as renamed one
def all_unique(obj: tuple[Any, ...]) -> bool:
    return len(set(obj)) == len(obj)


# Can involve recursion, so we cache it.
@_cache
def unpack(*obj: T) -> T:
    """
    Helper to turn a tuple into its first argument, 
    and to unpack nested tuples.
    """

    if isinstance(obj[0], tuple):
        return unpack(obj[0])
    return obj[0]


def pack(*args: T | tuple[T, ...], add_to_existing: bool = False) -> tuple[T] | tuple[T, ...]: #TODO retrun type annot cleaner (tvt)
    """
    Helper to pack all arguments into a tuple. 
    If the additional argument `add_to_existing` is True, 
    we turn each tuple found in the args into its args and 
    append them, as if we found all args in the *args tuple.
    """

    if not add_to_existing:
        return tuple(args)
    res: tuple[Any, ...] = ()
    for arg in args:
        if isinstance(arg, tuple):
            for sub_arg in arg:
                res += (sub_arg,)
        else:
            res += (arg,)
    return res


#TODO used once, so mve closer?
def _got_finalized(obj: obj) -> bool:
    """
    Internal Helper to check wether an object 
    has either been decorated with `@_Final`, 
    or has inherited from `_Final`.
    """

    return (
        (
            isinstance(obj, type) and 
            issubclass(obj, _Final)
        ) or 
        isinstance(obj, _Final) or
        getattr(obj, "__final__", False) # Using `__final__` makes this work with typing's `@final`, and type-checkers
    )


# Can involve recursion, so we cache it.
@_cache
def flatten(obj: tuple[T, ...]) -> tuple[T, ...]:
    final: tuple[T, ...] = ()
    for element in obj:
        if isinstance(element, tuple):
            if not _is_empty(element):
                final += flatten(element)
        else:
            final += pack(element)
    return final

# Sentinel
#TOD0 make this Any-like
@_Form
class _SENTINEL(int):
    """Simple Sentinel, used to indicate no argument given"""

    __slots__: slots = ()

    def __init__(self) -> None:
        return id(_SENTINEL) # type: ignore[return-value]

    def __getitem__(self, *args: _argsType) -> UnionType[int, Any]:
        return UnionType(int, *args)

    def __eq__(self, other) -> Literal[False]:
        return False

    def __ne__(self, other) -> Literal[True]:
        return True

    def __bool__(self) -> Literal[False]:
        return False


def _no_arg_given(obj: T) -> bool:
    # Some Implementations do:
    #
    # >>> obj == _SENTINEL
    #
    # But the `==` functionality 
    # can be influenced with the 
    # `__eq__` method. To prevent
    # false positives, we use the 
    # id of the object. The `id()` 
    # function does not need to get
    # called however, as doing
    # 
    # >>> obj is _SENTINEL
    #
    # is around 2-3x faster than:
    #
    # >>> o_id = id(obj) 
    # >>> s_id = id(_SENTINEL)
    # >>> o_id == s_id
    return obj is _SENTINEL


def get_cls_meta(cls: type) -> str:
    """
    Internal helper used to get the self of a class that is used as an instance. 
    Useful for converting `cls` of a `__init_subclass__` to `self` of methods like
    `__init__`. We can use it like:

        >>> class base_cls:
        ...
        >>>   some_class_registry = {}   
        ...
        >>>   def __init__(self, *args):
        >>>     kwargs = base_cls.some_class_registry[str(self)]
        ...
        >>>   def __init_subclass__(cls, **kwargs):
        >>>     self = get_cls_meta(cls)
        >>>     base_cls.some_class_registry[self] = kwargs

    Note that the result will always be a string.
    """

    # Only happens during testing
    # via this file. Generally, we
    # should always use tests, and
    # never run this file directly,
    # unless there is a new feature
    # being implemented.
    if __name__ == "__main__":
        # Windows uses \\ instead of /, 
        # so we need to substitute it.
        module = __file__.replace("\\", "/").split("/")[-1]
        return f"{module.removesuffix("py")}{cls.__qualname__}"
    return f"{cls.__module__}.{cls.__qualname__}"


# TODO used once, so move, use more often or depr?!
def _get_key(dict_: dict[_KT, _VT], key: _KT, default: Any = None) -> _VT:
    if key in dict_:
        return dict_[key]
    else: 
        return default


class _Mixin:

    __slots__: slots = (
        "_works_with",
        "_not_works_with"
    )

    _special: dict[str, dict[str, tuple[type, ...]]] = {}

    def __init_subclass__(
        cls,
        /,
        *,
        base: bool = False,
        _works_with: tuple[type, ...] = _SENTINEL,
        _not_works_with: tuple[type, ...] = _SENTINEL
    ) -> Instance:
        _Mixin._special[get_cls_meta(cls)] = {
            "_works_with": _works_with,
            "_not_works_with": _not_works_with
        }
        print(cls.__bases__, cls.__mro__)

#TODO copy stuff from file

#TODO make a system to annotate using a Var (tuple or other collection) of types -> ...[ALLOWED_TYPES] (instead of Any here!)
# In a sense, this may also be used to flatten nested type params
@_cache
def walk_type_tree(tree: tuple[Any, ...] | Any, /, *, level: int = 0) -> tuple[tuple[int, type, bool], ...]:
    """
    Return a tuple of all types with their 'indentation' level and the corresponding type.  
    The boolean is used to indicate any number of said argument following, if True:

        >>> tp = tuple[int, str, tuple[int, ...]], int
        ...
        >>> print(walk_type_tree(tp))
        ((0, <class 'tuple'>, False), (1, <class 'int'>, False), (1, <class 'str'>, False), 
        (1, <class 'tuple'>, False), (2, <class 'int'>, True), (0, <class 'int'>, False))

    In edge-cases, like `tp = ...`, `()` is returned.  
    Unions like `T1 | T2` will be converted to 
    `((0, Union, False), (1, T1, False), (1, T2, False))`,
    as will explicit Unions like `Union[T1, T2]`.  
    Note that the tree is not checked for validity
    """

    final: list[tuple[int, type, bool]] = []
    if not isinstance(tree, tuple):
        tree = (tree,)
    if tree == (...,):
        return ()
    for tp in tree:
        if tp is ...:
            for index_of_ellipsis_owner in range(len(final) - 1, -1, -1):
                if final[index_of_ellipsis_owner][0] == level:
                    prev_args = final[index_of_ellipsis_owner][:-1]
                    final[index_of_ellipsis_owner] = (prev_args + (True,))
                    break
            continue
        origin = get_origin(tp, allow_none=True, allow_all=True)
        args = get_args(tp, allow_all=True)
        if origin is None:
            final.append((level, tp, False))
        else:
            final.append((level, origin, False))
            inner = walk_type_tree(args, level=level + 1)
            if inner:
                final.extend(inner)
    return tuple(final)


def reconstruct_type_tree(tree: tuple[tuple[int, type, bool], ...]) -> tuple | Any:
    ordered_levels: tuple[int, ...] = ()
    ordered_types: tuple[type, ...] = ()
    ordered_follows_ellipsis: tuple[type, ...] = ()
    for t in tree:
        ordered_levels += (t[0],)
        ordered_types += (t[1],)
        ordered_follows_ellipsis += (t[2],)
    for level in sorted(set(ordered_levels), reverse=True):
        print(level)
        max_level: int = len(set(ordered_levels))
    return Any #TODO


# TODO make type_tree system!!
# TODO does gettable[gettable[...]] work?

def _has_ellipsis(obj: object) -> bool:
    return Ellipsis in obj or ... in obj


# TODO use this more often, in cases where isinstance wont work and more, 
# also copy stuff (assert_type(s) from Mixin-test.py)
def _is_type_or_ellipsis(tp: type) -> bool:
    if tp is Ellipsis:
        return True
    if isinstance(tp, type):
        return True
    origin = get_origin(tp, allow_all=True)
    if origin is tuple:
        for arg in get_args(tp, allow_all=True):
            if not _is_type_or_ellipsis(arg):
                return False
        return True
    return False


class gettableMeta(type):

    __slots__: slots = ()

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._init()
        return instance

# Note: Mypy thinks that T isn't a subtype of type, which is wrong. #TODO ?????
@_Form
class gettable[*T](metaclass=gettableMeta):
    """ #TODO document error msg's
    Base-Class to give an object gettable behavior. We can define the expected type
    during inheritance, like:

        >>> class cls(gettable[int]):
        >>>     ...

    If the argument given when inheriting is not a type, we throw a TypeError.  
    If we got a single type, we expect results like this: 

        >>> class cls(gettable[int]):
        >>>     ...
        ...
        >>> cls()[1]
        >>> cls()["A"] # Throws a TypeError
        >>> cls()[1, 2] # Throws a TypeError

    We can also insert multiple arguments, like:

        >>> class cls(gettable[int, str, str, tuple[int, ...]]):
        >>>     ...
        ...
        >>> # The tuple can be of any length, as long as all arguments
        >>> # are of type int, or no arguments are given at all.
        >>> cls()[1, "A", "B", ()]
        >>> cls()[1, "A", "B", (1)]
        >>> cls()[1, "A", "B", (1, 2)]

    We can even use `...` to show that any amount 
    of the previously defined type can be given:

        >>> class cls(gettable[int, ...]):
        >>>     ...
        ...
        >>> cls()[1, 2, 3]

    When trying to define a gettable with any type of
    Argument being allowed, use `object`, like:

        >>> class cls(gettable[object]):
        >>>     ...
        ...
        >>> cls()[1]
        >>> cls()["A"]
        >>> ...

    or:

        >>> class cls(gettable[object, ...]):
        >>>     ...
        ...
        >>> cls()[1]
        >>> cls()[1, 2]
        >>> ...

    Doing `cls()[arg]` will always return `(arg,)`, 
    if it is a instance of the defined type.  
    This functionality can be ignored changed by doing:

        >>> class cls(gettable[int, ...], add_with_getitem=True): #TODO test this?!
        >>>     ...
        ...
        >>> print(cls()[1])
        (1,)
        >>> print(cls()[2, 3])
        (1, (2, 3),)

    Note that doing `cls[...]` will raise a TypeError, 
    as that syntax is reserved for the subclassing process.
    """#TODO support Unions???? (+typing.Union)

    # Note that there are a lot of type: ignore[...]
    # comments, as we do NOT include a `__init__` in 
    # this class, so classes that inherit from this 
    # will skip this class, when looking for `__init__`.


    # Somehow we cannot put arguments defined
    # in `_init` into a __slots__, likely 
    # because using __slots__ prevents the 
    # usage of __dict__. Therefore we cannot
    # create a __slots__ for gettable. #TODO

    _gettable_add_with_getitem_registry: dict[str, bool] = {}

    def _init(self) -> Instance:
        self.add_with_getitem: bool = gettable._gettable_add_with_getitem_registry[str(self)]
        self._instance_tp: tuple[tuple[Any, ...], ...]  = ()
        self._accept_none: bool = False
        self._is_generic: bool = True
        self._is_gettable: bool = True

    def _get_expected_for_tp(self, items: tuple[Any, ...]) -> type:
        failed = all_isinstance_cache[items]
        return self.tp[items.index(failed)] # type: ignore[attr-defined]

    #TODO: err, msg, var names, ... # del this line later
    def _can_resolve_gettable_types(
        self, 
        items: tuple[Any], 
        *, 
        recursive_types: tuple[type, ...] = _SENTINEL # type: ignore[assignment]
    ) -> None:
        if _no_arg_given(recursive_types): # type: ignore[arg-type]
            recursive_types = self.tp
        for index, (item, tp) in enumerate(zip(items, recursive_types)): # type: ignore[call-overload]
            if index > 0:
                prev_tp = recursive_types[index-1] # type: ignore[index]
                if prev_tp is Ellipsis: #TODO do we need dis?
                    prev_tp = recursive_types[0] # type: ignore[index]
            origin = get_origin(tp, allow_all=True)
            args = get_args(tp, allow_all=True)
            if origin is tuple:
                if args[-1] is Ellipsis:
                    expected_type = args[0]
                    passed = True
                    try:
                        if not all(isinstance(x, expected_type) for x in item): #TODO
                            passed = False
                    except TypeError:
                        if not all_isinstance(item, expected_type):
                            passed = False
                    if not passed: 
                        raise TypeError(f"expected tuple of {expected_type}, got {item}")
                else:
                    if not len(args) == len(item):
                        raise TypeError(f"expected {len(args)} arguments, got {len(item)}")
                # Recurse into the tuple
                self._can_resolve_gettable_types(item, recursive_types=args) # type: ignore[arg-type]
            else:
                if tp is Ellipsis:
                    tp = prev_tp
                if not isinstance(item, tp):
                    raise TypeError(
                        f"expected {_repr_type(tp)}, "
                        f"got {_hl_arg(item)} of type {_repr_obj(item)}"
                    )

    # We call the class argument `self` here, 
    # because it is treated the same way.
    # TODO can we actually get tuple[type]???
    #TODO test this stuff again!!!!!!!
    def __class_getitem__(self, tp: type  | tuple[type]) -> type[gettable[*T]]:
        print(self) 
        #TODO
        if not isinstance(tp, tuple):
            tp = (tp,)
        if (
            not Ellipsis in tp and
            not self.__name__ == "gettable"
        ):
            # Only happens, when we do
            # cls[...] instead of cls()[...].
            raise TypeError(
                f"Cannot use {self.__name__} in this context. Try "
                f"`{self.__name__}()[{_collect_type_params_repr(tp)}]` instead."
            ) from None
        if _has_ellipsis(tp):
            if not len(tp) == 2:
                # Error-Message similar to Pylance https://github.com/microsoft/pyright/
                raise TypeError("'...' is only allowed as the second of two arguments")
            if (
                not isinstance(tp, INSTANCE_TYPES)
                and not _is_type_or_ellipsis(tp)
            ):
                raise TypeError(
                    f"`class cls(gettable[{_collect_type_params_repr(tp)}]): ...`: arg must be one or more type(s), "
                    f"got {_trunc(str(tp[0]))}"
                )
            # Handles cases with 0 arguments given
            self._accept_none = True
        elif not all_isinstance(tp, ALLOWED_TYPES):
            raise TypeError(
                f"`class cls(gettable[{_collect_type_params_repr(tp)}]): ...`: arg must be one or more type(s), "
                f"got {all_isinstance_cache[tp]}"
            )
        self.tp = tp # type: ignore[attr-defined]
        self._instance_tp = ()
        return self

    def __getitem__(self, *items: *T) -> tuple[*T] | tuple[tuple[*T], ...]:
        try:
            self.tp # type: ignore[attr-defined]
        except AttributeError:
            raise TypeError("cannot use `getitem` of gettable object before defining the type")
        # Note that the len(items) > 0 check always returns true
        # at runtime, and is only used to satisfy Typecheckers.
        if len(items) > 0:
            if isinstance(items[0], tuple):
                items = items[0]
        if (
            not len(items) == len(self.tp) # type: ignore[attr-defined]
            and not _has_ellipsis(self.tp) # type: ignore[attr-defined]
        ):
            raise TypeError(
                f"got {len(items)} arguments, "
                f"expected {len(self.tp)}" # type: ignore[attr-defined]
            )
        passed = False
        if _has_ellipsis(self.tp) and len(items) == 2: # type: ignore[attr-defined]
            passed = all_isinstance(items, self.tp) # type: ignore[attr-defined]
        # Try if passed still is False,
        # otherwise we happily take it.
        if not passed:
            self._can_resolve_gettable_types(items) # type: ignore[arg-type]
            passed = True
        if not passed:
            raise TypeError(
                f"{_repr_obj(all_isinstance_cache[items])} {all_isinstance_cache[items]!r} is "
                f"not of expected type {_repr_type(self._get_expected_for_tp(items))}"
            )
        if self.add_with_getitem: # type: ignore[attr-defined]
            try:
                print(self)
                self._instance_tp += (items,) # type: ignore[attr-defined, has-type]
            except AttributeError:
                return (items,)
        return (items,)

    def __init_subclass__(cls, /, *, add_with_getitem: bool=False) -> None:
        name = get_cls_meta(cls)
        gettable._gettable_add_with_getitem_registry[name] = add_with_getitem
        super().__init_subclass__()

#@_Form
#class test(gettable[int, int, int]):
#   ...
#print(get_args(test)) #DOC #TODO
#print(get_origin(test)) #DOC #TODO
#TODO: FUCK, does this work
#todo make cls[...] work maybe???????
#class cls(gettable[int, int, str, tuple[int, ...]]):
#  ...

#print(cls()[1, 2, "A", (1,2,3)])
"""
class cls2(
    gettable[
        tuple[
            int, 
            tuple[
                int, 
                ...
            ], 
            tuple[
                tuple[
                    int
                ],
                ...
            ]
        ], 
        ...
    ]
):
    ...

cls2()[
        (
            1, 
            (
                2, 
                3, 
                4
            ), 
            (
                (
                    5,
                ),
                (
                    6,
                )
            )
        ), 
    (
        7,
        (), 
        ()
    )
]
"""
class cls3(gettable[int, int, str], add_with_getitem=True):
    ...
#print(cls3().__dict__)
#x = cls3()["A", 2, 3]
#print(cls3()[1, 2, "A"])
#print(cls3()[3, 4, "B"])

#exit()
#print(cls3.add_with_getitem)
#print(cls3().add_with_getitem)
#print(gettable._gettable_registry)
#exit()


#cls3()[1, 2, "A"]
#cls3()[3, 4, "B"]
#print(cls3.__dict__)
#print(cls3()[5, 6, "C"]) # ((1, 2, "A"), (3, 4, "B"), (5, 6, "C")) #TODO this should raise TP Err, bc we defined int, int, str not tuple[int, int, str], ...

#exit()
# ---

"""
Uses typing.py TypeVar, as specified in PEP 484, https://peps.python.org/pep-0484/
just modified to be able to take any number of arguments
NOTE: This version does NOT support any of:
`__covariants__`, 
`__contravariants__`
"""#TODO change that, add repr, ...
@_Form #TODO!!!
@supportsWeakref #TODO pyi file!!
class TypeVar:
    """
    We could use TypeVar here instead of importing it as t_TypeVar, 
    however using super() would make us use another function that is not really needed.

    NOTE: This version does NOT support any of:
    `__covariants__`, 
    `__contravariants__`
    """

    __slots__: slots = ("name", "tp", "bound")
    #TODO make the stuff with __new__
    def __init__(self, name: str, /, *tp: obj, bound: type | None = None) -> Instance:
        self.name = name
        self.tp = tp
        self.bound = bound

    def __call__(self) -> t_TypeVar:
        if not self.tp: 
            return t_TypeVar(self.name, bound=self.bound) # type: ignore
        elif len(self.tp) == 1: 
            return t_TypeVar(self.name, self.tp[0], self.tp[0], bound=self.bound) # type: ignore
        return t_TypeVar(self.name, *self.tp, bound=self.bound) # type: ignore

    def __repr__(self):
        if not self.tp: 
            return t_TypeVar(self.name, bound=self.bound).__repr__() # type: ignore
        elif len(self.tp) == 1: 
            return t_TypeVar(self.name, self.tp[0], self.tp[0], bound=self.bound).__repr__() # type: ignore
        return t_TypeVar(self.name, *self.tp, bound=self.bound).__repr__() # type: ignore

    #def __new__(): ...
    def __repr__(): ...
    def __or__(self, other): ...
    def __ror__(self, other): ...
    def __typing_subst__(): ...
    def __typing_prepare_subst__(): ...
    def __reduce__(): ...
    def has_default(): ...
    def __mro_entries__(): ...

@_Form
@supportsWeakref
class GenericTypeVar:
    """
    We could use TypeVar here instead of importing it as t_TypeVar, 
    however using `super()` would make us use another function that is not really needed.

    NOTE: This version does NOT support any of:
    `__covariants__`, 
    `__contravariants__`,
    """

    __slots__: slots = ("name", "bound")

    def __init__(self, name: str, /, *, bound=None) -> Instance:
        self.name = name
        self.bound = bound

    def __call__(self, name: str, /, *, bound=None) -> t_TypeVar:
        return t_TypeVar(self.name, bound=self.bound) # type: ignore

genericTV = GenericTypeVar

# -------

# Key Type
type _KT = str
"""Internal Type for Key's of dictionaries."""

KeyType = _KT

# Value Type
type _VT = Any

ValueType = _VT


_argsType = tuple #TODO depr?!
# Note that for some of the first classes Type Annotations might not be valid / complete, as the objects they would use are defined later


# Theoretically doing it via `from builtins import object as obj` would be smarter,
# however it would not properly show as a variable if we do it like that.
type obj = object


# For some reason method type is not yet builtin,
# but we can assume it to be a subtype of
# both `object` and `function`,
# but with that limited knowledge we cannot set the mro,
# which is why this is empty at the moment.
class method:
    """Type to annotate Method Type."""

# We could use `typing.Self`, 
# however we get a function and not an object, 
# if we would do that, which might break some Type-Checkers
@_Form
class Self:
    """
    Class to annotate the Type of Instances, being the argument `self`, or the Type of a `self` or `other` argument in a method.
    Note that using this to annotate the return Type of a method will show as a problem with Typecheckers. 
    Use the class name for methods that are not the `__init__` instead.
    """

# Used for functions that return the instance, such as `__init__`.
# Has to be of Type `NoneType`, because Type-Checkers try to have
# `__init__` methods return `None`.
type Instance = None
"""Type to annotate the return Type of methods like `__init__`, which return the instance, without needing to define a return statement"""

T = type
"""
Implementation of Any, where we want to show that the type matches another type. 
`T` can be any type, but should be of type `type`, even though there is no runtime checking of that.
"""

# For Caches
CacheKey = tuple[Any, ...] | tuple[Any, list[Any]]
CacheType = dict[CacheKey, Any]
IdCacheKey = tuple[int, int, int]
IdCacheType = dict[IdCacheKey, Any]
TypeCacheKey = tuple[T, ...] | tuple[T, list[T]]
TypeCacheType = dict[TypeCacheKey, Any]

@DEPR # Just a class with a big Docstring
@_Form
class _:
    """
    Implementation for a quick filler Argument, 
    should work the same as `fillerArgs` / `fillerArgsType`, 
    but they cant be used to inherit, like::

        >>> class cls(_):
        ...
        >>> def __init__(self, *args: _, **kwargs: _) -> None:
        >>>     ...
        ...
        >>> def __call__(self, *args: _, **kwargs: _) -> None:
        >>>     ...

    It can also be used as a return Type Annotator, which can be inherited::

        >>> class cls(_):
        ...
        >>> def __init__(self, *args, **kwargs) -> None:
        >>>     ...
        ...
        >>> def __call__(self, *args, **kwargs) -> _:
        >>>     ...
    """

    def __init__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Instance:
        pass

    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> _:
        return self

@DEPR # Nearly Unused, builtin could be used OR we add the stuff seen in TODO, then it could be cool
@_Form
class _Property: #TODO: class read-only, class write-only, class all, class None, outside ro, outside wo, outside all, outside None, -> @_Property(private=True)
    """Property similar to builtin `Property`, however made purely in Python, and not deprecated."""

    __slots__: slots = ("_getter", "_setter")

    def __init__(self, getter: Callable) -> Instance: #TODO callables annot
        self._getter = getter
        self._setter = None

    def __set__(self, instance: Self, value: T) -> None | Never:
        if self._setter is None:
            raise AttributeError(f"Cannot set Attribute {self._getter}.")
        self._setter(instance, value)

    def __get__(self, instance: Self, owner: Any | None=None, /) -> T:
        return self._getter(instance)

    def setter(self, setter: Callable | None) -> _Property:
        self._setter = setter
        return self


# New Construct
# It is the base of most further classes, 
# and can be subclassed to be used, 
# when doing `subclassable=True`, like:
#
# class New(NewConstruct, subclassable=True):
#   ...
# New Constructs cannot be called by default, 
# unless doing `callable=True` when subclassing,
# like:
#
# class New(NewConstruct, callable=True):
#   ...
class NewConstruct: #TODO
    """
    Generate a new internal construct, with given specifications. It cannot be passed as a class to inherit from and cannot be subclassed,
    unless doing `subclassable=True` when generating a new instance. NewConstructs cannot be called, unless doing `callable=True` when 
    generating a new instance.
    """

    _subclassable_registry: dict[object | None, bool] = {}
    
    __slots__: slots = ("args", "kwargs")

    def __init__(self, *args: _argsType[T], **kwargs: dict[str, Any]) -> Instance:
        print("nc.__init__")
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, is_callable: bool=False) -> Any:
        print("NC.__call__()")
        self.kwargs["callable"] = is_callable
        if not self.kwargs["callable"]: 
            raise TypeError(f"{self} is not callable.")
        return self.args

    def __repr__(self, /) -> str:
        return str(type(self))

    def __eq__(self, other, /) -> bool | Never:
        if isinstance(other, NewConstruct):
            return self.__repr__() == other.__repr__()
        raise TypeError(f"Cannot compare NewConstruct type with instance of {Type(other)}.")

    def __ne__(self, other, /) -> bool | Never:
        if isinstance(other, NewConstruct):
            return not self.__repr__() == other.__repr__()
        raise TypeError(f"Cannot compare NewConstruct type with instance of {Type(other)}.")

    def __hash__(self, /) -> int:
        return hash(self.__repr__)

    def __init_subclass__(self, /, *, subclassable: bool = False, callable: bool=False) -> None:
        print("NC init subcls") #dbg
        if self.__name__ in ("_Decorator", "_helper", "helper"):
            print("NC __init_subcls__")
            print("   self, sb, cb", self, subclassable, callable)
            print("   self.__bases__", self.__bases__) #dbg end
        if callable:
            ...
            print("is callable")
            #TODO
        super().__init_subclass__()
        if self.__bases__: 
            base_class = self.__bases__[0] 
        else: 
            base_class =  object
        if base_class and base_class in NewConstruct._subclassable_registry:
            if not NewConstruct._subclassable_registry[base_class]:
                raise TypeError(f"{base_class.__name__} is not subclassable.")
        NewConstruct._subclassable_registry[self] = subclassable

    def __subclasscheck__(self, subclass: obj, /) -> bool:
        if "subclassable" in self.kwargs:
            if self.kwargs["subclassable"]: 
                return True
        return False


@_Form
class Generator(NewConstruct, subclassable=True): #TODO: Inherit from iterator? #TODO depr, alr in Typing
    pass


@_Form #DEPR?
class Base(NewConstruct, subclassable=True):
    pass


@_Form
class CallableConstruct(NewConstruct, callable=True, subclassable=True):
    """Callable `NewConstruct` Subclass."""
#TODO slots
    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> _argsType[T]:
        return self.args

# Use the builtin `type`, but checks and converts 
# the arguments, before handing them over.
@_Form
class NewTypeBaseClass:
    """
    Class to inherit from when creating new Types.
    Per convention `NewTypeBaseClass` should be the last class that is being inherited from.
    """

    __slots__: slots = ("__type__",)

    def __init__(self, name: str, bases: type | tuple[type, ...], dict: dict[str, Any], **kwargs: dict[str, Any]) -> Instance:
        # We append `object` to the end, to 
        # start generating the `mro` faster.
        bases = pack(bases, object, add_to_existing=True)
        for base in bases:
            if (
                not base in _Final._allowed and 
                _got_finalized(base)
            ): #TODO does this work?
                raise TypeError(f"Invalid `bases`: {base} is of Type `_Final` and can therefore not be used in this context.")
        self.__type__ = bases
        b_type(name, bases, dict, **kwargs)


# Parents that can be used to give objects certain functionality
@DEPR # Unused
class Immutable:
    """Class that can be subclassed and inherited from, to help make making Immutable objects easier."""

    def __setattr__(self, name: str, value: Any, /) -> Never:
        raise TypeError("Cannot modify immutable object.")  


@DEPR
class Mutable:
    """Class that can be subclassed and inherited from, to help make making Mutable objects easier."""

    def __setattr__(self, name: str, value: Any, /) -> None:
        try:
            self.name = value
        except AttributeError:
            pass


@DEPR
class Iterable:
    """Class that can be subclassed and inherited from, to help make making Iterable objects easier."""

    @_Property
    def __type__(): #TODO make _Property work, make return annot
        return str | list | tuple | set | dict | range | bytes | bytearray | memoryview | enumerate | filter | zip

    #@TODO
    def __iter__(self, *args: _argsType[T], **kwargs: dict[str, Any]) -> Any:
        # `*args` should not be a `Iterable` subtype, therefore it works
        # without calling itself over and over
        return iter(args)

#@TODO # TODO test, also @TODO breaks when this class is subclassed lol
@_Form
class NotCallable:
    """Class that can be subclassed and inherited from, to help make making Not Callable objects easier."""

    def __call__(self, *args: Any, **kwargs: dict[str, Any]) -> Never:
        raise TypeError(f"Cannot call non-callable object {Type(self)}.")

@_Form
class _Decorator(NewConstruct, subclassable=True, callable=True):
    """
    Construct to make the production of Decorators easier, by just inheriting from it, and leaving the Decorator empty.
    Used like::

        >>> class wrapper(_Decorator):
        >>>     ...
        ...
        >>> @wrapper
        >>> def func(arg) -> int:
        >>>     return arg

    So that if we do `func(arg)`, it will return `arg`.
    """
    
    __slots__: slots = ("func",)
    
    def __init__(self, func: T, *args: _argsType, **kwargs: dict[str, Any]) -> Instance:
        print("Called _dec init, and not Nc init")
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__annotations__ = func.__annotations__
        print(">>>", func)
        
    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Any: # type: ignore[override]
        print(self.func)
        print("yeah finally")
        return self.func(*args, **kwargs)

    def __init_subclass__(self, *args: _argsType, **kwargs: dict[str, Any]) -> None:
        print("we are at _Decorator now", args, kwargs)
        CallableConstruct().__init_subclass__(subclassable=True, callable=True)

@_Form
class _Factory(_Decorator):
    """Internal decorator to annotate Factories."""

@_Form
class UnionType(NewTypeBaseClass): #TODO NewTypeBaseClass? -> What exactly? #TODO gettable # TOD: remove duplicates?, ###TODO!!!! rename to TypedUnion? #TODO make instance be TypedUNion and this _TypedUnionAlias
    """
    Type Annotation for Multiple Types, 
    which can be used to satisfy runtime type checks 
    and the Annotations of code, like::

        >>> class cls:
        ...
        >>>     def __call__(self, *args) -> UnionType[None, Never]:
        >>>         raise TypeError

    or::

        >>> class Height: 
        >>>     ...
        ...
        >>> def func(a: UnionType(Height, int, float)) -> bool:
        >>>     ...

    We can also use syntax like `UnionType[Height, int, float]`
    to achieve the same. Per Convention the first Type(s) should be 
    those that are used, and the last Type(s) should be those that 
    satisfy Type Checkers. These can be used to satisfy overloads at 
    runtime, although using them is not (yet) supported by 
    Typecheckers.

    Note:
    Using `UnionType` will sometimes show as an Error with 
    Typecheckers, and when using the `UnionType()` syntax,
    there will be a warning because of the Call Expression being used. 
    """

    __slots__: slots = ("tp",)

    def __init__(self, *tp: _argsType[T]) -> Instance:
        self.tp: tuple[T] = tp

    def __call__(self, *tp: _argsType[T]) -> _argsType[T]: # type: ignore[override]
        self.tp += tp
        return self.tp

    def __repr__(self) -> str:
        if self.tp == ():
            # Default to the __repr__ provided via the `@_Form` decorator
            return NotImplemented
            #TODO this doesnt default to the `@_Form` repr i guess
        strings: tuple[str, ...] = ()
        for t in self.tp:
            strings += (t.__name__,) # Use _repr_type()
        return f"Union[{" | ".join(strings)}]"

    # TODO
    def __getitem__(self, *tp: _argsType[T]) -> _argsType[T]:
        self.tp += tp
        return tp

    def __setitem__(self, *args: Any, **kwargs: dict[str, Any]) -> Never: 
        raise TypeError("Cannot set items for UnionType.")

    def __eq__(self, other: UnionType) -> bool | Never:
        if not issubclass(other, UnionType): #TODO do this similar to __or__??
            raise TypeError(f"Cannot compare UnionType type with instance of {_repr_obj(other)}.")
        return self.tp == other.tp

    def __ne__(self, other: UnionType) -> bool | Never:
        if not isinstance(other, UnionType):
            raise TypeError(f"Cannot compare UnionType type with instance of {_repr_obj(other)}.")
        return not self.tp == other.tp

    def __hash__(self) -> int:
        return hash(self.tp)

    def __or__(self, other: "UnionType") -> "UnionType":
        if not isinstance(other, UnionType):
            other = UnionType(other)
        return UnionType(self.tp, other.tp)

    def __ror__(self, other: "UnionType") -> "UnionType":
        if not isinstance(other, UnionType):
            other = UnionType(other)
        return UnionType(self.tp, other.tp)
#TODO add `__add__`, `__radd__`, ...

@_Form
class _set[T: gettable](NotCallable): ##################, NewTypeBaseClass): #TO DO test NotCallable subclassing --> ERROR, TypedUnion works tho??
    """ 
    Unique internal Type used by the `_Final` class. Similar to builtin `set` type.
    """

    __slots__: slots = ("_set",)

    def __init__(self, *obj: T) -> Instance:
        self._set: list[Any] = list()
        for o in obj:
            self.add(obj)

    def __contains__(self, *obj: Any) -> bool:
        return (
            obj in self._set or 
            unpack(obj) in self._set
        )

    def __str__(self) -> str:
        string = "{"
        string += " ,".join(item for item in self._set)
        return string + "}"

    #def __repr__(self) -> str:
    #    return f"{self.__name__}({" ,".join(item for item in self._set)})"

    def __getitem__(self, *index: int, _special: bool=False) -> obj:
        # In some special cases we accept the arguments and just return them
        if not _special:
            return self._set[index[0]]
        if not all_isinstance(index, ALLOWED_TYPES):
            raise TypeError(f"Expected type(s), got {_repr_obj(all_isinstance_cache[index])}")
        return index

    def __setitem__(self, *args: Any, **kwargs: dict[str, Any]) -> Never:
        raise TypeError("Cannot set item for Type `_set`.")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if not "_root" in kwargs:
            raise TypeError("Cannot subclass `_set`.")
        super.__init_subclass__()

    def add(self, obj: Any, /) -> None:
        for o in obj:
            if not o in self._set:
                self._set.append(o)

    def remove(self, item: Any, /) -> None:
        try:
            self._set.remove(item)
        except ValueError:
            # Item is not it the `_set`,
            # which would raise a `ValueError`,
            # if we were using the builtin `set` type.
            # In this case we just ignore it.
            pass

    def __class_getitem__(cls, item: type) -> type:
        return set[item] #TODO fix, #TODO make @tp_getitem decorator that checks params, rewrites annotations, and always gives tuple[type, ...]

@_Form
class TypedSet[T: type](_set, _root=True):

    __slots__: slots = ("_set",)

    def __init__(self, *tp: type) -> Instance:
        self._set: list[type] = list()
        for type in tp:
            self.add(type)

    def __contains__(self, *tp: type) -> bool:
        return (
            tp in self._set or 
            unpack(tp) in self._set
        )

    def add(self, *tp: type) -> None:
        print(type(tp))
        print(flatten(tp))
        print(type(flatten(tp)[0]))
        if not all_isinstance(flatten(tp), type):
            raise TypeError(
                f"Cannot add object '{all_isinstance_cache[tp]}' of type "
                f"{tp} to TypedSet, TypedSet only accepts arguments of type 'type'."
            )
        for t in tp:
            if not t in self._set:
                self._set.append(t)

    def __class_getitem__(cls, item) -> type:
        if not isinstance(item, validType):
            raise TypeError(...)
        return set[item]

    def __getitem__(self, item) -> type:
        return item

#TODO test deeeeez
#a = TypedSet(int, str)
#a.add(int, tuple[int, ...], ..., 3) # Should fail only at the 3

# Test if can be subclassed (shouldn't be allowed)
#class c(TypedSet):
#    ... #FAILS -> THAT WORKS!!

# This implementation does not use `_Decorator` as a parent, even though
# that would likely work, as we would need to overwrite all methods anyways.

@_Form
@supportsWeakref #TODO should it?
class _Final: #TODO
    """
    Parent used for objects that cannot be subclassed further::

        >>> class a(_Final):
        >>>     ...
        ...
        >>> # Throws TypeError
        >>> class b(a):
        >>>     ...

    Can also be used as a decorator, like::

        >>> @_Final
        >>> class a: 
        >>>     ...
        ...
        >>> # Throws TypeError
        >>> class b(a):
        >>>    ...

    NOTE: `_Final` cannot be used for functions, methods or attributes.
    If you want to finalize a variable, use `_FinalVar`.
    """

    __slots__: slots = ("__dict__", "__final__", "cls")

    _allowed: TypedSet[type] = TypedSet()

    def _error(*args: _argsType, **kwargs: dict[str, Any]) -> Never:
        raise TypeError(f"cannot subclass {_Final._allowed[-1].__name__}.") from None

    def _setattr(*args: _argsType, **kwargs: dict[str, Any]) -> Never:
        raise TypeError("Cannot set Attribute of _Final Type.")

    def __setattr__(self, name: str, value: obj, /) -> None | Never:
        print(f"_Final.__setattr__: {self=}, {name=}, {value=}")
        if isinstance(value, object) and not str(Type(value)) in ("<class 'function'>", "<class 'type'>"):
            print("yep", name, value)
            self.__dict__[name] = value
        return None

    # `*_` has been added, because sometimes we seem to get up to 4 args
    def __init__(self, cls: Callable[..., Any], /, *_) -> Instance: #TODO make AnyCallable type
        if not _is_class(cls):
            if callable(cls):
                raise TypeError(f"@_Final only works for classes")
            raise TypeError(f"cannot subclass class decorated with @_Final")
        # Set the `__final__` attribute to True,
        # so that `__subclasshook__` can tell that
        # the object had been finalized, even though
        # it did not inherit from `_Final`.
        setattr(self, "__final__", True)
        #print(f"{Type(cls)=},{cls=}")
        #print(f"_Final.__init__: {self=}, {cls=}, {_=}")
        # Only works for classes
        if type(cls) == type:
            self.cls = cls
        # Add cls to the list of allowed classes
        _Final._allowed.add(cls)
        if not isinstance(cls, type) or not callable(cls):
            if not hasattr(cls, "__call__"):
                _Final._allowed.remove(cls)
        #TODO is this code being used??
        try:  
            print(f"{type(_Final._allowed)=}, {_Final._allowed=}")
            self.cls = _Final._allowed[-1]
            print(f"{self.cls=}", cls)
            print("yay")
            if not self.cls == cls:
                raise TypeError(f"Cannot subclass {self.cls}")
        except (AttributeError, IndexError) as E:
            self.cls = cls
            print("Error within _Final.__init__:", E)
            print(cls, Type(cls))

    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Any:
        return self.cls(*args, **kwargs)

    def __init_subclass__(cls, *args: _argsType, **kwargs: dict[str, Any]) -> None:
        _Final._allowed.add(cls)
        print("UWUW")
        setattr(cls, "__init_subclass__", _Final._error)
        if _Final in self.__bases__:
            return super().__init_subclass__()
        if getattr(cls, "allow_root", False): #TODO document, use and stuff
            if "_root" in kwargs:
                return super().__init_subclass__()
        raise TypeError(f"Cannot Subclass {cls.__name__}")

@supportsWeakref
class _FinalClassVar: 
    """ 
    Type used for Class Variables, and therefore constants, like:

        >>> class numbers:
        >>>    pi: _FinalClassVar[float] = _FinalClassVar("pi", 3.14)
        ...
        >>> # Throws TypeError
        >>> numbers.pi = 2.71

    When defining a `_FinalClassVar` without giving a Value, is going 
    to work for the first assignment and afterwards it going to throw 
    Errors:

        >>> class numbers:
        >>>     pi: _FinalClassVar[float] = _FinalClassVar("pi")
        ...
        >>> # Doesn't throw an Error
        >>> numbers.pi = 3.14
        >>> # Throws TypeError
        >>> num.pi = 2.71
    """

    __slots__: slots = ("_assigned", "_name", "_value")

    def __init__(self, name: str, value: Any = _SENTINEL, /) -> Instance:
        self._name = name
        self._value = value
        if _no_arg_given(value):
            self._assigned = True
        else:
            self._assigned = False 

    def __get__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Any | None:
        if not self._assigned:
            return None
        return self._value

    def __set__(self, instance: Self, value: Any, /) -> None:
        if self._assigned:
            raise TypeError(f"Cannot set attribute for immutable object {self._name} of Type `_FinalClassVar`")
        self._value = value
        self._assigned = True

    def __repr__(self) -> str:
        if self._assigned:
            return f"{self._name} = {self._value}"
        return f"{self._name} = Undefined"

    def __str__(self) -> str:
        return f"{self._value}"

class OwnedSet(_set, _root = True):
    """
    Set that can only be mutated by the owner.
    """


    __slots__: slots = ("_owner", "_set")

    def __init__(self, owner: obj, /) -> Instance:
        self._owner = owner
        self._set: list[Any] = []

    def __add__(self, item: Any, /) -> OwnedSet:
        this = getframe(1)
        if not this == self._owner:
            raise TypeError("...")
        if not item in self._set:
            self._set.append(item)
        return self

    def remove(self, item: Any, /) -> None:
        try:
            self._set.remove(item)
        except ValueError:
            # Item is not it the `_set`,
            # which would raise a `ValueError`,
            # if we were using the builtin `set` type.
            # In this case we just ignore it.
            pass

    def __setitem__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Never:
        raise TypeError("Cannot set item for Type `OwnedSet`.")

    def __init_subclass__(self, *args: _argsType, **kwargs: Any) -> None:
        if not _get_key(kwargs, "_root", False):
            raise TypeError("Cannot subclass `OwnedSet`.")
        super.__init_subclass__()

class static():

    __slots__: slots = ()

    def __init__(self, name: str, tp: ALLOWED_TYPES=Any, /) -> Instance:
        if not _isinstance(tp, ALLOWED_TYPES):
            raise TypeError(f"Expected type, got {tp}")
        _static_registry.__add__({name: tp}) # Throws tp error "...", when it shouldnt?!

_static_registry: set[dict[str, ALLOWED_TYPES]] = OwnedSet(static)

#a = static("a")
#print(Type(a)) # Support get_args / get_origin #TODO
#a = 3 # Any
#a: int = 3 # TypeChecker (and rt?) Error?!
#b = static("b", int)
#b = 3 # Works
#b = "A" # Should be rt Error (and TC?)

#TODO -> __get__ for all (global get!)

#for i in range(100):
#  print()
#print(_Final._allowed)
#exit("defining cls2 didn't make any error ;(") #TODO here 

@_Form
class _helper(_Decorator, subclassable=True, callable=True): #TODO make only this file calls work (or test???)
    """
    Featureless Class used to annotate functions or classes that are intended to be used as helpers, either by other functions, or by classes. 
    Usually functions or classes with the helper annotation are not intended to be used by importing from other files::

        >>> @_helper
        >>> def func(): 
        >>>     ... # Do something a helper function does

    Or::

        >>> @_helper
        >>> class cls:  
        >>>     ... # Define multiple helper functions, which can have the `@_helper` decorator
    """

@_Form
#DEPR
class helper(_helper): #, _Final):
    """
    Featureless Class used to annotate functions or classes that are intended to be used as helpers, either by other functions, or by classes.
    They can be used like::

        >>> @helper
        >>> def func(): 
        >>>     ... # Do something a helper function does

    Or::

        >>> @helper
        >>> class cls:  
        >>>    ... # Define multiple helper functions, which can have the `@helper` decorator
    """

# Helpers 

@helper
def validObjectCheck(obj: obj) -> bool:
    return isinstance(obj, object)

@helper
def _hasattr(obj: obj, name: str) -> bool:
    """
    Internal helper to check wether an object has a given attribute, 
    with more fallbacks than the default implementation.
    """

    try:
        return getattr(obj, name, False)
    except TypeError:
        return hasattr(obj, str(name))

@helper
def is_func(obj: obj) -> bool:
    """
    Helper to check wether a given `object` type is a valid function.
    Will return False for classes or methods.
    """

    return not isinstance(obj, type) and isinstance(obj, object) and not is_meth(obj)

@helper #TODO does this work?
def is_meth(obj: obj) -> bool:
    """
    Helper to check that for a given `obj`, like `a.b`, `b` is a valid method.
    It can be used like::
    
        >>> class cls:
        ...
        >>> def meth():
        >>>     ...
        ...
        >>> print(is_meth(cls.meth)) 
        True
    """

    cls = str(obj).split(" ")[1].split(".")
    return isinstance(globals().get(cls[0]), type) and _hasattr(globals().get(cls[0]), cls[1])

@helper
def get_name(obj: type) -> str:
    """Returns the name of the given object"""
    try:
        return str(obj).split()[1].removeprefix("'").removesuffix("'>")
    except IndexError:
        raise TypeError(f"Cannot get the name of type {Type(obj)}") from None


#print("-"*150)
#print(UnionType.__mro__)
#print(UnionType.mro())
#i = int()
#print(i.__type__)
#exit()
@TODO
class Not: #TODO
    """
    Can be used to show ...:

        def f(a: Not[None]):
            ...

        f("Hi")
        f(1)
        f([1, 2])
        ...

        f(None) # Will not raise any Errors at runtime, but would be a problem (make it raise Error / warning? -> TypeWarning )
    """


#print("before calling")
#print(is_obj(int))
"""
def w(func): #TODO

  def wrapper(*args, **kwargs):
    return func(*args, **kwargs)
    
  wrapper.__name__ = func.__name__
  wrapper.__doc__ = func.__doc__
  wrapper.__annotations__ = func.__annotations__
  return wrapper
"""
#@w
#def my_func(a: int, b: str) -> bool:
#  """HI"""
#  pass

#print(my_func(1, "hi"))
#exit("after call")


#TODO #Getitem returns Never | item #?
#TODO depr, this is wrong too, NEver -- Only NEVER
type CanRaiseError = Never | Any

@DEPR #or do something with it
class OwnReturnType: #TODO
  """
  Can be used in cases where we want to return the type of the class, function or outer class. Can be used the same way _self is.
  Sadly using a `__call__` function in a type annotation will show as warning, it will only return a type though, which is valid::

    class cls:
    
      def func() -> OwnReturnType("cls", enforced=False): 
        ...

  Instead of::

    clsReturnType = TypeVar("cls", ...)

    class cls:
      
      def func() -> clsReturnType:
        ...
        
  or::
  
    class cls:
    
      def func() -> cls:
        ...
  """
  
  def __init__(self, name: str, /, *, enforced: bool = True) -> Instance:
    self.__name__ = name
    self.__enforced__ = enforced
    self.__call__()

  def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Any | CanRaiseError:
    #TODO
    print(self.__name__)
    if not self.__enforced__: 
      return TypeVar(self.__name__)
    if (self.__name__ in globals() or 
        self.__name__ in locals() or 
        self.__name__ in {"int","float","str","None","bool","list","set","dict","obj","bytes"}):
      return TypeVar(self.__name__, self.__name__)
    if Type(self.__name__) == object: 
      return TypeVar(self.__name__, self.__name__) 
    raise TypeError(f"{self.__name__} is not a valid object")

#TODO
"""  
class cls:
  def func2() -> OwnReturnType("int", enforced=True): ...

  def func3() -> OwnReturnType("cls", enforced=True): ...


cls.func3()
exit() 
"""

@DEPR
class fillerArgs(object):
  """
  Featureless class, used to annotate and define fillerArgs, which can be used to give as parameter to classes or functions,
  used in cases the user might pass any arguments, and used like::

    def func(a, *fillerArgs: fillerArgsType) -> obj | TypeNotDefinedType:
      return a, *fillerArgs

  or::

    class cls:
      _fillerArgs = fillerArgs()

      def __init__(self, *fillerArgs=_fillerArgs) -> obj: 
        return self

  Can also be used as return Type::

    def func(*fillerArgs: fillerArgsType) -> None | fillerArgsType:
      # Returns only arguments that were given when called
      return *fillerArgs
  
  And in use with the __call__ dunder method, where we can replace any *args with *fillerArgs: fillerArgsType::

    class cls:
    
      def __init__(self, *fillerArgs: fillerArgsType) -> None:
        pass 
        
      def __call__(self, *fillerArgs: fillerArgsType) -> fillerArgsType:
        return *fillerArgs
  """
  
  def __init__(self, *args: _argsType) -> Instance:
    pass

  def __call__(self, *args: _argsType) -> Any:
    return self

@DEPR 
class typeNotDefinedType: 
  """
  Used to annotate Arguments or Return Types where the Type is not yet defined::

    def func(arg: typeNotDefinedType) -> None:
      pass
      
  or::

    arg: typeNotDefinedType = ...

  or::

    def func() -> typeNotDefinedType:
      return ...

  being the same as::

    def func(arg: typeNotDefinedType) -> typeNotDefinedType:
      return arg
      
  NOTE: Typecheckers will show using this as an Error
  """
  
  def __init__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Instance:
    pass
  
  def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> Any:
    return self

@DEPR
class multipleTypes(object): #TODO remove object here, as MyPy will complain #######TODO TypeVarTuple?!?!??
  """
  Used to annotate Variables that can be one of multiple Types::

    def func(arg: multipleTypes) -> None: 
      pass
      
  or::

    xTypeVar: multipleTypes = TypeVar("xTypeVar", ..., ...)

  Returns a TypeVar of name self with arguments self.__data__ when called.
  """

  def __init__(self, *args: _argsType) -> Instance: 
    self.__data__: list = []

  def __iter__(self, *args: _argsType) -> typeNotDefinedType:
    return iter(self.__data__)
  
  def __repr__(self, *args: _argsType) -> str:
    return str(self.__data__[i] for i in self.__data__)

  def __call__(self, *args: _argsType) -> obj:
    if args:
      self.__data__.extend(*args)
    return TypeVar(f"{self}", (self.__data__))
  
@DEPR
class MultipleTypesEnforced(object): #TODO: redo?
  """
  Used to annotate Variables that have to be one of multiple Types::

    def func(arg: multipleTypesEnforced) -> None: 
      pass
      
  or::

    xTypeVar: multipleTypesEnforced = TypeVar("xTypeVar", ..., ...)
  
  Returns nothing when called, will raise a TypeError if the wrong Type is used or if an argument is not valid Type whilst initializing.
  """

  def __init__(self, *allowedTypes, baseType = _SENTINEL) -> Instance:
    """
    Initiate the new Object, using the parameters `*allowedTypes`and `baseType`, where `baseType` is prioritized above `*allowedTypes`.
    If `baseType` is equal to the id of `ArgumentNotGiven()` we will return None, as that is used to check if no Argument was given, as `type(None)` can't be used as it is a valid type
    We could use `notDefined` to annotate the baseType and the default Argument, however this just helps to have more clarity
    """
    if not allowedTypes and _no_arg_given(baseType): return None
    if not isinstance(baseType, type): 
      raise TypeError(f"{Type(baseType)!r} is not a valid BaseType")
    # Check each Type given for validity
    for aT in allowedTypes:
      if not isinstance(aT, type): 
        raise TypeError(f"{Type(aT)!r} is not a valid Type")
    self.baseType = baseType
    self.allowedTypes = allowedTypes

  def __call__(self, obj) -> CanRaiseError:
    """
    Validate the object during instantiation, throw an Error if we fail
    """
    if not isinstance(obj, (self.baseType, self.allowedTypes)):
      raise TypeError(f"Object must be an instance of {Type(self.baseType)!r} or any of {Type(self.allowedTypes)!r}")

"""All show as Problems, however none are at runtime, as we return valid Types for each"""
"""Most Variables here are objects used for Type Hints, as introduced in Python 3.9"""
"""We will delete these Type Hint Variables later, as they are only intended to be used in this script"""
typeType: obj = object # All Types are just objects (classes)
_t: obj = typeType

type _T = object
_T_v: _T | typeNotDefinedType = type(TypeVar("")) # Object
_T_vSingle: _T | typeNotDefinedType = type(TypeVar("")) # Same as _T_v
type _T_vMultiple =  multipleTypes
type _T_vMultipleEnforced = MultipleTypesEnforced

type fillerArgsType = fillerArgs

_i: _T = int
_f: _T = float 
_accurateFloat: _T | typeNotDefinedType = ... #TODO
_aF: _T = _accurateFloat 
_l: _T = list
_tuple: _T = tuple
_b: _T = bool

#DEPR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
type expr = str | bytes | None

lambdaType: _T = Type(lambda a: a)

_N: _T = Type(None)
"""Shows as Problem in some cases, however type(None) returns NoneType, which is valid in this case"""

_E: _T = type(...)

_text: _T_vMultiple = TypeVar("_text", str, bytes, None)


@helper
def is_expr(expression: expr) -> bool:
  """Helper to check wether a given expression is of type `expr`."""
  return isinstance(expression, str, bytes, None)

#TODO DEPR these??
# Generic


class GenericError(Exception):
  """
  Base Class for all Generic Errors.
  """

# Import related Errors


class ModuleError(Exception):
  """ 
  Base Class for all Module Errors.
  """

class PackageError(Exception):
  """
  Base Class for all Package Errors.
  """

# Class / Function related Errors

class ClassError(Exception):
  """ 
  Base Class for all Object Errors (Class Errors). Is the same as using ObjectError.
  """
  
class FunctionError(Exception):
  """
  Base Class for all Errors that happen inside a Function.
  """  
  
class InitSubclassError(Exception):
  """ 
  Base Class for all Errors related to (trying to) subclassing a Class.
  """
  

class MethodError(Exception):
  """ 
  Base Class for all Errors that happen inside a Method.
  """

class ObjectError(Exception):
  """ 
  Base Class for all Object Errors. Is the same as using ClassError.
  """


try:
    raise TypeError()
except TypeError as _error:
    tb_Type = type(_error.__traceback__)
    f_Type = type(_error.__traceback__.tb_frame)

    """Delete _error Variable as it conflicted with function defined later, also it is not needed later on"""
    del _error

type ErrorType = BaseException #TODO does type(Error()) work better?
"""Type of Errors, using `BaseException`."""

def _r(*args: _argsType, **kwargs: dict[str, Any]) -> Literal["_r"]: 
    return "_r"

def __repr_self__(self: Any, *args: _argsType) -> str | Any:
    try:
        if args[0] == _r(): 
            return f"Self: {self.__repr__()}"
        else: 
            return self
    except (IndexError, AttributeError):
        return self

class catch(_Decorator): #, _Final): #TODO, seems to not work
    """ 
    Decorator to make the Function or Method decorated with it not exit the program if it raises an Error
    Can be used like::

        @catch
        def func(a: int) -> int:
            if a < 0: 
                raise ValueError("Value too small")
        return a

    or::

        class cls:

        @catch
        def func(self):
            raise TypeError("Cannot use `func`")

    The optional Argument `bound` can be set to a collection of Errors, for which it will check, and if another Error is raised, it will raise that.
    NOTE: If a Function / Method is called, and it raises an Error, the function returns None
    """

    def __init__(self, func, *bound: ErrorType | tuple[ErrorType] | None) -> Instance:
        self.func = func
        self._bound = bound

    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> UnionType(Any, None, CanRaiseError):
        print(self._bound, self)
        try:
            if "bound" in kwargs:
                self._bound = kwargs["bound"]
        except AttributeError:
            pass
        try:
            return self.func(*args, **kwargs)
        except Exception as E:
            if self._bound is None or Type(E) == self._bound:
                return None 
        try: 
            if Type(E) in self._bound:
                return None
        except TypeError: 
            raise E from None
        return None

    class set_bound:

        def __init__(self, func, *bound: ErrorType | tuple[ErrorType] | None) -> Instance:
            self.func = func
            print(f"setting bound for {self.func} to", bound)
            print(f"{self=}")
            self._bound = bound
            print(self._bound)

        def __call__(self, *bound: ErrorType | tuple[ErrorType] | None) -> Any:
            print("setting bound to", bound)
            print(f"{self=}")
            self._bound = bound
            print(self._bound)
            return self.func()

class onException(_Decorator, subclassable=True):
    """Decorator similar to the catch decorator, but executes given function on Exception."""

    def __init__(self, func, *bound: ErrorType | tuple[ErrorType] | None, other=None) -> Instance:
        self.func = func
        self.bound = bound
        self.other = other
        self.__name__ = self.func.__name__
        self.__doc__ = self.func.__doc__
        self.__annotations__ = self.func.__annotations__

    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> UnionType(Any, None, CanRaiseError):
        try:
            return self.func(*args, **kwargs)
        except Exception as E:
            if self.bound is None or Type(E) == self.bound:
                return self.other(*args, **kwargs)
        try: 
            if Type(E) in self.bound:
                return self.other(*args, **kwargs)
        except TypeError: 
            raise E from None
        return self.other(*args, **kwargs)

    @classmethod
    def set_bound(self, *bound: ErrorType | tuple[ErrorType] | None) -> lambdaType:
        return lambda func: self(func, bound)

    @classmethod
    def set_other(cls, other: Callable | None=None) -> lambdaType:
        return lambda func: cls(func, other)

class onError(onException): #, _Final):
    pass


#@catch(bound=TypeError)
#def func(a: int) -> bool:
#  """some doc"""
#  print("func got called")
#  raise TypeError("e")

#print(func(1))
#exit()

class descriptor: #TODO #TODO make docstring use repl formating
    """
    Creates a new descriptor, used to annotate Variables and return Types in more detail::

        >>> def func(a: descriptor(int, description="Price in €", default=33)) -> descriptor(bool, int, description="Bool to tell wether the price is low enough"):
        >>>     return a < 35 # Maximum price is 35€

    or::

        >>> x = descriptor(int, description="Number of Houses in a street", default=100) 
        ...
        >>> number: x = 10

    which is the same as:

        >>> number: descriptor(int, description="Number of Houses in a street", default=100) = 10 

    The `default` argument does not get enforced at runtime, neither do we get a TypeError when using a type not included in `parents`.
    Doing::

        >>> x = descriptor(int)
        ...
        >>> print(type(x))
        descriptor
        ...
        >>> print(x.__repr__())
        int #TODO make this use int.__repr__??

    NOTE that doing::

        >>> x = descriptor(aType, bType)

    is the same as::

        >>> x = descriptor((aType, bType))
    """

    __slots__: slots = ("__parents__", "__description__", "__default__", "__annotations__")

    def __set_doc_helper__(self, description: _text=None) -> None:
        try:
            setattr(obj, "__doc__", description)
        except TypeError: pass
        except AttributeError: pass

    def __init__(self, *parents: tuple[type], description: _text=None, default=None) -> Instance: 
        self.__parents__ = parents
        self.__description__ = description
        self.__default__ = default
        self.__annotations__ = {str(self): str(self.__parents__)}
        self.__set_doc_helper__(description=self.__description__)

    def __call__(self, *args: _argsType, **kwargs: dict[str, Any]) -> None | tuple[type]:
        if _is_empty(self.__parents__): 
            return None
        return self.__parents__

    def __getitem__(self, *item: tuple[type]) -> None | type:
        self.__parents__ = item
        if _is_empty(self.__parents__):
            return None
        return self.__parents__

    def __setitem__(self, *args: _argsType, **kwargs: dict[str, Any]): 
        raise TypeError("Cannot set items for descriptor type")

    def __repr__(self) -> str:
        return f"<descriptor with {self.__parents__}>" #TODO redo

    def __or__(self, other: obj) -> descriptor:
        if not Type(other) == descriptor: 
            raise TypeError(f"Cant combine type descriptor with type {Type(other)}") 
        return descriptor(
            self.__parents__ | other.__parents__, 
            description=self.__description__ | other.__description__, 
            default=self.__default__ | other.__default__
        )

    def __ror__(self, other: obj) -> descriptor:
        return self.__or__(other, self)

    def __eq__(self, other) -> bool | Never:
        if isinstance(other, descriptor):
            return (
                self.__parents__ == other.__parents__ and 
                self.__description__ == other.__description__ and 
                self.__default__ == other.__default__
            )
        raise TypeError(f"Cannot compare descriptor type with instance of {Type(other)}")

    def __ne__(self, other) -> bool:
        if isinstance(other, descriptor):
            return (
                not self.__parents__ == other.__parents__ and
                not self.__description__ == other.__description__ and
                not self.__default__ == other.__default__
            )
        raise TypeError(f"Cannot compare descriptor type with instance of {Type(other)}")

    def __hash__(self) -> int:
        return hash(self.__parents__)

def test_func(a: descriptor(int, float, description="Price in €", default=33)) -> descriptor(bool, description="Bool to tell us wether the price is low enough for us to buy it", default=...):
    return a < 35 # Maximum price is 35€

#print(Type(descriptor[int, float]))
#exit()

"""
test = descriptor(int, description="Number of Houses in a street", default=100)

print(Type(test_func(37)))
print(f"{Type(test_func(37))!r}")
exit()"""

#class constants: 
"""
    Defines most mathematical and physical constants needed, and gives 100 digits of accuracy if not followed with all Zeros

    Defined are:
        - π (pi)
        - e
        - γ (gamma)
        - g  -> m/s²
        - c -> m/s

    They can be accessed by doing:

        print(constants.pi)

    They should not be changed, and a TypeError is raised on runtime if they are.
    """
""" #TODO, DEPR??
  PI: _FinalClassVar[_aF] = _FinalClassVar("PI", 3.141_592_653_589_793_238_462_643_383_279_502_884_197_169_399_375_105_820_974_944_592_307_816_406_286_208_998_628_034_825_342_117_067_9)
  E: _FinalClassVar[_aF] = _FinalClassVar("E", 2.718_281_828_459_045_235_360_287_471_352_662_497_757_247_093_699_959_574_966_967_627_724_076_630_353_547_594_571_382_178_525_166_427_4)
  GAMMA: _FinalClassVar[_aF] = _FinalClassVar("GAMMA", 0.577_215_664_901_532_860_606_512_090_082_401_065_544_928_165_198_272_091_766_364_290_228_109_101_407_756_373_084_209_040_708_354_917_8)
  G: _FinalClassVar[_aF] = _FinalClassVar("G", 9.806_65) # m/s²
  C0: _FinalClassVar[_i] = _FinalClassVar("C0", 299_792_458) # m/s

  pi = PI
  e = E
  gamma = GAMMA
  g = G
  c0 = C0

  # In Python 3.14.0a4 and above we should have these being highlighted correctly and being able to use them as Constants
  π: _f = pi
  γ: _f = gamma"""


#TODO depr or use or something, this dont feel right tho lol

class empty(_Decorator): 
    """
    Used to annotate an empty or featureless function / class following:

        @empty
        class cls:
            ...

    or:

        @empty
        def func():
            ...
    """

stub = empty

#TODO: 
# Make `x: constant` type decorator!!!!! ####Alr done iirc
# Make __radd__ for all types (useful for x + customType), __iadd__ and __div__, __mul__, ... -> Same with bitwise 
# Make all vars be either xxxxx or xxxxYyyyy but not Xxxxxx #TODO
# Make descriptions for all stuff
# MAKE A MODIFY-IMMUTABLES OBJECT

#TODO: str(), int(),... FUNCTIONS IN THE CLASS ---> __int__, __str__, __bool__, ...
#DEL??
Types: dict[type, dict[str, bool]] = {
    int: {
        "mutable": False,
        "indexable": False,
        "callable": False,
        "iterable": False,
        "hashable": True,
        "numeric": True,
    },
    float: {
        "mutable": False,
        "indexable": False,
        "callable": False,
        "iterable": False,
        "hashable": True,
        "numeric": True,
    },
    complex: {
        "mutable": False,
        "indexable": False,
        "callable": False,
        "iterable": False,
        "hashable": True,
        "numeric": True,
    },
    bool: {
        "mutable": False,
        "indexable": False,
        "callable": False,
        "iterable": False,
        "hashable": True,
        "numeric": True,
    },
    str: {
        "mutable": False,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": True,
        "sequence": True,
    },
    list: {
        "mutable": True,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": False,
        "sequence": True,
    },
    tuple: {
        "mutable": False,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": True,
        "sequence": True,
    },
    dict: {
        "mutable": True,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": False,
        "mapping": True,
    },
    set: {
        "mutable": True,
        "indexable": False,
        "callable": False,
        "iterable": True,
        "hashable": False,
        "set": True,
    },
    frozenset: {
        "mutable": False,
        "indexable": False,
        "callable": False,
        "iterable": True,
        "hashable": True,
        "set": True,
    },
    bytes: {
        "mutable": False,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": True,
        "binary": True,
    },
    bytearray: {
        "mutable": True,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": False,
        "binary": True,
    },
    memoryview: {
        "mutable": False,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": False,
        "binary": True,
    },
    range: {
        "mutable": False,
        "indexable": True,
        "callable": False,
        "iterable": True,
        "hashable": False,
        "sequence": True,
    },
    type(None): {               # We cannot do `None`, so this needs to fallback to `type(None)`
        "mutable": False,
        "indexable": False,
        "callable": False,
        "iterable": False,
        "hashable": True,
    },
    type: {
        "mutable": False,
        "indexable": False,
        "callable": True,
        "iterable": False,
        "hashable": True,
    },
}


if not __name__ == "__main__":
    del _t
    del _T, _T_v, _T_vMultiple, _T_vMultipleEnforced
    del _i, _f, _accurateFloat, _aF, _l, _tuple, _b
    del _N 
    del _E 
    del _text

#TODO: Make all raise ...("...") be similar / same style
# No Dots, exclamation marks etc, ...
#TODO: Make all be:
# - classes: class abCd: ... # NOT class AbCd(): ..., NOT class AbCd(): ..., ...
# - classes that only consist of one name: class ab: ...
# - classes that are Errors: class AbError: ...
# --> Exceptions are any internal classes that are meant to be deleted after usage, _classReference and classes that are changed versions from other libs
# - functions: def abCd(): ...
# - functions that only consist of one name: def ab(): ...
# - dunder functions: def __ab_cd__(): ... 
# - dunder functions that only consist of one name: def __ab__(): ...
# --> Exceptions are any internal helper functions that are meant to be deleted after usage, _funcReference and functions that are changed versions from other libs
# - Constants: AB
# - variables: ? #TODO
# - variables that are used in a TypeVar like manner: NO REAL NORM YET #TODO
# - dunder variables: __ab_cd__ = ...
# - dunder variables that only consist of one name: __ab__ = ...
#TODO: BACKWARDS COMPATIBILITY, PEP with 
#TODO: 3.14.0a4, new exceptions, new files / versions, 
#TODO: make a def createNewEmpty(Name: str) -> obj
#TODO: Dont use ... for empty functions / objects, use pass! Only use ... to show a function that is being worked on, or use it in code examples / docstrings
#TODO: Make all ...: obj be ...: type if needed

#TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Make all methods decorated automatically have the _ignore_staticmethod_check set to True (unless ...?), 
# by having the _Decorator class do that (or the class of the decorator itself, if needed!)

#TODO: Make all return annotations with | in between in order they appear in code:
# def func() -> int | float:
#     if something: return int
#     return float
#
# INSTEAD OF:
# def func() -> float | int: 
#     ... #Same code

# IF we call another func do it like done in said func!

# PEP? --> TypeVArTuple??!
# def f() -> *int: ... #RETURN ANY AMOUNT OF INTS

# str, int, ... appends and removes!
# __type__ attribute, like:
# class newInt:
#   __type__ = int

#MAX: 1395; 1431; 1454; 1589; 1644; 1704; 1735; 1694 --> FINALLY WE ARE GOING DOWN, 1683 because deleting unneeded stuff, 1681 even tho i just added stuff :D, 1796, 1862, 1997, PERFECT 2000, 1990, 2008, 2105, 2093, 2100, 2222, 2301, 2341, 2629, 2630, 2711, 2709, 2794, 2806, prolly even more, 2197!!! we going down! (i deprecated like half the code, so no wonder), 2602, 3050!!!!!!!
# Goal: 5763 or 6537