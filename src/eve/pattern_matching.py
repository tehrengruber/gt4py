# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import inspect
from dataclasses import dataclass
from functools import singledispatch
from types import ModuleType
from typing import Any, Iterator


@dataclass(frozen=True)
class ObjectPattern:
    """
    Class to pattern match general objects.

    A pattern matches an object if it is an instance of the specified
    class and all attributes of the pattern (recursively) match the
    objects attributes.

    >>> class Foo:
    ...    def __init__(self):
    ...        self.bar = 1
    ...        self.baz = 2
    >>> assert ObjectPattern(Foo, {"bar": 1}).match(Foo())
    """

    cls: type
    attrs: dict[str, Any]

    def match(self, other: Any, raise_exception: bool = False) -> bool:
        """Return if object pattern matches `other` using :func:`get_differences`.

        If `raise_exception` is specified raises an exception with all differences
        found.
        """
        if raise_exception:
            diffs = [*get_differences(self, other)]
            if len(diffs) > 0:
                diffs_str = "\n  ".join(
                    [f"  {self.cls.__name__}{path}: {msg}" for path, msg in diffs]
                )
                raise ValueError(f"Object and pattern don't match:\n  {diffs_str}")
            return True

        return next(get_differences(self, other), None) is None

    def __str__(self) -> str:
        attrs_str = ", ".join([f"{str(k)}={str(v)}" for k, v in self.attrs.items()])
        return f"{self.cls.__name__}({attrs_str})"


@dataclass(frozen=True)
class ObjectPatternConstructor:
    cls: type

    def __call__(self, **kwargs: Any) -> ObjectPattern:
        return ObjectPattern(self.cls, kwargs)


@singledispatch
def get_differences(a: Any, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Compare two objects and return a list of differences.

    If the arguments are lists or dictionaries comparison is recursively per item. Objects are compared
    using equality operator or if the left-hand-side is an `ObjectPattern` its type and attributes
    are compared to the right-hand-side object. Only the attributes of the `ObjectPattern` are used
    for comparison, disregarding potential additional attributes of the right-hand-side.
    """
    if type(a) != type(b):
        yield (path, f"Expected a value of type {type(a).__name__}, but got {type(b).__name__}")
    elif a != b:
        yield (path, f"Values are not equal. `{a}` != `{b}`")


@get_differences.register
def _(a: ObjectPattern, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if not isinstance(b, a.cls):
        yield (
            path,
            f"Expected an instance of class {a.cls.__name__}, but got {type(b).__name__}",
        )
    else:
        for k in a.attrs.keys():
            if not hasattr(b, k):
                yield (path, f"Value has no attribute {k}.")
            else:
                yield from get_differences(a.attrs[k], getattr(b, k), path=f"{path}.{k}")


@get_differences.register
def _(a: list, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if not isinstance(b, list):
        yield (path, f"Expected list, but got {type(b).__name__}")
    elif len(a) != len(b):
        yield (path, f"Expected list of length {len(a)}, but got length {len(b)}")
    else:
        for i, (el_a, el_b) in enumerate(zip(a, b)):
            for diff in get_differences(el_a, el_b, path=f"{path}[{i}]"):
                yield diff


@get_differences.register
def _(a: dict, b: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if not isinstance(b, dict):
        yield (path, f"Expected dict, but got {type(b).__name__}")
    elif set(a.keys()) != set(b.keys()):
        a_min_b = set(a.keys()).difference(b.keys())
        b_min_a = set(b.keys()).difference(a.keys())
        if a_min_b:
            missing_keys_str = "`" + "`, `".join(map(str, a_min_b)) + "`"
            yield (
                path,
                f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are missing: {missing_keys_str}",
            )
        if b_min_a:
            extra_keys_str = "`" + "`, `".join(map(str, b_min_a)) + "`"
            yield (
                path,
                f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are extra: {extra_keys_str}",
            )
    else:
        for k, v_a, v_b in zip(a.keys(), a.values(), b.values()):
            yield from get_differences(v_a, v_b, path=f'{path}["{k}"]')


@dataclass(frozen=True)
class ModuleWrapper:
    """
    Small wrapper to conveniently create `ObjectPattern`s for classes of a module.

    Example:
    >>> import foo_ir  # doctest: +SKIP
    >>> foo_ir_ = ModuleWrapper(foo_ir)  # doctest: +SKIP
    >>> assert foo_ir_.Foo(bar="baz").match(foo_ir.Foo(bar="baz", foo="bar"))  # doctest: +SKIP
    >>> assert not foo_ir_.Foo(bar="bar").match(foo_ir.Foo(bar="baz", foo="bar"))  # doctest: +SKIP
    """

    module: ModuleType

    def __getattr__(self, item: str) -> ObjectPatternConstructor:
        val = getattr(self.module, item)
        if not inspect.isclass(val):
            raise ValueError("Only classes allowed.")
        return ObjectPatternConstructor(val)
