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
from typing import Any
from types import ModuleType
from dataclasses import dataclass

def get_differences(a: Any, b: Any, path: str = ""):
    """Compare two objects and return a list of differences.

    If the arguments are lists or dictionaries comparison is recursively per item. Objects are compared
    using equality operator or if the left-hand-side is an `ObjectPattern` its type and attributes
    are compared to the right-hand-side object. Only the attributes of the `ObjectPattern` are used
    for comparison, disregarding potential additional attributes of the right-hand-side.
    """
    if isinstance(a, ObjectPattern):
        if not isinstance(b, a.cls):
            yield [path, f"Expected an instance of class {a.cls.__name__}, but got {type(b).__name__}"]
        else:
            for k in a.attrs.keys():
                if not hasattr(b, k):
                    yield [path, f"Object has no attribute {k}."]
                else:
                    for diff in get_differences(a.attrs[k], getattr(b, k), path=f"{path}.{k}"):
                        yield diff
    elif isinstance(a, list):
        if not isinstance(b, list):
            yield [path, f"Expected list, but got {type(b).__name__}"]
        elif len(a) != len(b):
            yield [path, f"Expected list of length {len(a)}, but got length {len(b)}"]
        else:
            for i, (el_a, el_b) in enumerate(zip(a, b)):
                for diff in get_differences(el_a, el_b, path=f"{path}[{i}]"):
                    yield diff
    elif isinstance(a, dict):
        if not isinstance(b, dict):
            yield path
        elif set(a.keys()) != set(b.keys()):
            a_min_b = set(a.keys()).difference(b.keys())
            b_min_a = set(b.keys()).difference(a.keys())
            if a_min_b:
                missing_keys_str = "`" + '`, `'.join(map(str, a_min_b)) + "`"
                yield [path,
                       f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are missing: {missing_keys_str}"]
            if b_min_a:
                extra_keys_str = "`" + '`, `'.join(map(str, b_min_a)) + "`"
                yield [path,
                       f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are extra: {extra_keys_str}"]
        elif not set(b.keys()).issubset(set(a.keys())):
            missing_keys_str = "`"+'`, `'.join(map(str, set(a.keys()).difference(b.keys())))+"`"
            yield [path,
                   f"Expected dictionary with keys `{'`, `'.join(map(str, a.keys()))}`, but the following keys are missing: {missing_keys_str}"]
        else:
            for k, v_a, v_b in zip(a.keys(), a.values(), b.values()):
                for diff in get_differences(v_a, v_b, path=f"{path}[\"{k}\"]"):
                    yield diff
    else:
        if a != b:
            yield [path, f"Objects are not equal. {a} != {b}"]

@dataclass(frozen=True)
class ObjectPattern:
    cls: type
    attrs: dict[str, Any]

    def matches(self, other: Any, raise_: bool = False) -> bool:
        """Return if object pattern matches `other` using :func:`get_differences`.

        If `raise_` is specified raises an exception with all differences found.
        """
        diffs = list(get_differences(self, other))
        if raise_ and len(diffs) != 0:
            diffs_str = '\n  '.join([f"  {self.cls.__name__}{path}: {msg}" for path, msg in diffs])
            raise ValueError(f"Object and pattern don't match:\n  {diffs_str}")
        return len(diffs)==0

    def __str__(self):
        attrs_str = ', '.join([f"{str(k)}={str(v)}" for k, v in self.attrs.items()])
        return f"{self.cls.__name__}({attrs_str})"

@dataclass(frozen=True)
class ObjectPatternConstructor:
    cls: type

    def __call__(self, **kwargs):
        return ObjectPattern(self.cls, kwargs)

@dataclass(frozen=True)
class ModuleWrapper:
    """
    Small wrapper to conveniently create `ObjectPattern`s for classes of a module.

    Example:
    >>> import foo_ir  # doctest: +SKIP
    >>> foo_ir_ = ModuleWrapper(foo_ir)  # doctest: +SKIP
    >>> assert foo_ir_.Foo(bar="baz").matches(foo_ir.Foo(bar="baz", foo="bar"))  # doctest: +SKIP
    >>> assert not foo_ir_.Foo(bar="bar").matches(foo_ir.Foo(bar="baz", foo="bar"))  # doctest: +SKIP
    """
    module: ModuleType

    def __getattr__(self, item: str):
        val = getattr(self.module, item)
        if not inspect.isclass(val):
            raise ValueError("Only classes allowed.")
        return ObjectPatternConstructor(val)