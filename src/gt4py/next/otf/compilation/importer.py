# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import pathlib
from types import ModuleType


#: Keeps references to dynamically loaded modules alive for the lifetime of the
#: process. Required for nanobind extension modules, which must outlive the
#: functions imported from them (see PR #2431). Keyed by the resolved file path
#: so that a given file is only loaded once.
_loaded_modules: dict[str, ModuleType] = {}


def import_from_path(module_file: pathlib.Path, *, keep_reference: bool = False) -> ModuleType:
    """
    Load dynamically a python module from a given file path.

    Args:
        module_file: The path to the python file to load as a module.
        keep_reference: If `True`, keep a reference to the loaded module alive
            for the lifetime of the process (required for nanobind extension
            modules, which must outlive their functions, see PR #2431) and
            return the already loaded module on subsequent calls for the same
            file.

    Returns:
        The loaded module.
    """
    key = str(module_file)
    if keep_reference and (cached := _loaded_modules.get(key)) is not None:
        return cached

    module_name = module_file.name.split(".")[0]
    error_msg = f"Could not load module named {module_name} from {module_file}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if not spec or not spec.loader:
        raise ModuleNotFoundError(error_msg)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as ex:
        raise ModuleNotFoundError(error_msg) from ex

    if keep_reference:
        _loaded_modules[key] = module

    return module
