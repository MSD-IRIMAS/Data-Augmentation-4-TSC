"""Unit tests for modules import."""

import ast
import os
import pkgutil

import pytest


@pytest.mark.parametrize("folder_path", ["src/data_augs/"])
def test_all_functions_imported_inside_init(folder_path):
    """Test if all functions imported in init."""
    init_path = os.path.join(folder_path, "__init__.py")

    if not os.path.exists(init_path):
        pytest.fail(f"No __init__.py found in {folder_path}")

    def get_functions_from_module(module_path):
        """Parse a module and return a list of functions names defined in it."""
        with open(module_path) as file:
            node = ast.parse(file.read(), filename=module_path)

        functions = [
            n.name
            for n in node.body
            if isinstance(n, ast.FunctionDef)
            if n.name != "_get_random_cubic" and n.name != "_augment_one_series"
        ]
        return functions

    def get_imported_functions_from_init(init_path):
        """Parse the __init__.py file and return a list of imported function names."""
        with open(init_path) as file:
            node = ast.parse(file.read(), filename=init_path)
        imported_functions = []
        for n in node.body:
            if isinstance(n, ast.ImportFrom) and n.module and n.module != "__future__":
                for alias in n.names:
                    imported_functions.append(alias.name.split(".")[0])
        return imported_functions

    imported_functions = get_imported_functions_from_init(init_path)
    all_functions = []

    for _, module_name, is_pkg in pkgutil.iter_modules([folder_path]):
        if not is_pkg:
            module_path = os.path.join(folder_path, f"{module_name}.py")
            all_functions.extend(get_functions_from_module(module_path))

    missing_functions = set(all_functions) - set(imported_functions)
    assert (
        not missing_functions
    ), f"Missing Functions in __init__.py: {missing_functions}"
