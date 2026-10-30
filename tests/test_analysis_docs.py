"""`docu/analysis-package.md` documents the API that exists: its "Reference" block of stubs against `anb.analysis`.

Each stub's parameters (names, positional or keyword-only, defaults) must be the code's, each dataclass stub's
fields (names, defaults) the class's, and every public function and class of `anb.analysis` must have a stub.
Defaults are evaluated in the namespace of the function's module, so a stub can write `base=PAPER`.
"""

import ast
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

import anb.analysis

DOC = Path(__file__).parents[1] / "docu" / "analysis-package.md"
EMPTY = inspect.Parameter.empty


def reference() -> ast.Module:
    section = DOC.read_text().split("\n## Reference\n", 1)[1]
    return ast.parse(re.search(r"```python\n(.*?)```", section, re.S).group(1))


def stubs():
    """(qualified name, stub node, the code's object) for every top-level stub and every method of a class stub."""
    for node in reference().body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            obj = getattr(anb.analysis, node.name)
            yield node.name, node, obj
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    yield f"{node.name}.{member.name}", member, getattr(obj, member.name)


def evaluate(node: ast.expr, namespace: dict):
    return eval(compile(ast.Expression(node), str(DOC), "eval"), namespace)  # our own doc, not untrusted input


def documented_parameters(stub: ast.FunctionDef, namespace: dict) -> list[tuple]:
    """(name, keyword-only, default or `EMPTY`) per parameter, without `self`."""
    args = stub.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    params = [(a.arg, False, d) for a, d in zip(positional, defaults)]
    params += [(a.arg, True, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    return [(name, kw, EMPTY if d is None else evaluate(d, namespace)) for name, kw, d in params if name != "self"]


def code_parameters(func) -> list[tuple]:
    return [(p.name, p.kind is inspect.Parameter.KEYWORD_ONLY, p.default)
            for p in inspect.signature(func).parameters.values() if p.name != "self"]


def default(f: dataclasses.Field):
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return EMPTY if f.default is dataclasses.MISSING else f.default


def test_the_reference_block_has_stubs():
    assert len(list(stubs())) > 30


@pytest.mark.parametrize("name, stub, obj", [pytest.param(*entry, id=entry[0]) for entry in stubs()])
def test_each_stub_matches_the_code(name, stub, obj):
    if isinstance(stub, ast.FunctionDef):
        namespace = getattr(obj, "__globals__", vars(anb.analysis))
        assert documented_parameters(stub, namespace) == code_parameters(obj), name
    elif dataclasses.is_dataclass(obj):
        namespace = vars(inspect.getmodule(obj))
        documented = [(f.target.id, EMPTY if f.value is None else evaluate(f.value, namespace))
                      for f in stub.body if isinstance(f, ast.AnnAssign)]
        actual = [(f.name, default(f)) for f in dataclasses.fields(obj) if not f.name.startswith("_")]
        assert documented == actual, name
    if isinstance(stub, ast.ClassDef):
        bases = tuple(evaluate(base, vars(inspect.getmodule(obj))) for base in stub.bases) or (object,)
        assert bases == obj.__bases__, name


def test_every_public_function_and_class_has_a_stub():
    documented = {node.name for node in reference().body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    public = {name for name in anb.analysis.__all__
              if inspect.isfunction(getattr(anb.analysis, name)) or inspect.isclass(getattr(anb.analysis, name))}
    assert sorted(public - documented) == []
