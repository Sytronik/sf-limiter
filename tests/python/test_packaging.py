"""Contracts for the installed package, its public stub, and documented examples."""

import ast
import importlib.metadata
import inspect
import re
import tomllib
from pathlib import Path

import sf_limiter

ROOT = Path(__file__).resolve().parents[2]


def test_package_metadata_and_typing_files() -> None:
    distribution = importlib.metadata.distribution("sf-limiter")
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = tomllib.loads(pyproject_text)["project"]
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]
    assert distribution.version == project["version"] == cargo["version"]
    assert distribution.metadata["Requires-Python"] == project["requires-python"]
    files = {str(path): path for path in distribution.files or []}
    for name in ["sf_limiter/__init__.pyi", "sf_limiter/py.typed"]:
        assert name in files
        assert Path(distribution.locate_file(files[name])).is_file()
    installed_stub = Path(distribution.locate_file(files["sf_limiter/__init__.pyi"]))
    stub_text = (ROOT / "sf_limiter.pyi").read_text(encoding="utf-8")
    assert installed_stub.read_text(encoding="utf-8") == stub_text


def stub_signature(node: ast.FunctionDef, *, bound: bool) -> inspect.Signature:
    arguments = node.args
    parameters = []
    positional = arguments.posonlyargs + arguments.args
    defaults = [inspect.Parameter.empty] * (len(positional) - len(arguments.defaults))
    defaults += [ast.literal_eval(value) for value in arguments.defaults]
    for i_argument, (argument, default) in enumerate(zip(positional, defaults)):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if i_argument < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        parameters.append(inspect.Parameter(argument.arg, kind, default=default))
    if arguments.vararg:
        parameters.append(
            inspect.Parameter(arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        default = (
            inspect.Parameter.empty if default is None else ast.literal_eval(default)
        )
        parameters.append(
            inspect.Parameter(
                argument.arg, inspect.Parameter.KEYWORD_ONLY, default=default
            )
        )
    if arguments.kwarg:
        parameters.append(
            inspect.Parameter(arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
        )
    return inspect.Signature(parameters[1:] if bound else parameters)


def test_stub_matches_runtime_api() -> None:
    tree = ast.parse((ROOT / "sf_limiter.pyi").read_text(encoding="utf-8"))
    limiter = sf_limiter.SFLimiter(48_000)
    public_names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            public_names.add(node.name)
            assert inspect.signature(getattr(sf_limiter, node.name)) == stub_signature(
                node, bound=False
            )
        elif isinstance(node, ast.ClassDef) and node.name == "SFLimiter":
            public_names.add(node.name)
            members = set()
            for member in node.body:
                if not isinstance(member, ast.FunctionDef):
                    continue
                if member.name == "__init__":
                    assert inspect.signature(sf_limiter.SFLimiter) == stub_signature(
                        member, bound=True
                    )
                elif not member.name.startswith("_"):
                    members.add(member.name)
                    if any(
                        isinstance(d, ast.Name) and d.id == "property"
                        for d in member.decorator_list
                    ):
                        assert inspect.isdatadescriptor(
                            getattr(sf_limiter.SFLimiter, member.name)
                        )
                        assert type(
                            getattr(limiter, member.name)
                        ).__name__ == ast.unparse(member.returns)
                    else:
                        assert inspect.signature(
                            getattr(limiter, member.name)
                        ) == stub_signature(member, bound=True)
            assert members == {
                name for name in dir(sf_limiter.SFLimiter) if not name.startswith("_")
            }
    assert public_names == {
        name
        for name in dir(sf_limiter)
        if not name.startswith("_") and name != "sf_limiter"
    }


def test_readme_python_examples() -> None:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    examples = re.findall(
        r"^```python\s*\n(.*?)^```", readme_text, re.MULTILINE | re.DOTALL
    )
    assert examples, "README must contain executable Python examples"
    namespace = {"__name__": "__readme__"}
    for example in examples:
        # Execute repository-owned examples just like test code.
        exec(compile(example, str(ROOT / "README.md"), "exec"), namespace)  # noqa: S102
