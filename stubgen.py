# We use 4 Spaces for indents
# Note that a ton of stuff still needs to be fixed manually

import ast
import sys
from typing import Any

INDENT: str = "    "
all_imports: set[ast.Import | ast.ImportFrom] = set()
overloaded = []


def extract_stub(tree: ast.Module) -> str:
    """Extracts type hints, imports, comments, docstrings and more into a .pyi file."""
    lines = extract_imports(tree)
    extract_body(tree.body, lines)
    return "\n".join(lines)


def extract_body(body: list[ast.stmt], lines: list[str], indent: str = "") -> None:
    for node in body:
        if isinstance(node, ast.FunctionDef):
            extract_function_stub(node, lines, indent)
        elif isinstance(node, ast.ClassDef):
            extract_class_stub(node, lines, indent)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            extract_type_alias(node, lines, indent)
        elif isinstance(node, ast.TypeAlias):
            extract_type_alias_statement(node, lines, indent)
        elif isinstance(node, ast.Delete):
            extract_del_statement(node, lines, indent)
        elif isinstance(node, ast.If):
            extract_if_block(node, lines, indent)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if not node in all_imports:
                all_imports.add(node)
                lines.append(f"{indent}{ast.unparse(node)}")
        elif isinstance(node, ast.Raise):
            lines.append(f"{indent}{ast.unparse(node)}")
        elif isinstance(node, ast.Expr):
            if ast.unparse(node) in {"exit()"}:
                lines.append(f"{indent}{ast.unparse(node)}")
        else:
            lines.append(f"{indent}...  # Unhandled node: {type(node).__name__}, {node}")


def extract_if_block(node, lines, indent=""):
    test_code = ast.unparse(node.test)
    lines.append(f"{indent}if {test_code}:")
    extract_body(node.body, lines, indent + INDENT)
    for elif_node in node.orelse:
        if isinstance(elif_node, ast.If):
            test_code = ast.unparse(elif_node.test)
            lines.append(f"{indent}elif {test_code}:")
            extract_body(elif_node.body, lines, indent + INDENT)
        else:
            lines.append(f"{indent}else:")
            extract_body(node.orelse, lines, indent + INDENT)
            break


def extract_node(node, lines, indent=""):
    """Dispatch a single AST node to the correct extractor with indentation."""
    if isinstance(node, ast.FunctionDef):
        extract_function_stub(node, lines, indent)
    elif isinstance(node, ast.ClassDef):
        extract_class_stub(node, lines)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        extract_type_alias(node, lines)
    elif isinstance(node, ast.TypeAlias):
        extract_type_alias_statement(node, lines)
    elif isinstance(node, ast.Delete):
        extract_del_statement(node, lines)
    elif isinstance(node, ast.If):
        extract_if_block(node, lines, indent)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        if not node in all_imports:
            all_imports.add(node)
            lines.append(f"{indent}{ast.unparse(node)}")
    elif isinstance(node, ast.Raise):
        lines.append(f"{indent}{ast.unparse(node)}")
    elif isinstance(node, ast.Expr):
        if ast.unparse(node) in {"exit()"}:
            lines.append(f"{indent}{ast.unparse(node)}")
    else:
        lines.append(f"{indent}...  # Unhandled node: {type(node).__name__}")


def extract_imports(tree: ast.Module, indent: str = "") -> list[str]:
    """Extracts import statements from the original script."""
    imports = ["\n"]
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(f"{indent}{ast.unparse(node)}")
        if isinstance(node, ast.ImportFrom):
            imports[0] = "# Imports"
            all_imports.add(node)
            up = ast.unparse(node)
            if len(up.split(",")) > 2:
                new_up = up.split("import ")
                new_up[0] += "import ("
                upn = new_up[1].split(", ") # User is expected to use a space between commas
                for index in range(0, len(upn)):
                    upn[index] = f"{indent if not len(indent) < len(INDENT) else ''}" + INDENT + upn[index] + ",\n"
                new_up[1] = "".join(upn)
                new_up[-1] += ")\n"
                for u in new_up:
                    imports.append(f"{indent}{u}")
            else:
                imports.append(f"{indent}{up}")
    if not len(imports) == 1:
        imports.append("\n")
    return imports


def extract_function_stub(node, lines, indent: str = ""):
    """Extracts function signatures, docstrings, and comments."""
    arg_str = format_arguments(node.args, node.name)
    return_type = get_annotation(node.returns)
    if node.name in {"__init__"}:
        return_type = "None"
    if "Never" in str(node.returns):
        return_type += "Never"
        print(return_type)
    if is_overloaded_function(node):
        overloaded.append(node.name)
        lines.append(f"{indent}@overload")
    method_type = detect_method_type(node)
    if method_type == "class":
        lines.append(f"{indent}@classmethod")
    elif method_type == "static":
        lines.append(f"{indent}@staticmethod")
    if not(node.name in overloaded and not is_overloaded_function(node)):
        lines.append(f"{indent}def {node.name}({arg_str}) -> {return_type}: ...")


def extract_class_stub(node: ast, lines, indent: str = ""):
    """Extracts class definitions, methods, docstrings, and comments."""
    base_classes = extract_base_classes(node)
    lines.append(f"{indent}class {node.name}{base_classes}:")
    method_lines = []
    has_methods = False
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            has_methods = True
            extract_function_stub(item, method_lines, indent= indent + INDENT)
    if not has_methods:
        lines.append(f"{indent}  ...")
    else:
        lines.extend(method_lines)
    lines.append("")
    lines.append("")


def extract_del_statement(node, lines, indent: str = ""):
    """Handles `del` statements by marking names as deleted."""
    for target in node.targets:
        if isinstance(target, ast.Name):
            lines.append(f"{indent}del {target.id}")


def extract_type_alias(node, lines, indent: str = ""):
    """Handles type alias definitions and annotated variables."""
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            name = node.target.id
            annotation = get_annotation(node.annotation)
            lines.append(f"{indent}{name}: {annotation} = ...")
    elif isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
            inferred_type = get_annotation(node.value)
            lines.append(f"{indent}{name}: {inferred_type} = ...")


def extract_type_alias_statement(node, lines, indent: str = ""):
    """Handles `type NAME = OTHER_TYPE` statements correctly."""
    if isinstance(node, ast.TypeAlias):
        alias_name = node.name.id
        alias_type = get_annotation(node.value)
        lines.append(f"{indent}type {alias_name} = {alias_type}")
        # Returns nothing?


def extract_base_classes(node) -> str:
    """Extracts base classes of a class definition."""
    if not node.bases:
        return ""
    base_names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(ast.unparse(base))
        elif isinstance(base, ast.Subscript):
            base_names.append(ast.unparse(base))
    return f"({', '.join(base_names)})"


def is_overloaded_function(node) -> bool:
    """Checks if a function has multiple definitions (overloaded)."""
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "overload"
        for decorator in node.decorator_list
    )


def detect_method_type(node) -> str:
    """Detects if a function is an instance, class, or static method."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            if decorator.id == "classmethod":
                return "class"
            elif decorator.id == "staticmethod":
                return "static"
        return str(decorator)
    return "instance"


def format_arguments(args, function_name) -> str:
    """Formats function arguments with type hints, special markers, and fixes `self` handling."""
    params = []
    pos_only = [format_arg(arg) for arg in args.posonlyargs]
    if pos_only:
        params.extend(pos_only)
        params.append("/")
    params.extend(format_arg(arg) for arg in args.args)
    if args.vararg:
        params.append(f"*{args.vararg.arg}: {get_annotation(args)}")
    if args.kwarg:
        params.append(f"**{args.kwarg.arg}: {get_annotation(args)}")
    if args.kwonlyargs and not args.vararg and not pos_only:
        params.append("*")
        params.extend(format_arg(arg) for arg in args.kwonlyargs)
    if function_name in {"__call__", "__init__"} and params and params[0].startswith("self"):
        params[0] = "self"
    return ", ".join(params)


def format_arg(arg) -> str:
    """Formats a single function argument with its type annotation."""
    annotation = get_annotation(arg.annotation)
    if annotation is Any:
        annotation = "TODO"
    return f"{arg.arg}: {annotation}"


def get_annotation(node) -> str:
    """Extracts annotation from AST node or defaults to Any."""
    if node is None:
        return "Any"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        base = get_annotation(node.value)
        if base == "Literal":
            # Keep literal values intact (like Literal[True])
            if isinstance(node.slice, ast.Tuple):
                args = ", ".join(get_annotation(element) for element in node.slice.elts)
                return f"{base}[{args}]"
            else:
                return f"{base}[{get_annotation(node.slice)}]"
        else:
            return f"{base}[{get_annotation(node.slice)}]"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return "Any"


def generate_pyi(file_path: str):
    """Generates a .pyi file for the given Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    stub_content = extract_stub(tree)
    with open(file_path.replace(".py", ".pyi"), "w", encoding="utf-8", errors="strict") as f:
        f.write(stub_content)
    print(f"Stub file generated: {file_path.replace(".py", ".pyi")}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SyntaxError("Usage: python pyi-gen.py <python_file.py>")
    else:
        generate_pyi(sys.argv[1])
#TODO:
# default args
# types of args dont seem to work, always Any
# normal comments dont work