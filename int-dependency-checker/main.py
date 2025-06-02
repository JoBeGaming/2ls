import ast
from pathlib import Path
from collections import defaultdict, deque

class ClassFunctionDependencyAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.current_class = None
        self.current_function = None
        self.dependencies = defaultdict(set)
        self.defined_functions = set()

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class:
            func_name = f"{self.current_class}.{node.name}"
        else:
            func_name = node.name

        self.defined_functions.add(func_name)
        prev_function = self.current_function
        self.current_function = func_name

        self.generic_visit(node)

        self.current_function = prev_function

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr
        else:
            called = None

        if self.current_function and called:
            self.dependencies[self.current_function].add(called)

        self.generic_visit(node)

def qualify_dependencies(deps, defined_names):
    qualified = defaultdict(set)
    flat_lookup = {name.split(".")[-1]: name for name in defined_names}

    for caller, callees in deps.items():
        for callee in callees:
            if callee in flat_lookup:
                qualified[caller].add(flat_lookup[callee])
    return qualified

def topological_sort(deps):
    indegree = defaultdict(int)
    graph = defaultdict(set)

    # Build the graph
    for node, neighbors in deps.items():
        for n in neighbors:
            graph[node].add(n)
            indegree[n] += 1
        if node not in indegree:
            indegree[node] = 0

    # Topological sort (Kahn's algorithm)
    queue = deque([n for n in indegree if indegree[n] == 0])
    sorted_list = []

    while queue:
        node = queue.popleft()
        sorted_list.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_list

def save_dot_grouped(sorted_nodes, deps, path):
    by_class = defaultdict(list)
    for name in sorted_nodes:
        if '.' in name:
            cls, meth = name.split('.', 1)
            by_class[cls].append(name)
        else:
            by_class['__global__'].append(name)

    lines = ["digraph G {"]

    # Optional: use subgraphs for class groupings
    for cls, funcs in by_class.items():
        if cls != '__global__':
            lines.append(f'    subgraph cluster_{cls} {{')
            lines.append(f'        label="{cls}";')
        for func in funcs:
            lines.append(f'        "{func}";')
        if cls != '__global__':
            lines.append("    }")

    # Add edges
    for src, targets in deps.items():
        for tgt in targets:
            lines.append(f'    "{src}" -> "{tgt}";')

    lines.append("}")
    with open(path, "w") as f:
        f.writelines(lines)

# ---- USAGE ----

def analyze_python_file(file_path: str):
    source = Path(__file__.replace("main.py", file_path)).read_text()
    tree = ast.parse(source)

    analyzer = ClassFunctionDependencyAnalyzer()
    analyzer.visit(tree)

    qualified = qualify_dependencies(analyzer.dependencies, analyzer.defined_functions)
    sorted_nodes = topological_sort(qualified)
    save_dot_grouped(sorted_nodes, qualified, file_path.replace(".py", ".dot"))


if __name__ == "__main__":
    import sys
    try:
        analyze_python_file(sys.argv[1])
    except IndexError:
        analyze_python_file("check.py")