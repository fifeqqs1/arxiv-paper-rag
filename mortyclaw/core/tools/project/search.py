from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

from ..base import mortyclaw_tool
from ... import code_index
from .common import (
    EXCLUDED_DIRS,
    TEXT_EXTENSIONS,
    _iter_project_files,
    _relative_path,
    _resolve_project_path,
    _resolve_project_root,
    _safe_read_text,
    _truncate,
)


def _parse_python_file(path_value: str) -> ast.AST | None:
    try:
        return ast.parse(_safe_read_text(path_value), filename=path_value)
    except Exception:
        return None


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _node_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _node_name(node.func)
    if isinstance(node, ast.Subscript):
        return _node_name(node.value)
    return ""


def _call_matches(call_name: str, query: str) -> bool:
    normalized_query = query.strip()
    if not normalized_query:
        return False
    return call_name == normalized_query or call_name.endswith(f".{normalized_query}") or call_name.split(".")[-1] == normalized_query


def _python_module_name(project_root: str, path_value: str) -> str:
    rel = _relative_path(project_root, path_value)
    if not rel.endswith(".py"):
        return ""
    module = rel[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def _normalize_import_module(module: str) -> str:
    return (module or "").lstrip(".")


def _module_suffix(imported_module: str, target_module: str) -> str | None:
    if not imported_module or not target_module:
        return None
    if imported_module == target_module:
        return ""
    prefix = f"{imported_module}."
    if target_module.startswith(prefix):
        return target_module[len(prefix):]
    return None


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack: list[str] = []
        self.symbols: list[dict] = []

    def _record(self, node: ast.AST, kind: str, name: str) -> None:
        qualified = ".".join([*self.stack, name]) if self.stack else name
        self.symbols.append({
            "kind": kind,
            "name": name,
            "qualified_name": qualified,
            "line": getattr(node, "lineno", 0),
            "end_line": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
        })

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, "class", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node, "function", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node, "async_function", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, query: str, allowed_names: set[str] | None = None):
        self.query = query
        self.allowed_names = allowed_names
        self.stack: list[str] = []
        self.calls: list[dict] = []

    def _enter_scope(self, name: str):
        self.stack.append(name)

    def _exit_scope(self):
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter_scope(node.name)
        self.generic_visit(node)
        self._exit_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_scope(node.name)
        self.generic_visit(node)
        self._exit_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_scope(node.name)
        self.generic_visit(node)
        self._exit_scope()

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _node_name(node.func)
        matches = call_name in self.allowed_names if self.allowed_names is not None else _call_matches(call_name, self.query)
        if matches:
            self.calls.append({
                "call": call_name,
                "line": getattr(node, "lineno", 0),
                "scope": ".".join(self.stack) if self.stack else "<module>",
            })
        self.generic_visit(node)


class _ImportBindingVisitor(ast.NodeVisitor):
    def __init__(self):
        self.bindings: list[dict] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bindings.append({
                "module": alias.name,
                "name": None,
                "alias": alias.asname,
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.bindings.append({
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
            })
        self.generic_visit(node)


def _collect_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append(module)
    return sorted(set(item for item in imports if item))


def _collect_import_bindings(tree: ast.AST) -> list[dict]:
    visitor = _ImportBindingVisitor()
    visitor.visit(tree)
    return visitor.bindings


def _format_symbol_results(results: list[dict]) -> str:
    if not results:
        return "未找到匹配的 Python 函数或类。"
    lines = ["符号索引结果："]
    for item in results:
        lines.append(
            f"- {item['file']}:{item['line']} "
            f"[{item['kind']}] {item['qualified_name']}"
        )
    return "\n".join(lines)


def _search_text(project_root: str, query: str, file_glob: str, max_results: int) -> str:
    if not query.strip():
        return "search_project_code 参数错误：text 模式下 query 不能为空。"

    rg_path = shutil.which("rg")
    if rg_path:
        command = [
            rg_path,
            "--line-number",
            "--column",
            "--no-heading",
            "--color",
            "never",
            "--smart-case",
            "--hidden",
        ]
        for excluded in sorted(EXCLUDED_DIRS):
            command.extend(["--glob", f"!{excluded}/**"])
        if file_glob:
            command.extend(["--glob", file_glob])
        command.extend(["--", query, project_root])
        result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        if result.returncode not in {0, 1}:
            return f"rg 搜索失败：{result.stderr.strip() or result.stdout.strip()}"
        lines = []
        for raw_line in (result.stdout or "").splitlines()[:max_results]:
            parts = raw_line.split(":", 3)
            if len(parts) == 4:
                path_part, line_no, column, text = parts
                rel = _relative_path(project_root, path_part)
                lines.append(f"{rel}:{line_no}:{column}: {text}")
            else:
                lines.append(raw_line)
        return "全文搜索结果：\n" + ("\n".join(lines) if lines else "未找到匹配内容。")

    lowered_query = query.lower()
    matches: list[str] = []
    for path in _iter_project_files(project_root, extensions=TEXT_EXTENSIONS):
        if file_glob and not Path(_relative_path(project_root, path)).match(file_glob):
            continue
        try:
            for line_number, line in enumerate(_safe_read_text(path).splitlines(), start=1):
                if lowered_query in line.lower():
                    matches.append(f"{_relative_path(project_root, path)}:{line_number}: {line.strip()}")
                    if len(matches) >= max_results:
                        break
        except Exception:
            continue
        if len(matches) >= max_results:
            break
    return "全文搜索结果：\n" + ("\n".join(matches) if matches else "未找到匹配内容。")


def _search_symbols(project_root: str, query: str, max_results: int) -> str:
    normalized_query = query.strip().lower()
    results: list[dict] = []
    for path in _iter_project_files(project_root, extensions={".py"}):
        tree = _parse_python_file(path)
        if tree is None:
            continue
        visitor = _SymbolVisitor()
        visitor.visit(tree)
        for symbol in visitor.symbols:
            haystack = f"{symbol['name']} {symbol['qualified_name']}".lower()
            if not normalized_query or normalized_query in haystack:
                results.append({
                    **symbol,
                    "file": _relative_path(project_root, path),
                })
                if len(results) >= max_results:
                    return _format_symbol_results(results)
    return _format_symbol_results(results)


def _resolve_symbol_definitions(project_root: str, query: str) -> list[dict]:
    normalized_query = query.strip().lower()
    if not normalized_query:
        return []
    results: list[dict] = []
    for path in _iter_project_files(project_root, extensions={".py"}):
        tree = _parse_python_file(path)
        if tree is None:
            continue
        visitor = _SymbolVisitor()
        visitor.visit(tree)
        module_name = _python_module_name(project_root, path)
        rel = _relative_path(project_root, path)
        for symbol in visitor.symbols:
            qualified = symbol["qualified_name"].lower()
            if symbol["name"].lower() != normalized_query and qualified != normalized_query and not qualified.endswith(f".{normalized_query}"):
                continue
            results.append({
                **symbol,
                "file": rel,
                "module": module_name,
            })
    return results


def _allowed_call_names_for_definition(imports: list[dict], target_module: str, target_name: str) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for item in imports:
        imported_module = _normalize_import_module(item.get("module", ""))
        imported_name = item.get("name")
        alias = item.get("alias")

        suffix = _module_suffix(imported_module, target_module)
        if suffix is not None:
            base = alias or imported_module
            dotted = f"{base}.{suffix}" if suffix else base
            if dotted:
                allowed[f"{dotted}.{target_name}"] = f"import {item['module']}"

        if not imported_name:
            continue

        binding = alias or imported_name
        if imported_module == target_module and imported_name == target_name:
            allowed[binding] = f"from {item['module']} import {imported_name}"
            continue

        imported_symbol_module = ".".join(part for part in [imported_module, imported_name] if part)
        nested_suffix = _module_suffix(imported_symbol_module, target_module)
        if nested_suffix is not None:
            dotted = f"{binding}.{nested_suffix}" if nested_suffix else binding
            allowed[f"{dotted}.{target_name}"] = f"from {item['module']} import {imported_name}"
    return allowed


def _search_callers_naive(project_root: str, query: str, max_results: int) -> list[str]:
    results: list[str] = []
    for path in _iter_project_files(project_root, extensions={".py"}):
        tree = _parse_python_file(path)
        if tree is None:
            continue
        visitor = _CallVisitor(query.strip())
        visitor.visit(tree)
        rel = _relative_path(project_root, path)
        for call in visitor.calls:
            results.append(f"{rel}:{call['line']} scope={call['scope']} call={call['call']}")
            if len(results) >= max_results:
                return results
    return results


def _search_callers(project_root: str, query: str, max_results: int) -> str:
    if not query.strip():
        return "callers 模式下 query 必须是函数名或方法名。"
    definitions = _resolve_symbol_definitions(project_root, query)
    if not definitions:
        naive_results = _search_callers_naive(project_root, query, max_results)
        return "调用点搜索结果：\n" + ("\n".join(naive_results) if naive_results else "未找到调用点。")

    lines = ["调用点搜索结果："]
    total_results = 0
    seen_calls: set[tuple[str, int, str, str]] = set()

    for definition in definitions:
        if total_results >= max_results:
            break
        candidate_lines: list[str] = []
        for path in _iter_project_files(project_root, extensions={".py"}):
            tree = _parse_python_file(path)
            if tree is None:
                continue
            rel = _relative_path(project_root, path)
            allowed_names = {query.strip(): "local_definition"} if rel == definition["file"] else _allowed_call_names_for_definition(
                _collect_import_bindings(tree),
                definition["module"],
                definition["name"],
            )
            if not allowed_names:
                continue
            visitor = _CallVisitor(query.strip(), set(allowed_names))
            visitor.visit(tree)
            for call in visitor.calls:
                if call["call"] not in allowed_names:
                    continue
                key = (rel, call["line"], call["scope"], call["call"])
                if key in seen_calls:
                    continue
                seen_calls.add(key)
                via = allowed_names[call["call"]]
                candidate_lines.append(
                    f"- {rel}:{call['line']} scope={call['scope']} call={call['call']} via={via}"
                )
                total_results += 1
                if total_results >= max_results:
                    break
            if total_results >= max_results:
                break
        if candidate_lines:
            lines.append(
                f"定义候选：{definition['file']}:{definition['line']} "
                f"[{definition['kind']}] {definition['qualified_name']}"
            )
            lines.extend(candidate_lines)

    if len(lines) == 1:
        naive_results = _search_callers_naive(project_root, query, max_results)
        lines.extend(naive_results or ["未找到调用点。"])
    return "\n".join(lines)


def _find_project_file(project_root: str, query: str) -> str:
    candidate = query.strip()
    if not candidate:
        raise ValueError("该模式需要 query 指定模块路径或文件名。")
    try:
        return _resolve_project_path(project_root, candidate)
    except Exception:
        pass

    matches = []
    lowered = candidate.lower()
    for path in _iter_project_files(project_root, extensions=TEXT_EXTENSIONS):
        rel = _relative_path(project_root, path)
        if lowered in rel.lower() or lowered == os.path.basename(rel).lower():
            matches.append(path)
            if len(matches) >= 5:
                break
    if not matches:
        raise FileNotFoundError(f"未找到匹配模块或文件：{query}")
    return matches[0]


def _dependency_summary(project_root: str, query: str, max_results: int) -> str:
    target = _find_project_file(project_root, query)
    if not target.endswith(".py"):
        return "依赖分析当前仅支持 Python 文件。"
    tree = _parse_python_file(target)
    if tree is None:
        return "目标文件无法解析为 Python AST。"

    target_module = _python_module_name(project_root, target)
    imports = _collect_imports(tree)
    reverse_dependents: list[str] = []

    for path in _iter_project_files(project_root, extensions={".py"}):
        if path == target:
            continue
        other_tree = _parse_python_file(path)
        if other_tree is None:
            continue
        other_imports = _collect_imports(other_tree)
        for item in other_imports:
            normalized = item.lstrip(".")
            if target_module and (normalized == target_module or normalized.startswith(f"{target_module}.")):
                reverse_dependents.append(_relative_path(project_root, path))
                break

    lines = [
        f"模块依赖分析：{_relative_path(project_root, target)}",
        "",
        "直接 imports：",
    ]
    lines.extend(f"- {item}" for item in imports[:max_results])
    if not imports:
        lines.append("- 无")
    lines.append("")
    lines.append("项目内反向依赖：")
    lines.extend(f"- {item}" for item in reverse_dependents[:max_results])
    if not reverse_dependents:
        lines.append("- 未发现其他 Python 文件直接 import 该模块")
    return "\n".join(lines)


def _entrypoint_summary(project_root: str, max_results: int) -> str:
    candidates: list[tuple[int, str, list[str]]] = []
    for path in _iter_project_files(project_root, extensions={".py"}):
        tree = _parse_python_file(path)
        if tree is None:
            continue
        rel = _relative_path(project_root, path)
        basename = os.path.basename(path).lower()
        reasons: list[str] = []
        score = 0
        if basename in {"main.py", "train.py", "run.py", "finetune.py"} or basename.startswith(("train_", "run_")):
            score += 3
            reasons.append("入口型文件名")
        imports = _collect_imports(tree)
        if any(item in imports for item in {"argparse", "click", "typer", "hydra"}):
            score += 2
            reasons.append("包含 CLI/配置框架 import")
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test):
                score += 4
                reasons.append("包含 if __name__ == '__main__'")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"main", "train", "fit", "run", "cli"}:
                score += 1
                reasons.append(f"定义 {node.name}()")
            if isinstance(node, ast.Call):
                call_name = _node_name(node.func)
                if call_name.split(".")[-1] in {"fit", "train", "Trainer", "main"}:
                    score += 1
                    reasons.append(f"调用 {call_name}")
        if score:
            candidates.append((score, rel, sorted(set(reasons))))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        return "未发现明显训练/运行入口。可尝试 text 模式搜索 train、main、argparse、Trainer。"

    lines = ["可能的训练/运行入口："]
    for score, rel, reasons in candidates[:max_results]:
        lines.append(f"- {rel} score={score} | {'; '.join(reasons)}")
    return "\n".join(lines)


def _data_flow_summary(project_root: str, query: str, max_results: int) -> str:
    target = _find_project_file(project_root, query)
    if not target.endswith(".py"):
        return "数据流分析当前仅支持 Python 文件。"
    tree = _parse_python_file(target)
    if tree is None:
        return "目标文件无法解析为 Python AST。"

    visitor = _SymbolVisitor()
    visitor.visit(tree)
    imports = _collect_imports(tree)
    io_calls: list[str] = []
    ml_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _node_name(node.func)
            if call_name:
                if call_name.split(".")[-1] in {"open", "read_csv", "read_json", "load", "dump", "save", "to_csv", "to_json"}:
                    io_calls.append(f"{getattr(node, 'lineno', 0)}:{call_name}")
                if call_name.split(".")[-1].lower() in {"dataloader", "dataset", "trainer", "fit", "train", "eval", "evaluate", "forward"}:
                    ml_calls.append(f"{getattr(node, 'lineno', 0)}:{call_name}")

    lines = [
        f"模块数据流/结构摘要：{_relative_path(project_root, target)}",
        "",
        "1. 依赖输入：",
        *(f"- import {item}" for item in imports[:max_results]),
    ]
    if not imports:
        lines.append("- 无显式 import")

    lines.extend(["", "2. 主要结构："])
    if visitor.symbols:
        for symbol in visitor.symbols[:max_results]:
            lines.append(f"- {symbol['kind']} {symbol['qualified_name']} @ line {symbol['line']}")
    else:
        lines.append("- 未发现函数或类定义")

    lines.extend(["", "3. I/O 或数据读写线索："])
    lines.extend(f"- line {item}" for item in sorted(set(io_calls))[:max_results])
    if not io_calls:
        lines.append("- 未发现明显文件/数据读写调用")

    lines.extend(["", "4. 训练/推理相关线索："])
    lines.extend(f"- line {item}" for item in sorted(set(ml_calls))[:max_results])
    if not ml_calls:
        lines.append("- 未发现明显训练/推理调用")

    lines.extend(["", "5. 解释建议："])
    lines.append("- 结合 symbol/callers/dependencies 模式继续追踪关键函数的上游调用和下游依赖。")
    return "\n".join(lines)


@mortyclaw_tool
def search_project_code(
    query: str,
    project_root: str = "",
    mode: str = "text",
    file_glob: str = "",
    max_results: int = 50,
    use_index: bool = True,
) -> str:
    """
    在科研/代码项目中检索和理解代码结构。

    mode 可选：
    - text：基于 rg 的全文搜索，适合找关键词、报错、配置项。
    - symbol：基于 Python AST 搜索函数/类定义。
    - callers：基于 Python AST 搜索某个函数/方法在哪里被调用。
    - dependencies：分析某个 Python 模块的 import 和项目内反向依赖。
    - data_flow：汇总某个 Python 模块的 imports、函数/类、I/O 和训练/推理线索。
    - entrypoints：查找可能的训练/运行入口。

    默认会优先使用 workspace/code_index.sqlite3 做轻量增量索引查询；
    索引不可用或 use_index=false 时会回退到临时 AST 扫描。
    """
    try:
        root = _resolve_project_root(project_root)
        normalized_mode = (mode or "text").strip().lower()
        limit = max(1, min(int(max_results or 50), 200))
        if normalized_mode == "text":
            return _truncate(_search_text(root, query, file_glob, limit))
        if normalized_mode == "symbol":
            if use_index:
                try:
                    return _truncate(code_index.search_symbols(root, query, limit))
                except Exception:
                    pass
            return _truncate(_search_symbols(root, query, limit))
        if normalized_mode == "callers":
            if use_index:
                try:
                    return _truncate(code_index.search_callers(root, query, limit))
                except Exception:
                    pass
            return _truncate(_search_callers(root, query, limit))
        if normalized_mode == "dependencies":
            if use_index:
                try:
                    return _truncate(code_index.dependency_summary(root, query, limit))
                except Exception:
                    pass
            return _truncate(_dependency_summary(root, query, limit))
        if normalized_mode == "data_flow":
            if use_index:
                try:
                    return _truncate(code_index.data_flow_summary(root, query, limit))
                except Exception:
                    pass
            return _truncate(_data_flow_summary(root, query, limit))
        if normalized_mode == "entrypoints":
            if use_index:
                try:
                    return _truncate(code_index.entrypoint_summary(root, limit))
                except Exception:
                    pass
            return _truncate(_entrypoint_summary(root, limit))
        return "search_project_code 参数错误：mode 只能是 text/symbol/callers/dependencies/data_flow/entrypoints。"
    except Exception as exc:
        return f"search_project_code 失败：{exc}"
