from __future__ import annotations

import ast
import json
import os
import sqlite3
import time
from pathlib import Path

from ..config import WORKSPACE_DIR


CODE_INDEX_DB_PATH = os.path.join(WORKSPACE_DIR, "code_index.sqlite3")

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "logs",
    "node_modules",
    "rick",
    "workspace",
}

SENSITIVE_FILENAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SENSITIVE_EXTENSIONS = {".key", ".pem", ".p12", ".pfx"}
MAX_INDEX_BYTES = 2 * 1024 * 1024
MAX_INDEX_FILES = 3000

IO_CALL_NAMES = {"open", "read_csv", "read_json", "load", "dump", "save", "to_csv", "to_json"}
ML_CALL_NAMES = {"dataloader", "dataset", "trainer", "fit", "train", "eval", "evaluate", "forward"}
ENTRY_FUNCTION_NAMES = {"main", "train", "fit", "run", "cli"}
ENTRY_CALL_NAMES = {"fit", "train", "Trainer", "main"}
CLI_IMPORTS = {"argparse", "click", "typer", "hydra"}


def _connect(db_path: str = CODE_INDEX_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            project_root TEXT NOT NULL,
            file_path TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            indexed_at REAL NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            entrypoint_score INTEGER NOT NULL DEFAULT 0,
            entrypoint_reasons_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (project_root, file_path)
        );

        CREATE TABLE IF NOT EXISTS symbols (
            project_root TEXT NOT NULL,
            file_path TEXT NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            line INTEGER NOT NULL,
            end_line INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS calls (
            project_root TEXT NOT NULL,
            file_path TEXT NOT NULL,
            caller_scope TEXT NOT NULL,
            callee_name TEXT NOT NULL,
            callee_leaf TEXT NOT NULL,
            line INTEGER NOT NULL,
            is_top_level INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS imports (
            project_root TEXT NOT NULL,
            file_path TEXT NOT NULL,
            module TEXT NOT NULL,
            name TEXT,
            alias TEXT,
            level INTEGER NOT NULL DEFAULT 0,
            line INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_symbols_lookup
            ON symbols(project_root, name, qualified_name);
        CREATE INDEX IF NOT EXISTS idx_calls_lookup
            ON calls(project_root, callee_leaf, callee_name);
        CREATE INDEX IF NOT EXISTS idx_imports_lookup
            ON imports(project_root, module);
        CREATE INDEX IF NOT EXISTS idx_files_project
            ON files(project_root, file_path);
        """
    )
    conn.commit()


def _normalize_root(project_root: str) -> str:
    return os.path.realpath(os.path.expanduser(project_root))


def _relative_path(project_root: str, path_value: str) -> str:
    return os.path.relpath(path_value, project_root).replace(os.sep, "/")


def _is_sensitive_path(path_value: str) -> bool:
    basename = os.path.basename(path_value).lower()
    if basename in SENSITIVE_FILENAMES:
        return True
    _, ext = os.path.splitext(basename)
    return ext.lower() in SENSITIVE_EXTENSIONS


def _iter_python_files(project_root: str, max_files: int = MAX_INDEX_FILES):
    count = 0
    for current_root, dirs, files in os.walk(project_root):
        dirs[:] = [item for item in dirs if item not in EXCLUDED_DIRS and not item.startswith(".cache")]
        for filename in files:
            if Path(filename).suffix.lower() != ".py":
                continue
            path = os.path.join(current_root, filename)
            if _is_sensitive_path(path):
                continue
            count += 1
            if count > max_files:
                return
            yield path


def _read_indexable_text(path_value: str) -> str:
    if os.path.getsize(path_value) > MAX_INDEX_BYTES:
        raise ValueError("file exceeds code index size limit")
    with open(path_value, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


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


def _module_name_from_file(project_root: str, file_path: str) -> str:
    rel = file_path.replace(os.sep, "/")
    if os.path.isabs(file_path):
        rel = _relative_path(project_root, file_path)
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


def _allowed_call_names_for_definition(import_rows: list[sqlite3.Row], target_module: str, target_name: str) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for row in import_rows:
        imported_module = _normalize_import_module(row["module"])
        imported_name = row["name"]
        alias = row["alias"]

        suffix = _module_suffix(imported_module, target_module)
        if suffix is not None:
            base = alias or imported_module
            dotted = f"{base}.{suffix}" if suffix else base
            if dotted:
                allowed[f"{dotted}.{target_name}"] = f"import {row['module']}"

        if not imported_name:
            continue

        binding = alias or imported_name
        if imported_module == target_module and imported_name == target_name:
            allowed[binding] = f"from {row['module']} import {imported_name}"
            continue

        imported_symbol_module = ".".join(part for part in [imported_module, imported_name] if part)
        nested_suffix = _module_suffix(imported_symbol_module, target_module)
        if nested_suffix is not None:
            dotted = f"{binding}.{nested_suffix}" if nested_suffix else binding
            allowed[f"{dotted}.{target_name}"] = f"from {row['module']} import {imported_name}"
    return allowed


class _IndexVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack: list[str] = []
        self.symbols: list[dict] = []
        self.calls: list[dict] = []
        self.imports: list[dict] = []
        self.has_main_guard = False

    def _scope(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def _record_symbol(self, node: ast.AST, kind: str, name: str) -> None:
        qualified = ".".join([*self.stack, name]) if self.stack else name
        self.symbols.append(
            {
                "kind": kind,
                "name": name,
                "qualified_name": qualified,
                "line": getattr(node, "lineno", 0),
                "end_line": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_symbol(node, "class", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_symbol(node, "function", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_symbol(node, "async_function", node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee_name = _node_name(node.func)
        if callee_name:
            self.calls.append(
                {
                    "caller_scope": self._scope(),
                    "callee_name": callee_name,
                    "callee_leaf": callee_name.split(".")[-1],
                    "line": getattr(node, "lineno", 0),
                    "is_top_level": 1 if not self.stack else 0,
                }
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                {
                    "module": alias.name,
                    "name": None,
                    "alias": alias.asname,
                    "level": 0,
                    "line": getattr(node, "lineno", 0),
                }
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.imports.append(
                {
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "level": node.level,
                    "line": getattr(node, "lineno", 0),
                }
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        try:
            test_text = ast.unparse(node.test)
        except Exception:
            test_text = ""
        if "__name__" in test_text and "__main__" in test_text:
            self.has_main_guard = True
        self.generic_visit(node)


def _entrypoint_score(file_path: str, visitor: _IndexVisitor) -> tuple[int, list[str]]:
    basename = os.path.basename(file_path).lower()
    score = 0
    reasons: list[str] = []

    if basename in {"main.py", "train.py", "run.py", "finetune.py"} or basename.startswith(("train_", "run_")):
        score += 3
        reasons.append("入口型文件名")

    imported_modules = {item["module"].lstrip(".") for item in visitor.imports}
    if imported_modules.intersection(CLI_IMPORTS):
        score += 2
        reasons.append("包含 CLI/配置框架 import")

    if visitor.has_main_guard:
        score += 4
        reasons.append("包含 if __name__ == '__main__'")

    for symbol in visitor.symbols:
        if symbol["name"] in ENTRY_FUNCTION_NAMES:
            score += 1
            reasons.append(f"定义 {symbol['name']}()")

    for call in visitor.calls:
        if call["callee_leaf"] in ENTRY_CALL_NAMES:
            score += 1
            reasons.append(f"调用 {call['callee_name']}")

    return score, sorted(set(reasons))


def _replace_file_index(
    conn: sqlite3.Connection,
    project_root: str,
    file_path: str,
    absolute_path: str,
    stat_result: os.stat_result,
    visitor: _IndexVisitor | None,
    *,
    status: str,
    error: str = "",
) -> None:
    conn.execute("DELETE FROM symbols WHERE project_root = ? AND file_path = ?", (project_root, file_path))
    conn.execute("DELETE FROM calls WHERE project_root = ? AND file_path = ?", (project_root, file_path))
    conn.execute("DELETE FROM imports WHERE project_root = ? AND file_path = ?", (project_root, file_path))

    score = 0
    reasons: list[str] = []
    if visitor is not None and status == "indexed":
        score, reasons = _entrypoint_score(absolute_path, visitor)

    conn.execute(
        """
        INSERT INTO files (
            project_root, file_path, mtime_ns, size, indexed_at, status, error,
            entrypoint_score, entrypoint_reasons_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_root, file_path) DO UPDATE SET
            mtime_ns = excluded.mtime_ns,
            size = excluded.size,
            indexed_at = excluded.indexed_at,
            status = excluded.status,
            error = excluded.error,
            entrypoint_score = excluded.entrypoint_score,
            entrypoint_reasons_json = excluded.entrypoint_reasons_json
        """,
        (
            project_root,
            file_path,
            int(stat_result.st_mtime_ns),
            int(stat_result.st_size),
            time.time(),
            status,
            error,
            score,
            json.dumps(reasons, ensure_ascii=False),
        ),
    )

    if visitor is None or status != "indexed":
        return

    conn.executemany(
        """
        INSERT INTO symbols (
            project_root, file_path, kind, name, qualified_name, line, end_line
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                project_root,
                file_path,
                symbol["kind"],
                symbol["name"],
                symbol["qualified_name"],
                int(symbol["line"]),
                int(symbol["end_line"]),
            )
            for symbol in visitor.symbols
        ],
    )
    conn.executemany(
        """
        INSERT INTO calls (
            project_root, file_path, caller_scope, callee_name, callee_leaf, line, is_top_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                project_root,
                file_path,
                call["caller_scope"],
                call["callee_name"],
                call["callee_leaf"],
                int(call["line"]),
                int(call["is_top_level"]),
            )
            for call in visitor.calls
        ],
    )
    conn.executemany(
        """
        INSERT INTO imports (
            project_root, file_path, module, name, alias, level, line
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                project_root,
                file_path,
                item["module"],
                item["name"],
                item["alias"],
                int(item["level"]),
                int(item["line"]),
            )
            for item in visitor.imports
        ],
    )


def refresh_project_index(project_root: str, *, db_path: str = CODE_INDEX_DB_PATH) -> dict:
    root = _normalize_root(project_root)
    stats = {
        "project_root": root,
        "db_path": db_path,
        "total": 0,
        "indexed": 0,
        "unchanged": 0,
        "failed": 0,
        "removed": 0,
    }

    with _connect(db_path) as conn:
        current_files: dict[str, tuple[str, os.stat_result]] = {}
        for absolute_path in _iter_python_files(root):
            rel = _relative_path(root, absolute_path)
            try:
                stat_result = os.stat(absolute_path)
            except OSError:
                continue
            current_files[rel] = (absolute_path, stat_result)
        stats["total"] = len(current_files)

        existing_rows = conn.execute(
            "SELECT file_path, mtime_ns, size, status FROM files WHERE project_root = ?",
            (root,),
        ).fetchall()
        existing = {row["file_path"]: row for row in existing_rows}

        missing = set(existing) - set(current_files)
        for rel in missing:
            conn.execute("DELETE FROM files WHERE project_root = ? AND file_path = ?", (root, rel))
            conn.execute("DELETE FROM symbols WHERE project_root = ? AND file_path = ?", (root, rel))
            conn.execute("DELETE FROM calls WHERE project_root = ? AND file_path = ?", (root, rel))
            conn.execute("DELETE FROM imports WHERE project_root = ? AND file_path = ?", (root, rel))
            stats["removed"] += 1

        for rel, (absolute_path, stat_result) in current_files.items():
            old = existing.get(rel)
            if (
                old is not None
                and int(old["mtime_ns"]) == int(stat_result.st_mtime_ns)
                and int(old["size"]) == int(stat_result.st_size)
                and old["status"] == "indexed"
            ):
                stats["unchanged"] += 1
                continue

            try:
                text = _read_indexable_text(absolute_path)
                tree = ast.parse(text, filename=absolute_path)
                visitor = _IndexVisitor()
                visitor.visit(tree)
                _replace_file_index(conn, root, rel, absolute_path, stat_result, visitor, status="indexed")
                stats["indexed"] += 1
            except Exception as exc:
                _replace_file_index(
                    conn,
                    root,
                    rel,
                    absolute_path,
                    stat_result,
                    None,
                    status="failed",
                    error=str(exc),
                )
                stats["failed"] += 1

        conn.commit()

    return stats


def _find_indexed_file(conn: sqlite3.Connection, project_root: str, query: str) -> str:
    candidate = (query or "").strip()
    if not candidate:
        raise ValueError("该模式需要 query 指定模块路径或文件名。")

    normalized = candidate.replace(os.sep, "/")
    if os.path.isabs(candidate):
        try:
            normalized = _relative_path(project_root, os.path.realpath(candidate))
        except Exception:
            normalized = candidate

    direct = conn.execute(
        """
        SELECT file_path FROM files
        WHERE project_root = ? AND file_path = ? AND status = 'indexed'
        LIMIT 1
        """,
        (project_root, normalized),
    ).fetchone()
    if direct:
        return direct["file_path"]

    like = f"%{normalized.lower()}%"
    matches = conn.execute(
        """
        SELECT file_path FROM files
        WHERE project_root = ? AND status = 'indexed'
          AND (lower(file_path) LIKE ? OR lower(file_path) LIKE ?)
        ORDER BY length(file_path), file_path
        LIMIT 1
        """,
        (project_root, like, f"%/{os.path.basename(normalized).lower()}"),
    ).fetchone()
    if not matches:
        raise FileNotFoundError(f"未找到匹配模块或文件：{query}")
    return matches["file_path"]


def search_symbols(project_root: str, query: str, max_results: int) -> str:
    root = _normalize_root(project_root)
    refresh_project_index(root)
    normalized_query = (query or "").strip().lower()
    pattern = f"%{normalized_query}%"
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT file_path, line, kind, qualified_name
            FROM symbols
            WHERE project_root = ?
              AND (? = '' OR lower(name) LIKE ? OR lower(qualified_name) LIKE ?)
            ORDER BY file_path, line
            LIMIT ?
            """,
            (root, normalized_query, pattern, pattern, max_results),
        ).fetchall()
    if not rows:
        return "未找到匹配的 Python 函数或类。"
    lines = ["符号索引结果："]
    for row in rows:
        lines.append(f"- {row['file_path']}:{row['line']} [{row['kind']}] {row['qualified_name']}")
    return "\n".join(lines)


def search_callers(project_root: str, query: str, max_results: int) -> str:
    if not (query or "").strip():
        return "callers 模式下 query 必须是函数名或方法名。"
    root = _normalize_root(project_root)
    refresh_project_index(root)
    normalized_query = query.strip()
    with _connect() as conn:
        definitions = conn.execute(
            """
            SELECT file_path, line, kind, name, qualified_name
            FROM symbols
            WHERE project_root = ?
              AND (
                lower(name) = lower(?)
                OR lower(qualified_name) = lower(?)
                OR lower(qualified_name) LIKE ?
              )
            ORDER BY file_path, line
            LIMIT ?
            """,
            (root, normalized_query, normalized_query, f"%.{normalized_query.lower()}", max_results),
        ).fetchall()

        if not definitions:
            rows = conn.execute(
                """
                SELECT file_path, line, caller_scope, callee_name
                FROM calls
                WHERE project_root = ?
                  AND (callee_name = ? OR callee_name LIKE ? OR callee_leaf = ?)
                ORDER BY file_path, line
                LIMIT ?
                """,
                (root, normalized_query, f"%.{normalized_query}", normalized_query, max_results),
            ).fetchall()
            lines = ["调用点搜索结果："]
            if not rows:
                lines.append("未找到调用点。")
            for row in rows:
                lines.append(f"{row['file_path']}:{row['line']} scope={row['caller_scope']} call={row['callee_name']}")
            return "\n".join(lines)

        lines = ["调用点搜索结果："]
        total_results = 0
        seen_calls: set[tuple[str, int, str, str]] = set()
        for definition in definitions:
            if total_results >= max_results:
                break
            target_module = _module_name_from_file(root, definition["file_path"])
            candidate_lines: list[str] = []

            local_rows = conn.execute(
                """
                SELECT file_path, line, caller_scope, callee_name
                FROM calls
                WHERE project_root = ?
                  AND file_path = ?
                  AND (callee_name = ? OR callee_name LIKE ? OR callee_leaf = ?)
                ORDER BY line
                """,
                (root, definition["file_path"], normalized_query, f"%.{normalized_query}", normalized_query),
            ).fetchall()
            for row in local_rows:
                key = (row["file_path"], row["line"], row["caller_scope"], row["callee_name"])
                if key in seen_calls:
                    continue
                seen_calls.add(key)
                candidate_lines.append(
                    f"- {row['file_path']}:{row['line']} scope={row['caller_scope']} "
                    f"call={row['callee_name']} via=local_definition"
                )
                total_results += 1
                if total_results >= max_results:
                    break

            if total_results < max_results:
                import_rows = conn.execute(
                    """
                    SELECT file_path, module, name, alias
                    FROM imports
                    WHERE project_root = ? AND file_path != ?
                    ORDER BY file_path, line
                    """,
                    (root, definition["file_path"]),
                ).fetchall()
                imports_by_file: dict[str, list[sqlite3.Row]] = {}
                for row in import_rows:
                    imports_by_file.setdefault(row["file_path"], []).append(row)

                for file_path, bindings in imports_by_file.items():
                    if total_results >= max_results:
                        break
                    allowed_names = _allowed_call_names_for_definition(bindings, target_module, definition["name"])
                    if not allowed_names:
                        continue
                    placeholders = ",".join("?" for _ in allowed_names)
                    call_rows = conn.execute(
                        f"""
                        SELECT file_path, line, caller_scope, callee_name
                        FROM calls
                        WHERE project_root = ?
                          AND file_path = ?
                          AND callee_name IN ({placeholders})
                        ORDER BY line
                        """,
                        (root, file_path, *allowed_names.keys()),
                    ).fetchall()
                    for row in call_rows:
                        key = (row["file_path"], row["line"], row["caller_scope"], row["callee_name"])
                        if key in seen_calls:
                            continue
                        seen_calls.add(key)
                        candidate_lines.append(
                            f"- {row['file_path']}:{row['line']} scope={row['caller_scope']} "
                            f"call={row['callee_name']} via={allowed_names[row['callee_name']]}"
                        )
                        total_results += 1
                        if total_results >= max_results:
                            break

            if candidate_lines:
                lines.append(
                    f"定义候选：{definition['file_path']}:{definition['line']} "
                    f"[{definition['kind']}] {definition['qualified_name']}"
                )
                lines.extend(candidate_lines)

        if len(lines) == 1:
            rows = conn.execute(
            """
            SELECT file_path, line, caller_scope, callee_name
            FROM calls
            WHERE project_root = ?
              AND (callee_name = ? OR callee_name LIKE ? OR callee_leaf = ?)
            ORDER BY file_path, line
            LIMIT ?
            """,
                (root, normalized_query, f"%.{normalized_query}", normalized_query, max_results),
            ).fetchall()
            if not rows:
                lines.append("未找到调用点。")
            else:
                for row in rows:
                    lines.append(f"{row['file_path']}:{row['line']} scope={row['caller_scope']} call={row['callee_name']}")
        return "\n".join(lines)


def dependency_summary(project_root: str, query: str, max_results: int) -> str:
    root = _normalize_root(project_root)
    refresh_project_index(root)
    with _connect() as conn:
        target = _find_indexed_file(conn, root, query)
        target_module = target[:-3].replace("/", ".")
        if target_module.endswith(".__init__"):
            target_module = target_module[: -len(".__init__")]

        imports = [
            row["module"]
            for row in conn.execute(
                """
                SELECT DISTINCT module
                FROM imports
                WHERE project_root = ? AND file_path = ?
                ORDER BY module
                LIMIT ?
                """,
                (root, target, max_results),
            ).fetchall()
            if row["module"]
        ]
        reverse_rows = conn.execute(
            """
            SELECT DISTINCT file_path
            FROM imports
            WHERE project_root = ?
              AND file_path != ?
              AND (ltrim(module, '.') = ? OR ltrim(module, '.') LIKE ?)
            ORDER BY file_path
            LIMIT ?
            """,
            (root, target, target_module, f"{target_module}.%", max_results),
        ).fetchall()

    lines = [f"模块依赖分析：{target}", "", "直接 imports："]
    lines.extend(f"- {item}" for item in imports)
    if not imports:
        lines.append("- 无")
    lines.extend(["", "项目内反向依赖："])
    lines.extend(f"- {row['file_path']}" for row in reverse_rows)
    if not reverse_rows:
        lines.append("- 未发现其他 Python 文件直接 import 该模块")
    return "\n".join(lines)


def data_flow_summary(project_root: str, query: str, max_results: int) -> str:
    root = _normalize_root(project_root)
    refresh_project_index(root)
    with _connect() as conn:
        target = _find_indexed_file(conn, root, query)
        imports = [
            row["module"]
            for row in conn.execute(
                """
                SELECT DISTINCT module FROM imports
                WHERE project_root = ? AND file_path = ?
                ORDER BY module
                LIMIT ?
                """,
                (root, target, max_results),
            ).fetchall()
            if row["module"]
        ]
        symbols = conn.execute(
            """
            SELECT kind, qualified_name, line FROM symbols
            WHERE project_root = ? AND file_path = ?
            ORDER BY line
            LIMIT ?
            """,
            (root, target, max_results),
        ).fetchall()
        io_calls = conn.execute(
            """
            SELECT DISTINCT line, callee_name FROM calls
            WHERE project_root = ? AND file_path = ? AND lower(callee_leaf) IN ({})
            ORDER BY line
            LIMIT ?
            """.format(",".join("?" for _ in IO_CALL_NAMES)),
            (root, target, *sorted(IO_CALL_NAMES), max_results),
        ).fetchall()
        ml_calls = conn.execute(
            """
            SELECT DISTINCT line, callee_name FROM calls
            WHERE project_root = ? AND file_path = ? AND lower(callee_leaf) IN ({})
            ORDER BY line
            LIMIT ?
            """.format(",".join("?" for _ in ML_CALL_NAMES)),
            (root, target, *sorted(ML_CALL_NAMES), max_results),
        ).fetchall()
        top_level_calls = conn.execute(
            """
            SELECT DISTINCT line, callee_name FROM calls
            WHERE project_root = ? AND file_path = ? AND is_top_level = 1
            ORDER BY line
            LIMIT ?
            """,
            (root, target, max_results),
        ).fetchall()

    lines = [f"模块数据流/结构摘要：{target}", "", "1. 依赖输入："]
    lines.extend(f"- import {item}" for item in imports)
    if not imports:
        lines.append("- 无显式 import")

    lines.extend(["", "2. 主要结构："])
    if symbols:
        for row in symbols:
            lines.append(f"- {row['kind']} {row['qualified_name']} @ line {row['line']}")
    else:
        lines.append("- 未发现函数或类定义")

    lines.extend(["", "3. I/O 或数据读写线索："])
    lines.extend(f"- line {row['line']}:{row['callee_name']}" for row in io_calls)
    if not io_calls:
        lines.append("- 未发现明显文件/数据读写调用")

    lines.extend(["", "4. 训练/推理相关线索："])
    lines.extend(f"- line {row['line']}:{row['callee_name']}" for row in ml_calls)
    if not ml_calls:
        lines.append("- 未发现明显训练/推理调用")

    lines.extend(["", "5. 顶层调用线索："])
    lines.extend(f"- line {row['line']}:{row['callee_name']}" for row in top_level_calls)
    if not top_level_calls:
        lines.append("- 未发现明显顶层调用")

    lines.extend(["", "6. 解释建议："])
    lines.append("- 结合 symbol/callers/dependencies 模式继续追踪关键函数的上游调用和下游依赖。")
    return "\n".join(lines)


def entrypoint_summary(project_root: str, max_results: int) -> str:
    root = _normalize_root(project_root)
    refresh_project_index(root)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT file_path, entrypoint_score, entrypoint_reasons_json
            FROM files
            WHERE project_root = ? AND status = 'indexed' AND entrypoint_score > 0
            ORDER BY entrypoint_score DESC, file_path
            LIMIT ?
            """,
            (root, max_results),
        ).fetchall()
    if not rows:
        return "未发现明显训练/运行入口。可尝试 text 模式搜索 train、main、argparse、Trainer。"

    lines = ["可能的训练/运行入口："]
    for row in rows:
        try:
            reasons = json.loads(row["entrypoint_reasons_json"] or "[]")
        except json.JSONDecodeError:
            reasons = []
        lines.append(f"- {row['file_path']} score={row['entrypoint_score']} | {'; '.join(reasons)}")
    return "\n".join(lines)
