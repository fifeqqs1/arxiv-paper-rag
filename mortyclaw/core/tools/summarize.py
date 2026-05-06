import os
import shutil
import subprocess
import sys
from io import BytesIO
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import mortyclaw_tool
from ..config import OFFICE_DIR


ALLOWED_SUMMARY_LENGTHS = {"short", "medium", "long", "xl", "xxl"}
CODE_AND_CONFIG_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".env",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".lock",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
SENSITIVE_FILENAMES = {".env", ".npmrc", ".pypirc"}
SUMMARIZE_TIMEOUT_SECONDS = 300
SUMMARIZE_OUTPUT_LIMIT = 12000
PDF_DOWNLOAD_TIMEOUT_SECONDS = 60
PDF_DOWNLOAD_LIMIT_BYTES = 50 * 1024 * 1024
LOCAL_TEXT_LIMIT_BYTES = 5 * 1024 * 1024
LOCAL_BINARY_MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".png",
    ".wav",
    ".webm",
    ".webp",
}


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_within_dir(target_path: str, base_dir: str) -> bool:
    try:
        target_real = os.path.realpath(target_path)
        base_real = os.path.realpath(base_dir)
        return os.path.commonpath([target_real, base_real]) == base_real
    except ValueError:
        return False


def _has_blocked_code_or_config_extension(source: str) -> bool:
    parsed = urlparse(source)
    candidate_path = parsed.path if parsed.scheme else source
    basename = os.path.basename(candidate_path).lower()
    if basename in SENSITIVE_FILENAMES:
        return True
    _, ext = os.path.splitext(basename)
    return ext.lower() in CODE_AND_CONFIG_EXTENSIONS


def _is_pdf_source(source: str) -> bool:
    parsed = urlparse(source)
    candidate_path = parsed.path if parsed.scheme else source
    return candidate_path.lower().endswith(".pdf")


def _is_local_binary_media_source(source: str) -> bool:
    _, ext = os.path.splitext(source.lower())
    return ext in LOCAL_BINARY_MEDIA_EXTENSIONS


def _resolve_summary_source(source: str) -> str:
    normalized = os.path.expanduser((source or "").strip())
    if not normalized:
        raise ValueError("source 不能为空。")

    if _has_blocked_code_or_config_extension(normalized):
        raise PermissionError("summarize_content 不处理代码或项目配置文件。请使用 MortyClaw 原有代码分析能力。")

    if _is_url(normalized):
        return normalized

    if os.path.isabs(normalized):
        resolved = os.path.realpath(normalized)
    else:
        resolved = os.path.realpath(os.path.join(OFFICE_DIR, normalized))
        if not _is_within_dir(resolved, OFFICE_DIR):
            raise PermissionError("越权拦截：相对路径只能指向 workspace/office 内的文件。")

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"文件不存在：{source}")
    if os.path.isdir(resolved):
        raise IsADirectoryError(f"summarize_content 需要文件或 URL，不能直接总结目录：{source}")
    return resolved


def _find_summarize_binary() -> str | None:
    summarize_bin = shutil.which("summarize")
    if summarize_bin:
        return summarize_bin

    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    candidate_dirs = [
        os.path.join(sys.prefix, scripts_dir),
        os.path.dirname(sys.executable),
    ]
    seen: set[str] = set()
    executable_names = ["summarize.cmd", "summarize.exe", "summarize"] if os.name == "nt" else ["summarize"]
    for candidate_dir in candidate_dirs:
        for executable_name in executable_names:
            candidate = os.path.join(candidate_dir, executable_name)
            candidate_realpath = os.path.realpath(candidate)
            if candidate_realpath in seen:
                continue
            seen.add(candidate_realpath)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _build_summarize_command(source: str) -> list[str]:
    summarize_bin = _find_summarize_binary()
    if summarize_bin:
        return [summarize_bin, source, "--extract-only"]

    npx_bin = shutil.which("npx")
    if not npx_bin:
        raise FileNotFoundError(
            "summarize_content 不可用：未找到 summarize 或 npx。"
            "请执行 `npm i -g @steipete/summarize`，或安装 Node/npm 后使用 npx。"
        )
    return [npx_bin, "-y", "@steipete/summarize", source, "--extract-only"]


def _truncate_output(text: str, limit: int = SUMMARIZE_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...[summarize 输出过长，已安全截断]..."


def _read_url_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "MortyClaw/1.0"})
    with urlopen(request, timeout=PDF_DOWNLOAD_TIMEOUT_SECONDS) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > PDF_DOWNLOAD_LIMIT_BYTES:
            raise ValueError("PDF 文件过大，超过 50MB 安全限制。")
        data = response.read(PDF_DOWNLOAD_LIMIT_BYTES + 1)

    if len(data) > PDF_DOWNLOAD_LIMIT_BYTES:
        raise ValueError("PDF 文件过大，超过 50MB 安全限制。")
    return data


def _extract_pdf_text(source: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF 本地抽取不可用：缺少 pypdf，请执行 `pip install pypdf`。") from exc

    if _is_url(source):
        pdf_stream = BytesIO(_read_url_bytes(source))
        reader = PdfReader(pdf_stream)
    else:
        with open(source, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            return _extract_text_from_pdf_reader(reader)
    return _extract_text_from_pdf_reader(reader)


def _extract_text_from_pdf_reader(reader) -> str:
    parts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            parts.append(f"[PDF 第 {index} 页]\n{page_text}")

    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("PDF 未抽取到文本，可能是扫描件或图片型 PDF；当前本地模式暂不做 OCR。")
    return text


def _extract_local_text_document(source: str) -> str:
    if _is_local_binary_media_source(source):
        raise ValueError("本地图片/音频/视频暂不支持无模型抽取；请提供可公开访问的 URL，或先转换为文本/PDF。")

    file_size = os.path.getsize(source)
    if file_size > LOCAL_TEXT_LIMIT_BYTES:
        raise ValueError("本地文本文件过大，超过 5MB 安全限制。")

    with open(source, "rb") as document_file:
        data = document_file.read(LOCAL_TEXT_LIMIT_BYTES + 1)

    if b"\x00" in data[:4096]:
        raise ValueError("本地文件看起来不是可直接抽取的文本；当前本地模式支持 PDF 和普通文本/Markdown。")

    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("本地文本文件为空，未抽取到可总结内容。")
    return text


def _format_extracted_content(text: str, length: str, force_summary: bool) -> str:
    force_line = "是" if force_summary else "否"
    return (
        "summarize_content 已完成外部内容抽取。"
        "请基于下面的抽取内容，用 MortyClaw 当前 Agent 自己完成总结，"
        "不要声称外部 summarize 已经生成最终摘要。\n"
        f"摘要长度偏好：{length}\n"
        f"强制摘要：{force_line}\n\n"
        f"{_truncate_output(text)}"
    )


@mortyclaw_tool
def summarize_content(source: str, length: str = "medium", force_summary: bool = False) -> str:
    """
    抽取网页、YouTube、Podcast/RSS、PDF、图片、音频、视频或普通文本/Markdown 文档内容，
    再交给 MortyClaw 当前 Agent 自己总结。

    严格边界：
    - 只在用户明确要求“总结/概括/摘要网页、链接、PDF、视频、音频、播客或普通文档”时调用。
    - 外部 summarize CLI 只用于抓取/抽取内容，不用于调用外部 LLM 生成最终摘要。
    - 不用于读取代码、分析项目、检查错误、debug、review 或理解仓库。
    - 禁止处理代码和项目配置文件，例如 .py/.js/.ts/.json/.yaml/.toml/.env/.lock 等。
    - 本地代码读取和项目分析必须继续使用 read_office_file/list_office_files/execute_office_shell 以及 MortyClaw 原有分析能力。
    - length 是给 MortyClaw Agent 的摘要长度偏好，不传给外部 CLI 生成摘要。
    """
    normalized_length = (length or "medium").strip().lower()
    if normalized_length not in ALLOWED_SUMMARY_LENGTHS:
        return "summarize_content 参数错误：length 只能是 short、medium、long、xl、xxl。"

    try:
        resolved_source = _resolve_summary_source(source)
        if _is_pdf_source(resolved_source):
            pdf_text = _extract_pdf_text(resolved_source)
            return _format_extracted_content(pdf_text, normalized_length, bool(force_summary))
        if not _is_url(resolved_source):
            local_text = _extract_local_text_document(resolved_source)
            return _format_extracted_content(local_text, normalized_length, bool(force_summary))

        command = _build_summarize_command(resolved_source)
        result = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUMMARIZE_TIMEOUT_SECONDS,
        )
    except PermissionError as exc:
        return str(exc)
    except FileNotFoundError as exc:
        return str(exc)
    except IsADirectoryError as exc:
        return str(exc)
    except ValueError as exc:
        return f"summarize_content 参数错误：{exc}"
    except RuntimeError as exc:
        return str(exc)
    except subprocess.TimeoutExpired:
        return f"summarize_content 执行超时：summarize 超过 {SUMMARIZE_TIMEOUT_SECONDS}s 未返回。"
    except Exception as exc:
        return f"summarize_content 执行异常：{exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        detail = stderr or stdout or f"exit code {result.returncode}"
        return f"summarize_content 执行失败：{_truncate_output(detail, 2000)}"

    if not stdout:
        return "summarize_content 未抽取到可总结内容。"
    return _format_extracted_content(stdout, normalized_length, bool(force_summary))
