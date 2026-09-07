#!/usr/bin/env python3
"""Serve the local learning guide with Python's standard library only."""
from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "docs" / "web"
GUIDE = ROOT / "docs" / "learning-guide.zh-CN.md"
SOURCE_ROOTS = ("python/minisgl", "tests", "benchmark", "docs")
SOURCE_FILES = {"README.md", "pyproject.toml", "Dockerfile", "LICENSE"}
SOURCE_SUFFIXES = {".py", ".md", ".toml", ".cpp", ".cu", ".cuh", ".h", ".txt"}
MAX_SOURCE_SIZE = 1024 * 1024


def source_path(value: str) -> Path:
    """Only expose documentation and code; never hidden files or paths outside the repo."""
    if not value or "\\" in value or "\0" in value:
        raise ValueError("无效的源码路径")
    parts = PurePosixPath(value)
    if parts.is_absolute() or any(part.startswith(".") for part in parts.parts):
        raise ValueError("此路径不在可浏览的源码范围内")
    logical = parts.as_posix()
    allowed = logical in SOURCE_FILES or any(
        logical == base or logical.startswith(base + "/") for base in SOURCE_ROOTS
    )
    path = (ROOT / logical).resolve()
    if not allowed or not path.is_relative_to(ROOT):
        raise ValueError("此路径不在可浏览的源码范围内")
    # A symlink must not turn an allowed code directory into another repository area.
    resolved_relative = path.relative_to(ROOT).as_posix()
    if resolved_relative != logical:
        raise ValueError("源码浏览不跟随符号链接")
    if path.is_file() and logical not in SOURCE_FILES and path.suffix not in SOURCE_SUFFIXES:
        raise ValueError("此文件类型不支持源码浏览")
    return path


def source_payload(value: str) -> dict:
    path = source_path(value)
    if not path.exists():
        raise FileNotFoundError("源码文件不存在")
    if path.is_dir():
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
            relative = child.relative_to(ROOT).as_posix()
            try:
                source_path(relative)
            except ValueError:
                continue
            entries.append({"name": child.name, "path": relative, "kind": "directory" if child.is_dir() else "file"})
        return {"kind": "directory", "path": value, "entries": entries}
    if path.stat().st_size > MAX_SOURCE_SIZE:
        raise ValueError("文件过大，无法在源码窗口中显示")
    return {"kind": "file", "path": value, "content": path.read_text(encoding="utf-8")}


@lru_cache(maxsize=32)
def static_content(path: Path, modified_ns: int, compress: bool) -> bytes:
    content = path.read_bytes()
    return gzip.compress(content, compresslevel=5) if compress else content


class DocsHandler(BaseHTTPRequestHandler):
    server_version = "MiniSGLDocs/1.0"

    def log_message(self, format: str, *args) -> None:
        if args and isinstance(args[1] if len(args) > 1 else None, str) and args[1] == "200":
            return
        super().log_message(format, *args)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        # Reject foreign Host headers so DNS rebinding cannot expose local source files.
        try:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname
        except ValueError:
            host = None
        if host not in {"127.0.0.1", "localhost"}:
            return self.error_response(403, "文档服务仅支持本机地址")
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/guide":
                return self.respond(GUIDE.read_bytes(), "text/markdown; charset=utf-8")
            if parsed.path == "/api/source":
                values = parse_qs(parsed.query).get("path", [])
                if len(values) != 1:
                    raise ValueError("需要一个源码路径")
                payload = source_payload(values[0])
                return self.respond(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            if parsed.path in {"/", "/index.html"}:
                target = WEB_ROOT / "index.html"
            elif parsed.path.startswith("/assets/"):
                relative = unquote(parsed.path.removeprefix("/assets/"))
                if any(part.startswith(".") for part in PurePosixPath(relative).parts):
                    raise FileNotFoundError("页面不存在")
                target = (WEB_ROOT / relative).resolve()
                if not target.is_relative_to(WEB_ROOT) or not target.is_file():
                    raise FileNotFoundError("页面不存在")
            else:
                raise FileNotFoundError("页面不存在")
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or target.suffix == ".js":
                content_type += "; charset=utf-8"
            compressed = "gzip" in self.headers.get("Accept-Encoding", "") and target.stat().st_size > 2048
            data = static_content(target, target.stat().st_mtime_ns, compressed)
            return self.respond(data, content_type, compressed=compressed)
        except (FileNotFoundError, IsADirectoryError):
            self.error_response(404, "页面或源码不存在")
        except (ValueError, UnicodeDecodeError):
            self.error_response(400, "无法读取该源码路径")
        except OSError:
            self.error_response(500, "本地文件读取失败，请检查文件是否可访问")

    def error_response(self, status: int, message: str) -> None:
        self.respond(json.dumps({"error": message}, ensure_ascii=False).encode(), "application/json; charset=utf-8", status=status)

    def respond(self, data: bytes, content_type: str, *, status: int = 200, compressed: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("端口必须是整数") from error
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 0–65535 之间")
    return port


def main() -> int:
    if sys.version_info < (3, 10):
        print("文档服务需要 Python 3.10 或更新版本。", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(description="在本机打开 Mini-SGLang 中文学习指南。仅依赖 Python 3.10+。")
    parser.add_argument("--port", type=port_number, default=8765, help="本地端口，默认 8765；0 表示由系统分配")
    parser.add_argument("--no-open", action="store_true", help="只启动服务，不自动打开浏览器")
    args = parser.parse_args()
    required = [GUIDE, WEB_ROOT / "index.html", WEB_ROOT / "reader.js", WEB_ROOT / "document-model.mjs", WEB_ROOT / "reader.css", WEB_ROOT / "favicon.svg"]
    required += [WEB_ROOT / "vendor" / name for name in ("marked.js", "mermaid.js", "purify.js", "highlight.js")]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        print("文档文件不完整：" + "、".join(missing), file=sys.stderr)
        return 1
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), DocsHandler)
    except OSError as error:
        print(f"无法启动文档服务：{error}\n可指定其他端口：./start-docs.sh --port 8766", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"\nMini-SGLang 源码学习指南\n\n  {url}\n\n修改 Markdown 后刷新网页即可。按 Ctrl+C 停止服务。\n", flush=True)
    if not args.no_open:
        timer = threading.Timer(0.4, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n文档服务已停止。", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
