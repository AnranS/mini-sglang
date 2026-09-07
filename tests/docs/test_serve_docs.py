"""Documentation checks; run directly with unittest, without inference packages."""
from __future__ import annotations

import gzip
from http.client import HTTPConnection
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("serve_docs", REPO / "scripts" / "serve_docs.py")
serve_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve_docs)


class SourceAndHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="minisgl-docs-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.web = self.root / "docs" / "web"
        self.web.mkdir(parents=True)
        self.source = self.root / "python" / "minisgl"
        self.source.mkdir(parents=True)
        (self.source / "core.py").write_text('# 示例源码\nVALUE = 1\n', encoding="utf-8")
        (self.source / ".env").write_text("private test fixture")
        (self.source / "image.png").write_bytes(b"not source code")
        (self.root / "README.md").write_text("readme", encoding="utf-8")
        self.guide = self.root / "docs" / "learning-guide.zh-CN.md"
        self.guide.write_text("# 测试文档\n", encoding="utf-8")
        (self.web / "index.html").write_text("<!doctype html><title>学习指南</title>", encoding="utf-8")
        self.script = b"const local = true;\n" * 250
        (self.web / "reader.js").write_bytes(self.script)
        for name, value in {"ROOT": self.root, "WEB_ROOT": self.web, "GUIDE": self.guide}.items():
            patcher = patch.object(serve_docs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.server = serve_docs.ThreadingHTTPServer(("127.0.0.1", 0), serve_docs.DocsHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        serve_docs.static_content.cache_clear()

    def request(self, path, method="GET", headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_home_and_guide_refresh(self):
        status, headers, content = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("学习指南", content.decode())
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(self.request("/api/guide")[2], self.guide.read_bytes())
        self.guide.write_text("# 修改后的文档", encoding="utf-8")
        self.assertEqual(self.request("/api/guide")[2], self.guide.read_bytes())

    def test_local_assets_gzip_and_head(self):
        status, headers, content = self.request("/assets/reader.js", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertEqual(headers["Content-Encoding"], "gzip")
        self.assertEqual(gzip.decompress(content), self.script)
        status, headers, content = self.request("/assets/reader.js", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(content, b"")
        self.assertEqual(int(headers["Content-Length"]), len(self.script))

    def test_file_and_directory_source(self):
        status, _, content = self.request("/api/source?path=python/minisgl/core.py")
        self.assertEqual(status, 200)
        payload = json.loads(content)
        self.assertEqual(payload["kind"], "file")
        self.assertEqual(payload["content"], (self.source / "core.py").read_text())
        status, _, content = self.request("/api/source?path=python/minisgl")
        self.assertEqual(status, 200)
        self.assertEqual([entry["name"] for entry in json.loads(content)["entries"]], ["core.py"])

    def test_source_rejects_private_paths_traversal_and_binary_files(self):
        for path in [".git/config", ".env", "/etc/passwd", "python/minisgl/../../README.md", "python/minisgl/.env", "python/minisgl/image.png", "python\\minisgl\\core.py"]:
            with self.subTest(path=path):
                status, _, _ = self.request("/api/source?path=" + quote(path, safe=""))
                self.assertEqual(status, 400)

    def test_source_does_not_follow_symlinks(self):
        (self.source / "alias.py").symlink_to(self.root / "README.md")
        self.assertEqual(self.request("/api/source?path=python/minisgl/alias.py")[0], 400)

    def test_foreign_hosts_cannot_read_local_source(self):
        status, _, _ = self.request("/api/source?path=README.md", headers={"Host": "example.org"})
        self.assertEqual(status, 403)

    def test_missing_and_oversized_sources(self):
        self.assertEqual(self.request("/api/source?path=python/minisgl/missing.py")[0], 404)
        self.assertEqual(self.request("/api/source")[0], 400)
        self.assertEqual(self.request("/api/source?path=README.md&path=README.md")[0], 400)
        with patch.object(serve_docs, "MAX_SOURCE_SIZE", 1):
            self.assertEqual(self.request("/api/source?path=README.md")[0], 400)

    def test_static_routes_cannot_expose_other_repository_files(self):
        for path in ["/.git/config", "/README.md", "/assets/../learning-guide.zh-CN.md", "/assets/%2e%2e/%2e%2e/README.md", "/assets/%2fetc/passwd", "/assets/missing.js"]:
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)


class RepositoryContentTests(unittest.TestCase):
    def test_every_local_guide_link_is_available_to_source_viewer(self):
        markdown = (REPO / "docs" / "learning-guide.zh-CN.md").read_text(encoding="utf-8")
        links = re.findall(r"\]\((\.\./[^)]+)\)", markdown)
        self.assertGreater(len(links), 75)
        for link in set(links):
            path = (REPO / "docs" / link).resolve().relative_to(REPO).as_posix()
            with self.subTest(path=path):
                self.assertIn(serve_docs.source_payload(path)["kind"], ("file", "directory"))


if __name__ == "__main__":
    unittest.main()
