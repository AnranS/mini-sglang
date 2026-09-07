#!/usr/bin/env python3
"""Refresh the pinned browser libraries used by the offline documentation reader.

Maintenance command only. The documentation launcher never installs dependencies.
"""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request
import json

VENDOR_DIR = Path(__file__).resolve().parents[1] / "docs" / "web" / "vendor"
PACKAGES = (
    ("marked", "18.0.11", "lib/marked.umd.js", "marked.js"),
    ("mermaid", "11.17.2", "dist/mermaid.min.js", "mermaid.js"),
    ("dompurify", "3.4.15", "dist/purify.min.js", "purify.js"),
    ("@highlightjs/cdn-assets", "11.12.0", "highlight.min.js", "highlight.js"),
)


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, version, member, output in PACKAGES:
        metadata = json.loads(download(f"https://registry.npmjs.org/{name}/{version}"))
        distribution = metadata["dist"]
        archive = download(distribution["tarball"])
        algorithm, expected = distribution["integrity"].split("-", 1)
        actual = base64.b64encode(hashlib.new(algorithm, archive).digest()).decode()
        if actual != expected:
            raise RuntimeError(f"Integrity check failed for {name}@{version}")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
            selected = package.extractfile(f"package/{member}")
            if selected is None:
                raise RuntimeError(f"Missing browser bundle: {name}/{member}")
            bundle = selected.read()
            license_name = next(
                item.name for item in package.getmembers()
                if item.name.rsplit("/", 1)[0] == "package"
                and item.name.rsplit("/", 1)[-1].lower() in {"license", "license.md", "license.txt"}
            )
            license_file = package.extractfile(license_name)
            if license_file is None:
                raise RuntimeError(f"Missing license for {name}")
            license_bytes = license_file.read()
        # Keep the upstream license banner; the reader does not need source maps.
        import re
        bundle = re.sub(rb"(?m)^//# sourceMappingURL=.*$", b"", bundle)
        (VENDOR_DIR / output).write_bytes(bundle)
        (VENDOR_DIR / f"{output}.LICENSE").write_bytes(license_bytes)
        sha = hashlib.sha256(bundle).hexdigest()
        rows.append(f"| {name} | {version} | [{output}]({output}) | [License]({output}.LICENSE) | `{sha}` |")
        print(f"Vendored {name}@{version}: {len(bundle):,} bytes", flush=True)
    (VENDOR_DIR / "README.md").write_text(
        "# Browser dependencies\n\n"
        "These pinned upstream distributions are served locally. Opening the documentation "
        "requires no package installation or CDN access.\n\n"
        "Refresh from the repository root with `python3 scripts/vendor_docs.py` (network required). "
        "The script verifies each npm archive against its registry integrity hash and retains "
        "the upstream license. Runtime dependencies are not added to the inference environment.\n\n"
        "| Package | Version | Local bundle | License | SHA-256 |\n"
        "| --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
