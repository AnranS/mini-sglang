# Browser dependencies

These pinned upstream distributions are served locally. Opening the documentation requires no package installation or CDN access.

Refresh from the repository root with `python3 scripts/vendor_docs.py` (network required). The script verifies each npm archive against its registry integrity hash and retains the upstream license. Runtime dependencies are not added to the inference environment.

| Package | Version | Local bundle | License | SHA-256 |
| --- | --- | --- | --- | --- |
| marked | 18.0.11 | [marked.js](marked.js) | [License](marked.js.LICENSE) | `bbb4d508f8ef9f26faac1317928da06e35fd839ac46583f50a264486b5d0ca93` |
| mermaid | 11.17.2 | [mermaid.js](mermaid.js) | [License](mermaid.js.LICENSE) | `581ed7d74bd9048d0e3a91363927d72ef22942d7722546b27f7cc29e35390eb8` |
| dompurify | 3.4.15 | [purify.js](purify.js) | [License](purify.js.LICENSE) | `570cafd8080aa3b60b44e2a8904921adb59977a2e861d66aa81db77d495ab3e5` |
| @highlightjs/cdn-assets | 11.12.0 | [highlight.js](highlight.js) | [License](highlight.js.LICENSE) | `8ab71eb09c51f501e5e25157d9cff100e46cc29bcbfc744d0b746d451fca7f53` |
