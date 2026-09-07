# Mini-SGLang 本地学习文档

这是 [中文源码学习指南](../learning-guide.zh-CN.md) 的网页阅读器，包含完整的 21 章内容。正文直接读取 Markdown 文件，修改文档后刷新网页即可。

## 一键启动

在仓库根目录执行：

```bash
./start-docs.sh
```

浏览器会自动打开 **http://127.0.0.1:8765/**。终端保持运行，按 **Ctrl+C** 停止服务。

仅需要 **Python 3.10 或更新版本**，不需要安装项目的 Python 包、CUDA、模型权重、Node.js 或 npm 依赖。启动脚本适用于 macOS / Linux；也可以直接运行跨平台的 Python 入口：

```bash
python3 scripts/serve_docs.py
```

Windows 使用 `py -3 scripts/serve_docs.py`。文档服务可以在没有 GPU 的电脑上运行；实际模型推理所需环境见学习指南第 17 章。

### 启动选项

```bash
# 端口被占用时换一个端口
./start-docs.sh --port 8766

# 不自动打开浏览器
./start-docs.sh --no-open

# 由系统分配空闲端口，以终端打印的地址为准
./start-docs.sh --port 0 --no-open

# 查看帮助
./start-docs.sh --help

# 指定 Python 解释器
MINISGL_DOCS_PYTHON=/path/to/python3 ./start-docs.sh
```

脚本根据自身位置定位仓库，可以从其他目录用绝对路径启动。服务只监听 `127.0.0.1`，用于本机阅读。

## 阅读方式

- **章节目录**：左侧按学习阶段组织 21 章；小屏幕使用顶部“目录”按钮。
- **本章目录**：桌面右侧定位小节；较窄屏幕可展开正文上方的“本章目录”。
- **连续阅读**：每章末尾有上一章、下一章入口；地址中的章节和小节锚点支持刷新与浏览器前进、后退。
- **源码查看**：点击正文中的本地源码链接，在窗口中查看带行号的源文件或浏览目录。源码窗口同样读取当前磁盘文件。
- **代码与图表**：代码支持语法高亮和复制；第 4、5 章的项目架构图与执行流程图可点击查看原图。3 张 Mermaid 图表在本地绘制，支持滚动和“放大查看”，同时保留可展开的图表源码。
- **保留原文**：顶部“Markdown”可下载完整原稿。网页的“阅读说明”保留分析基线和验证范围。

正常阅读不访问 CDN，渲染库、样式和字体配置均使用本地资源。原文引用的外部论文、官方资料仍需联网打开。

源码浏览入口仅开放仓库中的指定代码和文档目录，不提供整个工作目录的 HTTP 文件服务。隐藏路径、越界路径和符号链接不会作为源码展示。

## 内容与实现

```text
start-docs.sh                      一键启动入口
scripts/serve_docs.py              Python 标准库 HTTP 服务
scripts/vendor_docs.py             维护者更新本地渲染库的工具
docs/learning-guide.zh-CN.md        完整正文，唯一内容源
docs/web/
├── index.html                    页面结构
├── reader.css                    阅读布局与移动端样式
├── reader.js                     导航、渲染、源码窗口与复制
├── document-model.mjs             章节切分与链接解析
├── favicon.svg                   页面图标
├── figures/                      架构、执行流程配图及生成提示词
└── vendor/                       固定版本的渲染库及许可证
```

这是静态前端与轻量文件服务，不需要构建步骤。服务提供三个入口：

| 路径 | 用途 |
| --- | --- |
| `/`、`/assets/…` | 页面及本地资源 |
| `/api/guide` | 原始 Markdown |
| `/api/source?path=python/minisgl/core.py` | 指定源码文件或目录的 JSON 数据 |

新增章节时保留原文约定：先写 `<a id="22-example"></a>`，紧接 `## 22. 章节标题`。页面会据此生成导航。小节使用 `###` 标题。

### 维护与检查

启动和阅读无需下载依赖。仅在维护者主动更新本地库时运行：

```bash
python3 scripts/vendor_docs.py
```

该脚本从 npm registry 下载代码中指定的固定版本，校验包完整性，保留许可证，并更新 [依赖清单与 SHA-256](vendor/README.md)。升级时应先修改脚本中的版本，再检查 Markdown、高亮、消毒和图表渲染的兼容性。

服务与内容解析有独立检查，不加载 Mini-SGLang 或 GPU 依赖：

```bash
# Python 标准库测试：文件访问边界、HTTP 响应、资源加载
python3 -m unittest discover -s tests/docs -p 'test_*.py' -v

# 可选的维护者检查，需要 Node.js 20+，无需 npm install
node --test tests/docs/document-model.test.mjs
node --check docs/web/reader.js
```

这两组测试检查服务行为和文档结构，不替代真实浏览器中的视觉与交互验收，也不运行推理功能。

Mermaid 使用根级 `htmlLabels: false` 生成 SVG 文本节点，并继续通过 DOMPurify 的 SVG 规则过滤输出。根级选项在当前版本中优先于已弃用的 `flowchart.htmlLabels`，详见 [Mermaid 官方配置说明](https://mermaid.js.org/config/schema-docs/config-properties-htmllabels.html)。升级图表依赖后，应在浏览器确认三张图的节点文字、箭头、滚动与“放大查看”；仅通过语法解析不足以确认文字实际显示。

## 常见问题

**提示端口被占用**：使用 `--port 8766` 或 `--port 0`。

**浏览器没有自动打开**：复制终端打印的完整地址，在浏览器中打开；命令行或无桌面环境可使用 `--no-open`。

**找不到 Python**：安装 Python 3.10+，或通过 `MINISGL_DOCS_PYTHON` 指定解释器路径。

**显示“文档文件不完整”**：检查 Markdown、页面文件和 `vendor` 目录是否全部保留；正常运行无需重新下载依赖。

**双击 HTML 后正文没有加载**：请通过启动脚本打开 HTTP 地址。正文和源码由本地服务读取，直接打开 `file://` 页面不支持这些功能。
