# Mini-SGLang 源码学习站

线上地址：<https://anrans.github.io/mini-sglang/>

这是一个 [Astro Starlight](https://starlight.astro.build/) 站点。正文是 MDX，交互部分（首页的请求旅程图、学习路线、阶段进度）是 React 组件。推到 `main` 后由 `.github/workflows/deploy-docs.yml` 构建并发布到 GitHub Pages。

## 本地预览

需要 Node 22 或更新版本。

```bash
cd docs/learn
npm ci
npm run dev      # http://localhost:4321/mini-sglang/
```

其他命令：

| 命令 | 作用 |
| --- | --- |
| `npm run check` | 不构建站点，只检查 MDX 语法、源码引用、阶段 id、页面标题和站内链接，几秒钟跑完 |
| `node scripts/check-refs.mjs stages/05-scheduler.mdx` | 只检查一个页面 |
| `npm run build` | 完整构建到 `dist/`，然后检查 `dist/` 里的每一个站内链接 |
| `npm run preview` | 预览构建结果 |

## 目录

```text
src/content/docs/        页面。index.mdx 是首页，stages/ 是 11 个阶段，appendix/ 是附录
src/data/stages.mjs      学习路线的唯一来源：阶段 id、标题、用时、前置阶段
src/data/journey.mjs     首页请求旅程图的每一步
src/data/models.mjs      首页的心智模型卡片
src/components/          MDX 里用的组件；react/ 下是需要在浏览器里运行的交互组件
src/lib/source.mjs       源码引用的解析：按符号或文本找到当前行号
scripts/                 check-refs.mjs（写作时用）和 check-links.mjs（构建后用）
```

## 源码引用不会过时

页面里不写死行号。所有指向源码的链接和摘录都写「文件 + 符号」或「文件 + 一段文本」，构建时由 `src/lib/source.mjs` 找到当前的行号，链接指向构建所用 commit 的 GitHub 页面。代码改了、符号找不到时，构建直接失败，而不是悄悄发布一个过时的链接。

路径默认相对于 `python/minisgl/`；以 `/` 开头时相对于仓库根目录，例如 `/benchmark/offline/bench.py`。

符号的写法：Python 用点号连接的限定名，例如 `Scheduler.overlap_loop`、`PrefillAdder._try_allocate_one`，也可以是模块级函数或变量名；类属性和 `self.xxx =` 形式的赋值也能找到。C++ / CUDA 用函数名。

## 组件

```mdx
import { Aside, Steps } from '@astrojs/starlight/components';
import Answer from '~/components/Answer.astro';
import FileTable from '~/components/FileTable.astro';
import Mermaid from '~/components/Mermaid.astro';
import Src from '~/components/Src.astro';
import SrcCode from '~/components/SrcCode.astro';
import StageHeader from '~/components/StageHeader.astro';
import StageLink from '~/components/StageLink.astro';
import Timeline from '~/components/Timeline.astro';
```

| 组件 | 用途 | 例子 |
| --- | --- | --- |
| `StageHeader` | 阶段页的页头：用时、前置阶段、进度按钮。每个阶段页第一行 | `<StageHeader id="s5" />` |
| `FileTable` | 精读清单，行数在构建时算出来 | `<FileTable files={[{ path: 'core.py', sym: 'Req', note: '先看三个长度' }]} />` |
| `Src` | 行内源码链接 | `<Src path="scheduler/scheduler.py" sym="Scheduler._forward" />`，也可以用 `text="..."` 定位到某一行，用 `label="..."` 改显示文字 |
| `SrcCode` | 源码摘录，带真实行号和 GitHub 链接 | `<SrcCode path="core.py" sym="Req" />`，见下文 |
| `StageLink` | 链接到另一个阶段 | `<StageLink id="s6" />` 显示为「阶段 6 · Radix cache」；`<StageLink id="s6">前缀缓存那一章</StageLink>` |
| `Answer` | 自测题下面折叠的参考答案 | 见下文 |
| `Mermaid` | 流程图、时序图 | `<Mermaid caption="..." code={\`sequenceDiagram ...\`} />` |
| `Timeline` | CPU / GPU 时间线 | 见 `stages/05-scheduler.mdx` |
| `Aside` | 提示框：`note`、`tip`、`caution`、`danger` | `<Aside type="caution" title="...">...</Aside>` |
| `Steps` | 动手实验的编号步骤 | 包住一个有序列表 |

`SrcCode` 的参数：

- 只给 `sym`：摘录整个函数或类（含装饰器）。
- `from` / `to`：在符号内部（或整个文件里）按文本截取一段，`to` 所在行包含在内。`toEnd={true}` 时，如果 `to` 是一个代码块的开头，会一直截到这个块结束。
- `mark`、`ins`、`del`：高亮包含这些文本的行，可以是字符串或数组。
- 摘录超过 90 行会报错。先用 `from` / `to` 缩小范围；确实需要更长时再显式传 `maxLines`。

`Answer` 放在有序列表项里，缩进和列表项的正文对齐（`1. ` 后面是 3 个空格）：

```mdx
1. `reserved_size` 的初值是什么？

   <Answer>
   初值是 `decode_manager.inflight_tokens`……
   </Answer>
```

## 阶段章节的结构

每个阶段页按下面的顺序写，标题用原文。`stages/05-scheduler.mdx` 是样板，写之前先通读一遍。

1. `<StageHeader id="sN" />`
2. `## 这一阶段要搞懂什么`：一段话说清这一阶段在整个系统里的位置，然后 3 到 5 个读完后应该能回答的问题。
3. `## 精读清单`：一个 `FileTable`，按建议的阅读顺序排列，`note` 写读这一段时要带着的问题。
4. `## 讲解`：若干 `###` 小节。每个小节先放 `SrcCode` 摘录，再逐条解释。讲清楚「为什么这样写」，不要逐行复述代码。
5. `## 常见误区`：3 到 5 条，每条先用引号写出错误的说法，再纠正。
6. `## 自测题`：6 到 8 道，每道带 `<Answer>`。最后一两道可以是进阶题。
7. `## 动手实验`：用 `<Steps>` 包住 3 到 4 个实验，至少一个要改代码或加打印。
8. `## 过关标准`：3 条，都是「能不看代码做到某件事」。

frontmatter 的 `title` 必须和 `src/data/stages.mjs` 里的 `title` 一致，`sidebar.label` 写成 `N · 标题`，`sidebar.order` 写 N。

## 写作规则

**每一条说法都要对照当前代码核实。** 旧版的学习指南（已删除，可以在 git 历史里找到 `docs/learning-guide.zh-CN.md`）和旧的学习地图可以参考，但里面的说法不能直接照搬。行为、默认值、调用顺序都以当前代码为准。

**实测数字只用有记录的。** 所有在本机测过的数字都在 `appendix/measurements.mdx` 里，写明了硬件和配置。引用时带上条件，例如「在 RTX 4080 SUPER 上用默认参数启动，KV 池有 112,081 个 token」。没有测过的，不要写成确定的结果：实验步骤里可以让读者自己测，写「观察……」，不要替读者预告一个没测过的数字。推测要明说是推测。

**文字要求：**

- 中文写作，完整的句子。代码里的名字保持原样，用行内代码标出。
- 术语第一次出现时解释一句，之后保持同一个叫法。
- 不要用箭头链（`A → B → C`）代替句子，也不要堆砌缩写。流程可以用有序列表。
- 数字用阿拉伯数字，四位以上加千分位逗号：112,081。
- 链接到其他阶段一律用 `StageLink`，不要手写 URL。站内的其他链接用相对路径，并以 `/` 结尾，例如 `[实测记录](../../appendix/measurements/)`。

## MDX 的坑

- 正文里的 `{`、`}` 会被当成 JavaScript 表达式，`<` 后面紧跟字母会被当成组件。写到这些字符时放进行内代码，或者写成 `&#123;`、`&#125;`、`&lt;`。
- 组件的字符串属性里不能有未转义的同种引号。属性值里包含反引号或 `${` 时，用普通字符串而不是模板字符串。
- `import` 必须写在 frontmatter 之后、正文之前。没用到的组件不要导入。
- 列表项里的组件、代码块要和列表项的正文对齐缩进，前后各空一行。
- 写完先跑 `node scripts/check-refs.mjs <页面>`，再 `npm run build`。
