# 学习文档配图生成记录

生成方式：内置 image_gen 工具（用户指定 image2 绘图），两张独立图片。

内容依据：本地源码基线 `9a91cfa`，主要核对 `server/launch.py`、`scheduler/io.py`、`scheduler/scheduler.py` 与学习指南第 4、5 章。

最终文件：

- [项目架构图](mini-sglang-architecture-image2.png)：用于第 4 章，已拆成逻辑职责与进程部署两个视图，GPU 设备位于进程框外；两个文本 Manager 之间无直接消息箭头。
- [请求执行流程图](mini-sglang-execution-image2.png)：用于第 5 章，展示 Prefill、Decode 回路和完成分支。

两张图片均为 1536 × 1024 PNG，存放在项目内，网页不依赖外部图片地址。

## 项目架构图：初始提示词

```text
Use case: infographic-diagram
Asset type: A finished architecture figure for a Chinese Mini-SGLang source-code learning handbook. A single landscape raster image, high resolution, approximately 2400 × 1600, 3:2 aspect ratio.
Primary request: Draw an accurate, exceptionally readable software architecture diagram of Mini-SGLang. Flat technical publication artwork on pure white, crisp thin lines, large dark navy Chinese typography, muted teal for request/message flow, blue for GPU execution and communication, violet dashed arrows for rank broadcast. Match a documentation website with navy #12333f, teal #126e6d, mint #edf6f3. No decorative servers, 3D, gradients, mascots, fake UI, logos, watermark or decorative microtext.
Title, verbatim: "Mini-SGLang 项目架构"
Subtitle, verbatim: "在线推理 · 单机多 GPU · TP = 2 示意"
Use distinct outer containers as actual process boundaries. From left to right: client, frontend process, shared tokenizer worker, two TP scheduler processes. Preserve generous whitespace and route arrows cleanly around boxes, with no line crossing through labels.
Required topology:
1. Small external box "客户端" connects to outer box "前端进程" containing "FastAPI" and "FrontendManager". Outgoing HTTP request and returning streaming text are distinct directions.
2. One outer box "共享分词 Worker" contains TWO separate logical roles: "Tokenizer" above and "Detokenizer" below. Explicit process caption "默认 num_tokenizer = 0". Do not draw these roles as two physical processes in this default topology.
3. To their right place two matching outer process boxes side by side, labeled "Scheduler 进程 · Rank 0" and "Scheduler 进程 · Rank 1". INSIDE EACH process show "Scheduler" above "Engine", and an attached inner GPU area. Rank 0 GPU label is "GPU 0"; Rank 1 is "GPU 1". Each GPU area contains "模型分片" and "KV Cache". Within the scheduler/engine relationship include short labels "Prefill / Decode" and "page_table". Engine is part of its Scheduler process, NOT a separate physical process. KV is local to each rank, NOT a single shared global cache.
4. Directed teal message arrows, exact labels: frontend -> Tokenizer "TokenizeMsg"; Tokenizer -> Scheduler Rank 0 "UserMsg"; Scheduler Rank 0 -> Detokenizer "DetokenizeMsg"; Detokenizer -> frontend "UserReply"; frontend -> client "流式文本".
5. Dashed violet arrow from Scheduler Rank 0 to Scheduler Rank 1, labeled "ZMQ PUB / SUB" and shorter explanation "相同请求".
6. Blue bidirectional arrow between GPU 0 and GPU 1 labeled "GPU collective" with "AllReduce / AllGather". This is tensor parallel execution of the same request, not independent replicas and not a pipeline parallel arrow.
Bottom explanatory strip, large enough to read, exact text on separate lines:
"进程间请求与结果：ZMQ PUSH / PULL"
"CPU 协调：Gloo    GPU 通信：NCCL / PyNCCL"
"离线入口：LLM.generate → Scheduler / Engine（单 Rank）"
Small but legible source footer: "源码基线 9a91cfa"
Constraints: Technical accuracy and arrow endpoints are more important than decorative richness. Render all supplied Chinese and English labels correctly. No added components such as API Gateway, Redis, database, Kubernetes, vector retrieval or multi-node coordinator. Do not imply Tokenizer and Detokenizer are separate processes by default. Keep the main drawing large and legible; avoid dense paragraphs.
```

## 项目架构图：第一次定向修订（已被下方最终版替代）

```text
Use case: precise-object-edit. Image 1 is the EDIT TARGET: the Mini-SGLang architecture infographic.
Make exactly ONE technical correction: DELETE the vertical dashed teal double-headed arrow between the "Tokenizer" box and the "Detokenizer" box inside "共享分词 Worker". Those roles share a process but do not send messages directly to each other. Replace that connector with clean white background. Leave the gap empty; do NOT add a label, line, arrow or decorative replacement.
Preserve every other element: title and subtitle, all four process boundaries, client box, the two rank Scheduler/Engine/GPU structures, every other arrow, all Chinese and English text including "ZMQ PUSH / PULL", colors, fonts, line weights, spacing, image size and composition. Do not redraw the diagram from scratch or change routing elsewhere. Technical accuracy is essential.
```

## 项目架构图：区分组件、进程和设备（最终提示词）

初稿把 GPU 画在进程框内，容易混淆硬件设备与进程边界。本轮继续使用内置 image_gen 编辑原图，保留配色与字体，重新组织为两个视图。最终文件已替换为本节生成结果，并逐项核对组件归属、四个主要进程、消息方向和设备关联。

```text
Use case: precise-object-edit
Asset type: Revised Chinese technical architecture infographic for the local Mini-SGLang learning website.
Input image 1 is the EDIT TARGET. Preserve its white background, crisp navy typography, teal message arrows and blue GPU accents. Redesign its structure to CORRECT the conceptual error that placed GPU hardware inside an operating-system process. Make one finished landscape image, 3:2 aspect ratio, all text readable.

Title: "Mini-SGLang 项目架构"
Subtitle: "逻辑组件与进程部署分开看"

Organize into TWO clearly separated horizontal sections:
A. Upper section, heading "A · 逻辑组件：代码职责". Four simple compact cards, with NO process borders and NO flow arrows except the explicit Scheduler-to-Engine call described below:
1 "FrontendManager" / "请求关联与流式回复"
2 "TokenizeManager / DetokenizeManager" (may use two lines) / "文本编码与解码"
3 "Scheduler" / "选批、资源准入、请求生命周期"
4 "Engine" / "模型前向、采样、GPU 资源"
One short arrow from Scheduler card to Engine card labeled "进程内调用". Do not connect TokenizeManager and DetokenizeManager.
A visible note under this section: "组件框不代表独立进程".

B. Lower section, heading "B · 进程部署：num_tokenizer = 0，TP = 2". Draw four GREEN-tinted OS process rectangles in one row, with an external small client box to the left. Exactly these process boxes:
- "前端主进程" with internal text "FastAPI + FrontendManager"
- "共享 Worker 进程" with internal text "TokenizeManager" and "DetokenizeManager"
- "Rank 0 进程" with internal text "Scheduler + Engine"
- "Rank 1 进程" with internal text "Scheduler + Engine"
These are SINGLE process rectangles each, do not draw the internal text as additional process boxes.
Draw the main CPU message path between the process rectangles, clean separate request and response arrows:
client -> frontend: "HTTP"
frontend -> shared Worker: "TokenizeMsg"
shared Worker -> Rank 0: "UserMsg"
Rank 0 -> shared Worker: "DetokenizeMsg"
shared Worker -> frontend: "UserReply"
frontend -> client: "SSE"
Rank 0 -> Rank 1: violet dashed arrow "ZMQ PUB/SUB".
You may place response arrows on a separate lower horizontal lane to prevent overlap.

OUTSIDE and BELOW the process row, draw exactly two BLUE GPU-device rectangles:
"GPU 0" below Rank 0; "GPU 1" below Rank 1.
Both GPU rectangles contain "模型分片 · KV Cache · 页表".
Leave clear WHITE SPACE between process rectangles and GPU rectangles. No enclosing process boundary may contain a GPU.
A thin dashed association line from Rank 0 process to GPU 0 labeled "使用 cuda:0". Same from Rank 1 to GPU 1 labeled "使用 cuda:1". These are resource-use associations, not messages.
A blue double-headed arrow BETWEEN the GPU device boxes: "NCCL / PyNCCL".
Place a concise legend: "绿色框：操作系统进程    蓝色框：GPU 设备" and note "4 个主要进程；GPU 不计入进程数".
Small source footer: "源码基线 9a91cfa".

Critical invariants: Scheduler and Engine share one process PER rank. TokenizeManager and DetokenizeManager share ONE worker for num_tokenizer=0. Only Rank 0 sends DetokenizeMsg. The GPU collective must connect GPU boxes, not CPU components. GPU devices are OUTSIDE every process box. Internal function calls, interprocess messages and device-use associations must be visibly distinct. Do not add a database, Redis, gateway, distributed multi-node cluster or direct TokenizeManager–DetokenizeManager arrow. Accurate text and arrow endpoints matter more than decoration.
```

## 执行流程图：最终提示词

```text
Use case: infographic-diagram
Asset type: A finished execution-flow figure for a Chinese Mini-SGLang source-code learning handbook. A single high resolution landscape raster image, approximately 2400 × 1600, 3:2 aspect ratio.
Primary request: Draw an accurate and readable request execution flowchart for Mini-SGLang with an explicit Prefill entry, autoregressive Decode loop and termination branch. Flat technical publication artwork on pure white. Crisp thin arrow lines, large dark navy Chinese typography, muted teal #126e6d and very pale mint #edf6f3 for control flow, blue for GPU compute, a subtle amber decision diamond. Match a navy #12333f documentation reader. Generous whitespace, disciplined grid, readable labels. No decorative server illustrations, gradients, 3D, fake UI, logos or watermark.
Title, verbatim: "Mini-SGLang 请求执行流程"
Subtitle, verbatim: "单请求逻辑流程 · 普通循环视角"
Draw exactly this graph, with unambiguous directed edges. Use a clean multi-row arrangement; route the Decode return around the outside so it never passes through another node. Number the six main initial stages with small 01–06 markers:
01 "接收请求" — sublabels "POST /v1/chat/completions" and "Frontend：uid / 回复队列"
  -> 02 "文本预处理" — sublabels "Tokenizer：模板 + 编码" and "TokenizeMsg → UserMsg"
  -> 03 "等待与资源准入" — sublabels "PendingReq / 前缀匹配" and "请求槽 / KV 页"
  -> 04 "Prefill" — sublabels "Engine.forward_batch" and "计算 prompt 后缀，可分块"
  -> 05 "采样与写回" — sublabels "生成有效 token" and "写入 GPU token_pool"
  -> 06 "CPU 结果处理" — sublabels "等待 token 回拷" and "追加请求历史"
  -> decision diamond "结束？" — sublabel "EOS / 长度上限".
From the decision draw TWO clearly labeled branches:
- branch "否" -> box "发送文本增量" with sublabel "反分词 → Frontend → Client" -> blue compute box "Decode" with sublabel "每请求输入 1 个新 token" -> an explicit long return arrow to stage 05 "采样与写回". Label the return area "自回归循环". Decode includes Engine forward; it reuses existing KV Cache. The loop must return to stage 05, not to HTTP ingestion, tokenization or Prefill.
- branch "是" -> box "完成与资源处理" with sublabels "释放请求槽" and "缓存前缀 / 释放可回收资源" -> box "最终增量与完成标记" -> clear terminal "结束".
Include a short footer explaining the result message chain, verbatim:
"输出链路：DetokenizeMsg → Detokenizer → UserReply → Frontend → Client"
Include one readable note, verbatim:
"中间 Prefill 分块不输出有效 token；Overlap 可重叠下一批执行与上一批结果处理。"
Include small legible source footer: "源码基线 9a91cfa"
Technical constraints: This is a conceptual ordinary-loop view, not exact CPU/GPU wall-clock timing. Chunked Prefill may repeat inside stage 04 until its final chunk; only that final prefill output is sent to the client. Stop-condition checking and resource cleanup precede sending the finished result in this implementation. No claim that all KV is immediately erased on completion. No prefill/decode mixed batch claim. Use the exact graph and labels supplied; do not invent extra functional stages or branches.
```
