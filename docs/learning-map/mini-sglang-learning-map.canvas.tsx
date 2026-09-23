import {
  Button,
  Callout,
  Code,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Row,
  Select,
  Stack,
  Table,
  Text,
  TodoListCard,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
  useState,
} from "cursor/canvas";
import type { TodoItem, TodoStatus } from "cursor/canvas";

const PKG = "/root/mini-sglang/python/minisgl/";
const REPO = "/root/mini-sglang/";
const PR = "https://github.com/sgl-project/mini-sglang/pull/";
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

// ---------------------------------------------------------------------------
// Data: request journey
// ---------------------------------------------------------------------------

type Step = {
  lane: number;
  title: string;
  box: string;
  fn: string;
  files: string[];
  detail: string;
  edge?: string;
};

const LANES = [
  { name: "API Server", sub: "主进程 · FastAPI / shell" },
  { name: "Tokenizer", sub: "独立进程 · 分词 + 反分词" },
  { name: "Scheduler", sub: "每卡一个进程 · CPU 侧" },
  { name: "Engine", sub: "同一进程 · GPU 侧" },
];

const STEPS: Step[] = [
  {
    lane: 0,
    title: "接收请求",
    box: "v1_completions",
    fn: "v1_completions()  /  shell()",
    files: ["server/api_server.py:256", "server/api_server.py:351"],
    detail:
      "为请求分配一个 uid，把 messages 和采样参数打包成 `TokenizeMsg` 发给 tokenizer 进程。`FrontendManager` 为每个 uid 建一个 `asyncio.Event`，之后就在它上面等回包。",
    edge: "ZMQ · TokenizeMsg",
  },
  {
    lane: 1,
    title: "分词",
    box: "tokenize_worker",
    fn: "tokenize_worker()  →  TokenizeManager.tokenize()",
    files: ["tokenizer/server.py:31", "tokenizer/tokenize.py"],
    detail:
      "套用 chat template 后编码成一维 int32 的 CPU tensor，包成 `UserMsg` 发给 scheduler。默认 `num_tokenizer=0`，分词和反分词在同一个进程里完成。",
    edge: "ZMQ · UserMsg",
  },
  {
    lane: 2,
    title: "入队",
    box: "_process_one_msg",
    fn: "receive_msg()  →  Scheduler._process_one_msg()",
    files: ["scheduler/scheduler.py:169", "scheduler/io.py"],
    detail:
      "rank 0 从 ZMQ 收消息，多卡时再转发给其他 rank。prompt 超过 `max_seq_len` 的请求直接丢弃，`max_tokens` 被截到 `max_seq_len − input_len`，然后变成 `PendingReq` 进入 prefill 等待队列。",
  },
  {
    lane: 2,
    title: "选下一批",
    box: "_schedule_next_batch",
    fn: "Scheduler._schedule_next_batch()",
    files: [
      "scheduler/scheduler.py:219",
      "scheduler/prefill.py:92",
      "scheduler/decode.py:32",
      "scheduler/cache.py:27",
    ],
    detail:
      "prefill 优先。`PrefillAdder` 先在 radix 树里匹配前缀（只匹配 `input_ids[:-1]`），再估算「本次要算的 token + `max_tokens`」能否放进「空闲 + 可驱逐」的 KV 空间；放得下就锁住前缀、分配 table 行，超出 token 预算的部分变成 `ChunkedReq`。没有 prefill 可做时才跑 decode：`DecodeManager` 把所有在跑的请求按 uid 排序组成一批。",
  },
  {
    lane: 2,
    title: "准备 batch",
    box: "_prepare_batch",
    fn: "Scheduler._prepare_batch()",
    files: [
      "scheduler/scheduler.py:204",
      "engine/graph.py:160",
      "scheduler/cache.py:42",
      "attention/fi.py:190",
    ],
    detail:
      "先用 dummy 请求把 batch 补到已录制的 CUDA graph 尺寸；按页给新 token 分配 KV 槽位并写进 `page_table`；算出 positions 和 `out_loc`；让 attention 后端准备 metadata；把采样参数搬上 GPU。overlap 模式下这些都在 scheduler 自己的 CUDA stream 上完成，不会等 GPU 上正在跑的那一批。",
    edge: "发射到 engine stream",
  },
  {
    lane: 3,
    title: "前向",
    box: "forward_batch",
    fn: "Engine.forward_batch()",
    files: ["engine/engine.py:191", "engine/graph.py:78", "core.py:101"],
    detail:
      "decode 且 batch size 不超过上限时 replay CUDA graph，否则直接调 `model.forward()`。batch 通过全局 `Context` 交给模型，所以 forward 不需要参数。算完后对每个请求调用 `complete_one()` 推进长度，再去采样。",
  },
  {
    lane: 3,
    title: "逐层计算",
    box: "Qwen3.forward",
    fn: "Qwen3ForCausalLM.forward()",
    files: ["models/qwen3.py:66", "models/utils.py", "layers/linear.py"],
    detail:
      "embedding → 28 个 decoder 层（RMSNorm → QKV 投影 → attention → O 投影 → RMSNorm → MLP）→ LM head。prefill 时只对每个请求的最后一个 token 算 logits。",
  },
  {
    lane: 3,
    title: "Attention",
    box: "AttentionLayer",
    fn: "AttentionLayer.forward()  →  attn_backend.forward()",
    files: ["layers/attention.py:47", "attention/fi.py:176", "kvcache/mha_pool.py:45", "kernel/csrc/jit/store.cu"],
    detail:
      "先对 q、k 做 RMSNorm 和 RoPE，再交给后端：后端用 `store_kv` 按 `out_loc` 把新 token 的 K/V 写进显存池（一个 warp 搬一个 token），然后调 attention kernel，通过 `page_table` 读出全部历史 KV。",
  },
  {
    lane: 3,
    title: "采样",
    box: "Sampler.sample",
    fn: "Sampler.sample()  →  token_pool[write_tuple] = next_tokens",
    files: ["engine/sample.py:71", "scheduler/scheduler.py:227"],
    detail:
      "全是贪心就 argmax，否则走 FlashInfer 的 top-k / top-p 采样。新 token 直接写进 GPU 上的 `token_pool`，下一批 decode 从那里读输入；同时异步拷一份回 CPU。",
    edge: "next token 异步拷回 CPU",
  },
  {
    lane: 2,
    title: "收尾（下一轮）",
    box: "_process_last_data",
    fn: "Scheduler._process_last_data()",
    files: ["scheduler/scheduler.py:138", "scheduler/cache.py:55"],
    detail:
      "overlap 模式下这一步发生在下一批已经发射之后。等拷贝事件完成，把 token 追加到 CPU 侧的 `input_ids`，判断 EOS 或长度上限；结束的请求释放 table 行，KV 插回 radix 树，并记进 `finished_reqs` 防止重复释放。",
    edge: "ZMQ · DetokenizeMsg",
  },
  {
    lane: 1,
    title: "增量反分词",
    box: "detokenize",
    fn: "DetokenizeManager.detokenize()",
    files: ["tokenizer/detokenize.py:70"],
    detail:
      "只解码新增的 token。如果结果以 � 结尾，说明一个 UTF-8 字符还没拼完，先扣住不发；中文流式输出不乱码靠的就是这里。",
    edge: "ZMQ · UserReply",
  },
  {
    lane: 0,
    title: "流式返回",
    box: "wait_for_ack",
    fn: "FrontendManager.listen()  →  wait_for_ack()",
    files: ["server/api_server.py:116", "server/api_server.py:134"],
    detail:
      "后台协程收到 `UserReply` 后按 uid 唤醒对应的等待者，以 SSE 推给客户端。客户端断开时发出 `AbortMsg`，一路传到 scheduler 释放资源。",
  },
];

// ---------------------------------------------------------------------------
// Data: mental models
// ---------------------------------------------------------------------------

const MODELS = [
  {
    title: "一个请求就是三个长度",
    body:
      "`cached_len` 是 KV 已经算好的 token 数，`device_len` 是 token id 已经确定的 token 数，`max_device_len` 是 prompt 长度加 `max_tokens`。每轮要算 `extend_len = device_len − cached_len` 个 token。prefill、chunked prefill、decode 用的是同一套公式，decode 只是 `extend_len` 等于 1 的特例。",
    files: ["core.py:29"],
  },
  {
    title: "两张二维表",
    body:
      "`token_pool[table_idx, pos]` 存 token id，`page_table[table_idx, pos]` 存这个 token 的 KV 在显存池里的槽位，每个在跑的请求占一行。调度器只维护这两张表；模型看到的只是从表里查出来的 `input_ids`、`positions` 和 `out_loc`，根本不知道「请求」是什么。",
    files: ["scheduler/table.py", "scheduler/scheduler.py:252"],
  },
  {
    title: "显存和归属分离",
    body:
      "`MHAKVCache` 是一整块张量，自己不记录哪个槽位属于谁。归属由 `CacheManager` 的空闲页列表和 radix 树管理：树里存「token 前缀 → 槽位」，用引用计数防止正在用的前缀被驱逐，空间不够时按访问时间淘汰叶子。",
    files: ["kvcache/mha_pool.py:10", "kvcache/radix_cache.py:101", "scheduler/cache.py"],
  },
  {
    title: "全局 Context 代替传参",
    body:
      "`model.forward()` 没有参数，每一层通过 `get_global_ctx().batch` 取 positions、attention metadata 和 KV 池。输入都放在地址固定的 buffer 里，decode 才能录成 CUDA graph 反复 replay；换一个 attention 后端也只需要实现 5 个方法。",
    files: ["core.py:101", "engine/graph.py:78", "attention/base.py"],
  },
  {
    title: "Overlap：CPU 永远领先 GPU 一拍",
    body:
      "GPU 在跑第 N 批时，CPU 处理第 N−1 批的结果，接着就去调度第 N+1 批。所以采样出的 token 先写进 GPU 上的 `token_pool`，下一批直接从那里读。一个请求可能在第 N 批已经结束、却已经被排进了第 N+1 批，代码用 `finished_reqs` 防止重复释放。",
    files: ["scheduler/scheduler.py:83"],
  },
];

// ---------------------------------------------------------------------------
// Data: learning stages
// ---------------------------------------------------------------------------

type QA = { q: string; a: string };
type Lab = { text: string; code?: string };
type Stage = {
  id: string;
  name: string;
  time: string;
  goal: string;
  reading?: string[];
  files: string[];
  questions: QA[];
  labs: Lab[];
};

const STAGES: Stage[] = [
  {
    id: "s0",
    name: "预备知识",
    time: "按需 1–3 天",
    goal:
      "搞清楚 LLM 推理服务到底在解决什么问题：KV cache 为什么要分页、prefill 和 decode 有什么不同、为什么要拼命组批。已经熟悉的话直接跳到阶段 1。",
    reading: [
      "[PagedAttention（vLLM 论文）](https://arxiv.org/abs/2309.06180)：KV cache 分页管理的来源",
      "[SGLang 论文](https://arxiv.org/abs/2312.07104) 和 [RadixAttention 博客](https://lmsys.org/blog/2024-01-17-sglang/)：用 radix 树复用前缀",
      "[SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)：零开销的 overlap 调度",
      "[Sarathi-Serve](https://arxiv.org/abs/2403.02310)：chunked prefill",
      "[NanoFlow](https://arxiv.org/abs/2408.12757)：CPU 和 GPU 重叠执行",
    ],
    files: ["@docs/structures.md|项目自带的结构说明", "@docs/features.md|项目自带的特性说明"],
    questions: [
      {
        q: "Qwen3-0.6B 有 28 层、8 个 KV 头、`head_dim` 为 128，用 bf16 存。每个 token 的 KV cache 占多少字节？",
        a: "2（K 和 V）× 28 × 8 × 128 × 2 字节 = 114,688 字节，也就是 112 KiB。阶段 4 会在启动日志里对上这个数。",
      },
      {
        q: "它有 16 个 q 头、只有 8 个 KV 头，attention 时怎么对应？",
        a: "GQA：每 2 个 q 头共用 1 组 K/V。KV cache 只存 8 个头，显存比 16 个头的 MHA 省一半。",
      },
      {
        q: "decode 每步只算一个 token，为什么说它卡在访存上，而不是算力上？",
        a: "每一步都要把全部权重和整段历史 KV 从显存读一遍，计算量却很小。batch 越大，读一次权重能服务的请求就越多，这就是调度器拼命组批的原因。",
      },
    ],
    labs: [
      {
        text: "用 HF transformers 手写一个带 KV cache 的贪心生成：prompt 先整段过一次模型（prefill），之后每步只喂上一步的 token 并传入 `past_key_values`（decode）。30 行以内，写完你就知道 mini-sglang 在优化什么。",
      },
    ],
  },
  {
    id: "s1",
    name: "跑起来，单步调试",
    time: "半天",
    goal:
      "在单进程里把一个请求从头跟到尾。`LLM` 类直接继承 `Scheduler`，没有 ZMQ、没有子进程，普通调试器就能打断点。",
    files: ["llm/llm.py|98 行", "@benchmark/offline/bench.py", "env.py|87 行"],
    questions: [
      {
        q: "`LLM` 只重写了两个方法，为什么就能绕开 tokenizer 进程和 ZMQ？",
        a: "`offline_mode=True` 时 IO 层不建任何 socket，收发消息直接走 `offline_receive_msg` 和 `offline_send_result`。`LLM` 在这两个方法里自己分词、收集输出，调度器和引擎的代码一行没改。",
      },
      {
        q: "`generate()` 调用的 `run_forever()` 是个死循环，它是怎么返回的？",
        a: "所有请求都处理完后，调度器以阻塞模式调用 `offline_receive_msg`，这时待处理列表是空的，它就抛出 `RequestAllFinished`，`generate()` 捕获这个异常后整理结果返回。",
      },
    ],
    labs: [
      {
        text: '下面是在你机器上验证过的最小脚本。必须显式写 `attention_backend="fi"`：默认的 `auto` 会在 RTX 50 系上选中 trtllm，然后崩溃。',
        code: `from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", attention_backend="fi",
          max_seq_len_override=4096, cuda_graph_max_bs=8, memory_ratio=0.5)
out = llm.generate(["The capital of France is"],
                   SamplingParams(temperature=0.0, max_tokens=16))
print(out[0]["text"])
llm.shutdown()`,
      },
      {
        text: "调试前先设环境变量 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`。主循环会变成「调度 → 前向 → 处理结果」的顺序执行，断点不会乱跳。",
      },
      {
        text: "断点打在 `_schedule_next_batch`、`_prepare_batch`、`Engine.forward_batch`、`_process_last_data`，看同一个请求先走 1 次 prefill，再走 15 次 decode。",
      },
    ],
  },
  {
    id: "s2",
    name: "核心数据结构",
    time: "半天",
    goal: "吃透 `Req`、`Batch`、`Context` 和两张二维表。后面所有模块都在读写这几样东西。",
    files: ["core.py|136 行", "scheduler/table.py|21 行", "scheduler/utils.py|33 行"],
    questions: [
      {
        q: "prompt 有 5 个 token、没有缓存命中、`max_tokens=3`。prefill 前后和每步 decode 后，`cached_len` / `device_len` 各是多少？",
        a: "prefill 前 0 / 5（`extend_len` = 5）；prefill 后 5 / 6；第 1 步 decode 后 6 / 7；第 2 步后 7 / 8。这时 `device_len` 等于 `max_device_len`（5 + 3），`can_decode` 变成 False，请求结束。3 个输出 token 里 1 个来自 prefill，2 个来自 decode。",
      },
      {
        q: "`Req` 为什么断言 `cached_len < device_len`，连相等都不允许？",
        a: "每轮至少要算一个 token，才能拿到预测下一个 token 的 logits。所以前缀匹配只匹配 `input_ids[:-1]`：就算整个 prompt 都在缓存里，最后一个 token 也要重算。",
      },
      {
        q: "为什么 `Context` 是全局单例，`model.forward()` 一个参数都没有？",
        a: "每层需要的东西（positions、attention metadata、KV 池、后端）都挂在 Context 上，省掉层层传参。录 CUDA graph 时只要把 batch 的字段指向固定的 buffer，模型代码一行都不用改。`Context.forward_batch` 还保证同一时刻只有一个 batch 在跑。",
      },
    ],
    labs: [
      {
        text: "先在纸上推出第 1 题的答案，再在 `Engine.forward_batch` 里 `req.complete_one()` 的前后各打印一次 `req`（它的 `__repr__` 会打出 `table_idx` 和三个长度），对照你的推导。",
      },
    ],
  },
  {
    id: "s3",
    name: "模型前向",
    time: "1–2 天",
    goal: "看懂一个 decoder 层从加载权重到前向计算的每一行，并能和 HF transformers 的实现一一对应。",
    files: [
      "models/qwen3.py|83 行",
      "models/utils.py|126 行",
      "layers/base.py|99 行",
      "layers/linear.py|127 行",
      "layers/norm.py|38 行",
      "layers/rotary.py|145 行",
      "layers/embedding.py|110 行",
      "models/weight.py|124 行",
    ],
    questions: [
      {
        q: "`BaseOP` 没有继承 `nn.Module`，`state_dict()` 是怎么找到所有参数的？",
        a: "遍历对象的 `__dict__`：张量就是参数，`BaseOP` 就递归进去，下划线开头的属性直接跳过（缓存和通信句柄都这样藏起来）。加载时按同样的规则把张量逐个换进去。",
      },
      {
        q: "模型为什么先建在 meta device 上？RoPE 为什么是例外？",
        a: "meta 张量只有形状、不占显存，建模型时不分配内存，加载权重时整块替换。RoPE 的 cos/sin 表不在 checkpoint 里，要现场算出来，所以必须建在真实设备上（`set_rope_device`）。",
      },
      {
        q: "`RMSNormFused.forward` 返回两个值，残差加法藏在哪？",
        a: "FlashInfer 的 `fused_add_rmsnorm` 原地做 `residual += x`，再对 residual 做归一化写回 x。一个 kernel 同时完成残差相加和归一化，少读写一次显存。",
      },
      {
        q: "prefill 时 LM head 为什么只算每个请求最后一个 token？",
        a: "只有最后一个位置的 logits 用来采样下一个 token，其余位置算了也白算。词表有 15 万维，`get_last_indices` 先挑出每个请求最后一个 token 的下标，能省下大量计算。",
      },
      {
        q: "checkpoint 里 q、k、v 是三个权重，模型里为什么只有一个 `qkv_proj`？",
        a: "`load_weight` 边读边把 q、k、v 按行拼成一个矩阵（gate 和 up 同理），前向时一次 GEMM 算完三个投影。",
      },
    ],
    labs: [
      {
        text: '用 `cache_type="naive"` 让 mini-sglang 贪心生成 16 个 token，再用 HF transformers 跑同一个 prompt，对比输出。bf16 下后面可能分叉，但开头应该一致。',
        code: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda()
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=16, do_sample=False)
print(repr(tok.decode(out[0, ids.shape[1]:])))`,
      },
      { text: "读 `models/qwen3.py` 时把 HF 的 `modeling_qwen3.py` 开在旁边，逐行对应。" },
    ],
  },
  {
    id: "s4",
    name: "显存与 KV 池",
    time: "1 天",
    goal: "搞清楚显存怎么分给模型权重和 KV 池，以及一个 token 的 KV 最终落在哪个地址。",
    files: [
      "engine/engine.py:148|_determine_num_pages",
      "kvcache/mha_pool.py|68 行",
      "kernel/store.py|42 行",
      "kernel/csrc/jit/store.cu|123 行",
      "scheduler/cache.py:42|allocate_paged",
    ],
    questions: [
      {
        q: "`MHAKVCache._kv_buffer` 的 6 个维度分别是什么？",
        a: "`[2（K/V）, 层数, 页数, page_size, 本卡 KV 头数, head_dim]`。写入时把「页数 × 页大小」这两维展平成 token 槽位来寻址，所以 `out_loc` 就是展平后的下标。",
      },
      {
        q: "`page_table` 存的是页号还是 token 槽位？`page_size > 1` 时 `free_slots` 里存什么？",
        a: "`page_table` 永远按 token 粒度存槽位。`page_size > 1` 时，`free_slots` 存的是每个空闲页的首槽位（页号 × 页大小），分配时再展开成连续的 token 槽位。",
      },
      {
        q: "`out_loc` 是怎么来的？",
        a: "`_prepare_batch` 先按页分配槽位、写进 `page_table`，再用每个新 token 所在的行和位置去查：`page_table[table_idx, positions]`。",
      },
      {
        q: "`memory_ratio` 控制什么？KV 的页数是怎么算出来的？",
        a: "它是「模型 + KV cache」能占用的空闲显存比例。先记下加载模型前的空闲显存，加载后相减得到模型大小；`memory_ratio × 初始空闲 − 模型大小` 就是 KV 的预算，再除以每页的字节数。",
      },
    ],
    labs: [
      {
        text: "算账：2 × 28 × 8 × 128 × 2 字节 = 112 KiB/token。你的机器上 `memory_ratio=0.5` 时日志显示 56,909 个 token、6.08 GiB，112 KiB × 56,909 正好是 6.08 GiB。",
      },
      { text: "把 `memory_ratio` 分别改成 0.3 和 0.7，先用公式预测 token 数，再看日志验证。" },
    ],
  },
  {
    id: "s5",
    name: "调度器",
    time: "2 天",
    goal:
      "整个项目的心脏。搞清楚准入控制、chunked prefill、prefill 优先的策略，以及 overlap 模式下 CPU 和 GPU 怎么错开一拍。",
    files: [
      "scheduler/prefill.py|162 行",
      "scheduler/decode.py|39 行",
      "scheduler/scheduler.py|267 行",
      "scheduler/scheduler.py:108|normal_loop",
      "scheduler/scheduler.py:83|overlap_loop",
    ],
    questions: [
      {
        q: "准入时 `PrefillAdder` 为什么要加上 `reserved_size`？",
        a: "它的初值是所有在跑 decode 请求还要生成的 token 数，每接纳一个新请求再加上它的剩余长度。这是在给每个请求预留到 `max_tokens` 的空间，保证已经接纳的请求一定能跑完。",
      },
      {
        q: "mini-sglang 没有抢占。显存紧张时会发生什么？",
        a: "新请求在准入阶段就被挡住，留在队列里等别的请求结束；已经在跑的请求不会被踢掉。代价是按 `max_tokens` 保守预留，显存利用率偏低。完整版 SGLang 用 retract（抢占）解决这个问题。",
      },
      {
        q: "`ChunkedReq` 为什么不能采样？",
        a: "它只算了 prompt 的一部分，最后一个位置的 logits 并不预测真正的下一个 token。它的 `can_decode` 恒为 False，`_process_last_data` 直接跳过它；下一轮用更新后的 `cached_len` 切下一块。",
      },
      {
        q: "同一个 batch 里会同时有 prefill 和 decode 请求吗？",
        a: "不会。`_schedule_next_batch` 先尝试 prefill，有就整批都是 prefill，没有才跑 decode。长 prompt 进来时，所有 decode 请求都要等这一批 prefill 算完。",
      },
      {
        q: "overlap 模式下，一个请求在第 N 批已经结束，却已经被排进了第 N+1 批，代码怎么处理？",
        a: "CPU 处理第 N 批结果时才知道它结束了，这时第 N+1 批已经发射。处理第 N 批时释放它的资源并记进 `finished_reqs`；处理第 N+1 批时发现它在 `finished_reqs` 里，就跳过释放。多算的那一步结果直接丢弃。",
      },
      {
        q: "进阶：你能找出一处疑似竞争吗？上一题里多算的那一步，还会往它原来那行 `token_pool` 写一个 token，而这一行可能已经分给了新请求。两次写入分别在哪个 CUDA stream 上？中间有同步吗？",
        a: "这是我读代码时发现的疑点，没有实测过。多出来的写入在 engine stream 上（`_forward` 里的 `token_pool[write_tuple] = ...`），新请求的 prompt 拷贝在 scheduler stream 上（`_add_one_req` 里的非阻塞 `copy_`），`TableManager` 又是后进先出地复用行号，我没找到两者之间的同步。你可以自己设计实验：让新请求的 prompt 比刚结束的请求更长，对比 overlap 开和关时的输出。",
      },
    ],
    labs: [
      {
        text: "用 `LLM(..., max_extend_tokens=16)` 跑一个 100 个 token 左右的 prompt，在 `PrefillManager.schedule_next_batch` 里打印每批的请求类型和 `extend_len`，看它被切成几块。",
      },
      {
        text: "先读 `normal_loop` 再读 `overlap_loop`，在纸上画 CPU 和 GPU 两条时间线，标出第 N 批的发射、第 N 批结果的处理、第 N+1 批的调度分别发生在什么时候。",
      },
    ],
  },
  {
    id: "s6",
    name: "Radix cache",
    time: "1–2 天",
    goal: "理解前缀树怎么复用 KV：匹配、加锁、插入、分裂、驱逐，以及请求结束时哪些槽位必须还回去。",
    files: [
      "kvcache/radix_cache.py|237 行",
      "kvcache/base.py|135 行",
      "scheduler/cache.py:55|cache_req",
      "kernel/csrc/src/radix.cpp|44 行",
      "@tests/core/test_cache_allocate.py",
    ],
    questions: [
      {
        q: "插入前缀时为什么要按 `page_size` 向下对齐？",
        a: "KV 按页分配和驱逐，树里的每个节点必须是整页。不满一页的尾巴不进树，请求结束时直接释放。",
      },
      {
        q: "`ref_count` 什么时候加、什么时候减？为什么只有 `ref_count` 为 0 的叶子能被驱逐？",
        a: "请求准入时锁住它匹配到的节点（从该节点一路到根都加 1），prefill 后解锁旧 handle、锁住新 handle，请求结束时解锁。`ref_count` 大于 0 表示有请求正在读这段 KV；只驱逐叶子，是因为内部节点是别的分支的公共前缀。",
      },
      {
        q: "什么时候会 `split_at`？",
        a: "新序列和某个节点只匹配了前半段时，把节点在分叉处切成父子两段：父节点是公共前缀，子节点是原来的后半段，新序列的剩余部分再挂成子节点的兄弟。",
      },
      {
        q: "`cache_req` 的注释列了 5 段区间，哪一段不释放会泄漏显存？",
        a: "`[old_handle.cached_len, cached_len)`：这段 KV 是本请求自己算的，但在它算完之前，别的请求已经把同样的前缀插进了树。树里保留别人那份，本请求这份必须释放，否则这些槽位永远回不来。",
      },
      {
        q: "split 出来的父节点为什么要刷新时间戳？（提示：commit ba3b55c）",
        a: "驱逐按时间戳从旧到新淘汰叶子。父节点刚被访问过，如果沿用旧时间戳，等它的子节点被驱逐、它自己变成叶子时，就会被过早淘汰。",
      },
    ],
    labs: [
      {
        text: "复现实测：同一个 prompt 贪心生成两次后，打印 `llm.cache_manager.prefix_cache.size_info`。你的机器上是 `evictable_size = 30`：每次插入 20 个 token（prompt 5 个加输出的前 15 个），两次共享前 10 个后分叉，10 + 10 + 10 = 30。先自己推出这个数再跑。",
      },
      {
        text: '确定性实验：radix 模式下第二次的贪心输出变了，换成 `cache_type="naive"` 三次完全一致。解释为什么「数学上等价」的前缀复用会改变贪心结果。（提示：第二次 prefill 只算了 1 个 token，矩阵形状变了，bf16 的累加顺序也跟着变。）',
      },
      {
        text: "照着 `tests/core/test_cache_allocate.py` 的写法，写一个不需要 GPU 的测试，覆盖 insert → match → split → evict。当前 venv 里没装 pytest，要先装上。",
      },
    ],
  },
  {
    id: "s7",
    name: "Attention 后端与 CUDA graph",
    time: "1–2 天",
    goal: "理解后端接口的 5 个方法、FlashInfer 先 plan 再 run 的两阶段，以及 CUDA graph 的录制和 replay。",
    files: [
      "attention/base.py|63 行",
      "attention/fi.py|271 行",
      "attention/__init__.py|77 行",
      "engine/graph.py|171 行",
      "attention/fa.py|182 行",
      "attention/trtllm.py|162 行",
      "engine/engine.py:218|_adjust_config",
    ],
    questions: [
      {
        q: "为什么只有 decode 用 CUDA graph？",
        a: "CUDA graph 要求每次 replay 的形状和地址都不变。decode 每个请求只有 1 个 token，形状只取决于 batch size，可以按 1、2、4、8… 预先录好；prefill 的长度千变万化，没法预录。",
      },
      {
        q: "`pad_batch` 补进来的 dummy 请求会不会写坏别人的 KV？",
        a: "不会。dummy 请求用 `page_table` 的最后一行（`table_idx = max_running_req`），这一行全部指向额外多分配的一页。它的写入只落在这页上，结果直接丢弃。",
      },
      {
        q: "`prepare_for_replay` 为什么只能 `copy_` 数据，不能换一个新张量？",
        a: "录制时 kernel 记住的是当时张量的显存地址，replay 只会读同一块地址。换张量等于 kernel 读不到新数据，所以只能把新值拷进录制时的 buffer。",
      },
      {
        q: "FlashInfer 的 `plan()` 之前为什么要 `last_event.synchronize()`？（提示：commit 20fcd7f）",
        a: "`plan()` 会复用一块 pinned 的 host 缓冲区，并发起异步的 H2D 拷贝。overlap 模式下，下一批的 `plan()` 可能在上一次拷贝完成之前就改写这块缓冲区，GPU 就会读到改了一半的 metadata。",
      },
      {
        q: "`HybridBackend` 怎么让 prefill 和 decode 用不同的 kernel？",
        a: "每次调用都按 `batch.is_prefill` 分派给两个后端之一。CUDA graph 只录 decode，所以录制相关的方法只转发给 decode 后端。",
      },
    ],
    labs: [
      {
        text: "修一个你亲身踩过的 bug：`_adjust_config` 用 `is_sm100_supported()`（判断的是算力 ≥ 10.0）来选 trtllm，于是算力 12.0 的 RTX 50 系也被选中，一提问就崩。我在你的卡上也试了 `fa,fi`：它同样按 ≥ 10.0 选了 FA4，直接报 cutlass 的 AttributeError。所以 12.x 应该落到 `fi`。改完后用默认的 `--attn auto` 验证。",
      },
      { text: "用 `cuda_graph_max_bs=0` 关掉 CUDA graph，对比 decode 每个 token 的耗时。" },
    ],
  },
  {
    id: "s8",
    name: "多进程服务化",
    time: "1 天",
    goal: "理解 API server、tokenizer、scheduler 三类进程怎么启动、怎么通信，以及取消请求的信号怎么传递。",
    files: [
      "server/launch.py|119 行",
      "server/api_server.py|452 行",
      "tokenizer/server.py|110 行",
      "tokenizer/detokenize.py|111 行",
      "message/utils.py|69 行",
      "utils/mp.py|151 行",
    ],
    questions: [
      {
        q: "默认参数下一共有几个进程？启动时要等几个 ack？",
        a: "3 个进程：主进程（API server 或 shell）、1 个 scheduler（单卡）、1 个同时负责分词和反分词的 tokenizer 进程。要等 `num_tokenizer + 2` = 2 个 ack：scheduler 只有 rank 0 回 ack，再加上 detokenizer。",
      },
      {
        q: "`num_tokenizer=0` 时，分词和反分词怎么会在同一个进程里？",
        a: "`share_tokenizer` 为真时，`zmq_tokenizer_addr` 直接返回 detokenizer 的地址，API server 和 scheduler 往同一个 socket 发消息。那个进程按类型分拣：`TokenizeMsg` 去分词，`DetokenizeMsg` 去反分词，`AbortMsg` 转成 `AbortBackendMsg` 发给 scheduler。",
      },
      {
        q: "消息怎么序列化？为什么只支持一维 tensor？",
        a: "dataclass 递归转成带 `__type__` 字段的 dict，tensor 转成字节串加 dtype 字符串，再用 msgpack 打包。没有保存 shape，反序列化时只能按一维还原；token id 序列恰好是一维的。",
      },
      {
        q: "客户端断开后，取消信号经过哪几跳到达 scheduler？",
        a: "`stream_with_cancellation` 发现断开 → `abort_user` 发出 `AbortMsg` → tokenizer 进程转成 `AbortBackendMsg` → scheduler 从 prefill 或 decode 队列里摘掉请求，释放 table 行和 KV。",
      },
      {
        q: "detokenizer 为什么要维护 `read_offset`、`surr_offset`、`sent_offset` 三个位置？",
        a: "单独解码一个 token 可能得到半个 UTF-8 字符或错误的前导空格。它每次把「已确认文本末尾的几个 token + 新 token」一起解码，减去已确认的部分，只输出新增的字符；结果以 � 结尾就先扣住，等下一个 token 补齐。`sent_offset` 记录已经发出去的字符串长度。",
      },
    ],
    labs: [
      {
        text: "起服务后用 curl 发一个流式请求，让模型用中文回答。在 `DetokenizeManager.detokenize` 里打印每次的增量输出，观察 � 被扣住又补齐的时刻。",
        code: `python -m minisgl --model Qwen/Qwen3-0.6B --attn fi

curl -N http://127.0.0.1:1919/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model": "Qwen/Qwen3-0.6B", "stream": true, "max_tokens": 64,
       "messages": [{"role": "user", "content": "用一句话介绍北京"}]}'`,
      },
    ],
  },
  {
    id: "s9",
    name: "张量并行",
    time: "1–2 天",
    goal: "理解权重怎么切、每层在哪里通信、多个 rank 怎么保持步调完全一致。",
    files: [
      "layers/linear.py|127 行",
      "layers/embedding.py|110 行",
      "models/weight.py:34|_shard_tensor",
      "distributed/impl.py|97 行",
      "scheduler/io.py|133 行",
      "kernel/pynccl.py|78 行",
    ],
    questions: [
      {
        q: "`qkv_proj`、`gate_up_proj` 为什么不需要通信，`o_proj`、`down_proj` 为什么要 all-reduce？",
        a: "前两个按输出维切（列并行），每张卡算出一部分头或一部分中间维，直接接着算本卡的 attention 或激活；后两个按输入维切（行并行），每张卡只得到部分和，必须 all-reduce 求和才是完整结果。所以每层只在 attention 末尾和 MLP 末尾各通信一次。",
      },
      {
        q: "KV 头数少于卡数时（比如 4 张卡、2 个 KV 头）怎么切？",
        a: "`div_even(..., allow_replicate=True)` 让每张卡保留 1 个 KV 头，相邻两张卡共用同一个头的权重。`_shard_tensor` 用 `r * num_kv_heads // n` 算出第 r 张卡拿第几个头。",
      },
      {
        q: "rank 0 怎么保证所有 rank 收到完全相同的消息序列？",
        a: "只有 rank 0 连 tokenizer。它把收到的原始字节经 ZMQ PUB 转发，同时用 gloo 广播本轮消息条数，其他 rank 按这个条数从 SUB 读，保证每一轮处理的消息一模一样。",
      },
      {
        q: "decode batch 为什么要按 uid 排序？（提示：commit 9a91cfa，只改了一行）",
        a: "`running_reqs` 是一个 set，`Req` 按对象身份哈希，不同进程里的遍历顺序不一样。顺序不一致时，各卡 batch 的第 i 行是不同的请求，all-reduce 会把不同请求的部分和加在一起。",
      },
      {
        q: "每张卡各自采样，靠什么保证选出同一个 token？（提示：commit 690f55a，只加了一行）",
        a: "`torch.manual_seed(42)`：各卡的 logits 经 all-gather 后完全相同，随机数状态也相同，采样结果才一致。少了这行，各卡可能采出不同的 token，后面的输入就对不上了。",
      },
    ],
    labs: [
      {
        text: "单卡跑不了真正的 TP：`CUDA_VISIBLE_DEVICES=0,0` 在你的机器上可见 GPU 数是 0，而且 NCCL 本身不允许两个 rank 共用一张卡。用纯 PyTorch 在 CPU 上验证 TP 的数学（这段已在你的 venv 里跑过，输出 True True）：",
        code: `import torch
x, W = torch.randn(4, 8), torch.randn(6, 8)      # y = x @ W.T
full = x @ W.T
# 列并行：按输出维切 W，各卡算一部分输出，拼起来
col = torch.cat([x @ w.T for w in W.chunk(2, dim=0)], dim=-1)
# 行并行：按输入维切 x 和 W，各卡得到部分和，all-reduce 就是求和
row = sum(xs @ ws.T for xs, ws in zip(x.chunk(2, dim=-1), W.chunk(2, dim=1)))
print(torch.allclose(full, col, atol=1e-5), torch.allclose(full, row, atol=1e-5))`,
      },
      { text: "真要跑 `--tp 2` 需要两张卡，可以只为这一章租一台双卡云机器。" },
    ],
  },
  {
    id: "s10",
    name: "进阶：kernel 与性能",
    time: "按兴趣",
    goal: "看懂 JIT kernel 怎么编译和调用，并用 profiler 量化 overlap 到底省了多少。",
    files: [
      "kernel/utils.py|129 行",
      "kernel/csrc/jit/store.cu|123 行",
      "kernel/csrc/jit/index.cu",
      "moe/fused.py|256 行",
      "kernel/triton/fused_moe.py|193 行",
      "utils/torch_utils.py|37 行",
    ],
    questions: [
      {
        q: "JIT 编好的 kernel 缓存在哪？为什么你第一次提问特别慢？",
        a: "mini-sglang 自己的 kernel 缓存在 `~/.cache/tvm-ffi`，FlashInfer 的也有各自的 JIT 缓存。第一次遇到某个配置要现场编译：你的机器上首次 generate 用了 15.87 秒，有缓存之后新进程第一次只要 0.10 秒。",
      },
      {
        q: "`store_cache` 为什么让一个 warp 搬一个 token？",
        a: "一个 token 的 K（或 V）是「KV 头数 × `head_dim` × 2 字节」的连续内存，Qwen3-0.6B 上是 2 KB。32 个线程各搬一段，正好合并访存；不同 token 的目标槽位是离散的，按 token 分给不同 warp 最自然。",
      },
    ],
    labs: [
      {
        text: '先在 `benchmark/offline/bench.py` 的 `LLM(...)` 里加上 `attention_backend="fi"`（否则你的卡上会崩），装好 Nsight Systems 后用 `nsys profile` 跑它。代码里的 NVTX 标注（`Layer_i`、`MHA`、`MLP`、`Sampler`）能帮你看清 CPU 调度和 GPU 计算之间有没有空隙。',
      },
      {
        text: "对比设置 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 前后 bench.py 的吞吐，量化 overlap 省了多少。",
      },
    ],
  },
];

const STATUS_OPTIONS = [
  { value: "pending", label: "未开始" },
  { value: "in_progress", label: "进行中" },
  { value: "completed", label: "已完成" },
];

// ---------------------------------------------------------------------------
// Data: tables
// ---------------------------------------------------------------------------

const BUILD_UP: [string, string, string][] = [
  ["6d6df1c", "最初的版本，只有 engine，全项目约 2,540 行", "起点"],
  ["f583dc0", "CUDA graph", "+268"],
  ["900ed45", "scheduler 核心", "+1,166"],
  ["cd7bbad", "正确性修复", "+135"],
  ["cb3b0c8", "tokenizer 进程", "+273"],
  ["436b019", "前端 API server，全项目约 4,730 行", "+516"],
  ["dc209b6", "FlashInfer 后端", "+451"],
  ["fec1c61", "prefill 和 decode 分别选后端（HybridBackend）", "+114"],
  ["80e7cfd", "张量并行", "+1,202"],
  ["ddc6dfa", "OpenAI v1 接口和 benchmark", "+673"],
  ["e3b406b", "overlap 调度修复", "+13"],
  ["35b06c3", "chunked prefill，全项目约 6,260 行", "+133"],
  ["8c560d1", "radix 树", "+307"],
  ["5e52ef3", "支持 page_size > 1", "+202 / −116"],
];

const FIX_EXAMS: [string, string, string, string][] = [
  ["20fcd7f", "FlashInfer 的竞争条件", "103", "阶段 7"],
  ["dae78f6", "page_size > 1 时的驱逐（附带测试）", "80", "阶段 4、6"],
  ["7c59f4d", "bs > 1 时 LM head all-gather 后的 reshape（只改一行）", "98", "阶段 9"],
  ["9a91cfa", "多卡之间 decode 请求顺序不一致", "113", "阶段 9"],
  ["690f55a", "采样的随机种子", "88", "阶段 9"],
  ["ab5d260", "max_seq_len 向上对齐时溢出", "68", "阶段 4"],
  ["ba3b55c", "split 之后父节点的时间戳", "124", "阶段 6"],
];

// ---------------------------------------------------------------------------
// Small building blocks
// ---------------------------------------------------------------------------

function parseSpec(spec: string) {
  const [path, note] = spec.split("|");
  const fromRepo = path.startsWith("@");
  const rel = fromRepo ? path.slice(1) : path;
  const m = /^(.*):(\d+)$/.exec(rel);
  return {
    label: rel,
    note,
    abs: (fromRepo ? REPO : PKG) + (m ? m[1] : rel),
    line: m ? Number(m[2]) : undefined,
  };
}

function FileRef({ spec }: { spec: string }) {
  const t = useHostTheme();
  const dispatch = useCanvasAction();
  const f = parseSpec(spec);
  const open = () =>
    dispatch(
      f.line
        ? {
            type: "openFile",
            path: f.abs,
            selection: {
              startLineNumber: f.line,
              startColumn: 1,
              endLineNumber: f.line,
              endColumn: 1,
            },
          }
        : { type: "openFile", path: f.abs },
    );
  return (
    <span style={{ fontSize: 12, lineHeight: "20px" }}>
      <span
        onClick={open}
        title={f.abs}
        style={{ color: t.text.link, cursor: "pointer", fontFamily: MONO }}
      >
        {f.label}
      </span>
      {f.note ? <span style={{ color: t.text.tertiary }}>{"  ·  " + f.note}</span> : null}
    </span>
  );
}

function CodeBlock({ code }: { code: string }) {
  const t = useHostTheme();
  return (
    <pre
      style={{
        margin: 0,
        padding: "8px 10px",
        borderRadius: 6,
        background: t.fill.tertiary,
        color: t.text.primary,
        fontFamily: MONO,
        fontSize: 12,
        lineHeight: "18px",
        whiteSpace: "pre",
        overflowX: "auto",
      }}
    >
      {code}
    </pre>
  );
}

function Label({ children }: { children: string }) {
  return (
    <Text size="small" tone="tertiary" weight="medium">
      {children}
    </Text>
  );
}

// ---------------------------------------------------------------------------
// Journey swimlane
// ---------------------------------------------------------------------------

const SVG_W = 600;
const LANE_W = 150;
const BOX_W = 128;
const BOX_H = 36;
const BOX_DX = 11;
const TOP = 62;
const GAP = 16;
const LABEL_GAP = 30;

const STEP_Y = STEPS.reduce<number[]>((acc, _s, i) => {
  acc.push(i === 0 ? TOP : acc[i - 1] + BOX_H + (STEPS[i - 1].edge ? LABEL_GAP : GAP));
  return acc;
}, []);
const SVG_H = STEP_Y[STEP_Y.length - 1] + BOX_H + 16;
const laneX = (lane: number) => lane * LANE_W;
const midX = (lane: number) => lane * LANE_W + LANE_W / 2;

function Journey() {
  const t = useHostTheme();
  const [sel, setSel] = useCanvasState("journeyStep", 1);
  const idx = Math.min(Math.max(sel, 1), STEPS.length) - 1;
  const step = STEPS[idx];

  const loopFrom = 9;
  const loopTo = 3;
  const boxLeft = laneX(2) + BOX_DX;
  const loopX = boxLeft - 9;
  const yA = STEP_Y[loopFrom] + BOX_H / 2;
  const yB = STEP_Y[loopTo] + BOX_H / 2;

  return (
    <Grid columns="minmax(0, 600px) minmax(260px, 1fr)" gap={24} align="start">
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ width: "100%", height: "auto", display: "block" }}
        role="img"
        aria-label="一个请求在四类执行单元之间的流转"
      >
        <defs>
          <marker
            id="journey-arrow"
            viewBox="0 0 8 8"
            refX={7}
            refY={4}
            markerWidth={7}
            markerHeight={7}
            markerUnits="userSpaceOnUse"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" fill={t.text.tertiary} />
          </marker>
        </defs>

        {LANES.map((lane, k) => (
          <g key={lane.name}>
            {k % 2 === 0 ? (
              <rect x={laneX(k)} y={0} width={LANE_W} height={SVG_H} rx={6} fill={t.fill.quaternary} />
            ) : null}
            <text x={midX(k)} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>
              {lane.name}
            </text>
            <text x={midX(k)} y={38} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
              {lane.sub}
            </text>
          </g>
        ))}
        <line x1={0} x2={SVG_W} y1={50} y2={50} stroke={t.stroke.tertiary} />

        {STEPS.slice(0, -1).map((a, i) => {
          const b = STEPS[i + 1];
          const x1 = midX(a.lane);
          const y1 = STEP_Y[i] + BOX_H;
          const x2 = midX(b.lane);
          const y2 = STEP_Y[i + 1] - 1;
          const my = y1 + 18;
          const d =
            a.lane === b.lane
              ? `M${x1},${y1} L${x2},${y2}`
              : `M${x1},${y1} L${x1},${my} L${x2},${my} L${x2},${y2}`;
          return (
            <g key={`edge-${i}`}>
              <path d={d} fill="none" stroke={t.text.tertiary} strokeWidth={1.2} markerEnd="url(#journey-arrow)" />
              {a.edge ? (
                <text x={(x1 + x2) / 2} y={y1 + 13} textAnchor="middle" fontSize={10} fill={t.text.secondary}>
                  {a.edge}
                </text>
              ) : null}
            </g>
          );
        })}

        <path
          d={`M${boxLeft},${yA} L${loopX},${yA} L${loopX},${yB} L${boxLeft - 1},${yB}`}
          fill="none"
          stroke={t.text.tertiary}
          strokeWidth={1.2}
          strokeDasharray="4 3"
          markerEnd="url(#journey-arrow)"
        />
        <text
          transform={`translate(${loopX - 5}, ${(yA + yB) / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={10}
          fill={t.text.secondary}
        >
          decode 循环：每轮一个 token
        </text>

        {STEPS.map((s, i) => {
          const active = i === idx;
          const x = laneX(s.lane) + BOX_DX;
          const y = STEP_Y[i];
          return (
            <g key={s.title} onClick={() => setSel(i + 1)} style={{ cursor: "pointer" }}>
              <rect
                x={x}
                y={y}
                width={BOX_W}
                height={BOX_H}
                rx={6}
                fill={t.bg.editor}
                stroke={active ? t.accent.primary : t.stroke.secondary}
                strokeWidth={active ? 1.8 : 1}
              />
              <text
                x={x + 9}
                y={y + 15}
                fontSize={11.5}
                fontWeight={600}
                fill={active ? t.accent.primary : t.text.primary}
              >
                {`${i + 1} · ${s.title}`}
              </text>
              <text x={x + 9} y={y + 29} fontSize={9.5} fontFamily={MONO} fill={t.text.secondary}>
                {s.box}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ position: "sticky", top: 12 }}>
        <Stack gap={10}>
          <Text size="small" tone="tertiary">
            {`第 ${idx + 1} / ${STEPS.length} 步 · ${LANES[step.lane].name}`}
          </Text>
          <H3>{step.title}</H3>
          <CodeBlock code={step.fn} />
          <Text>{step.detail}</Text>
          <Stack gap={2}>
            <Label>相关代码（点击跳到对应行）</Label>
            {step.files.map((f) => (
              <FileRef key={f} spec={f} />
            ))}
          </Stack>
          <Row gap={8}>
            <Button variant="secondary" disabled={idx === 0} onClick={() => setSel(idx)}>
              上一步
            </Button>
            <Button variant="primary" disabled={idx === STEPS.length - 1} onClick={() => setSel(idx + 2)}>
              下一步
            </Button>
          </Row>
        </Stack>
      </div>
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// Mental models
// ---------------------------------------------------------------------------

function MentalModels() {
  const t = useHostTheme();
  return (
    <Stack gap={16}>
      {MODELS.map((m, i) => (
        <div
          key={m.title}
          style={{ display: "grid", gridTemplateColumns: "28px minmax(0, 1fr)", columnGap: 12 }}
        >
          <div style={{ fontSize: 18, fontWeight: 600, lineHeight: "22px", color: t.accent.primary }}>{i + 1}</div>
          <Stack gap={4}>
            <Text weight="semibold">{m.title}</Text>
            <Text tone="secondary">{m.body}</Text>
            <Row gap={16} wrap>
              {m.files.map((f) => (
                <FileRef key={f} spec={f} />
              ))}
            </Row>
          </Stack>
        </div>
      ))}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Roadmap (master / detail)
// ---------------------------------------------------------------------------

function Roadmap() {
  const t = useHostTheme();
  const [status, setStatus] = useCanvasState<Record<string, TodoStatus>>("stageStatus", {});
  const [selId, setSelId] = useCanvasState("selectedStage", "s1");
  const [showAnswers, setShowAnswers] = useState(false);

  const stageIndex = Math.max(
    0,
    STAGES.findIndex((s) => s.id === selId),
  );
  const stage = STAGES[stageIndex];
  const todos: TodoItem[] = STAGES.map((s, i) => ({
    id: s.id,
    content: `阶段 ${i} · ${s.name}（${s.time}）`,
    status: status[s.id] ?? "pending",
  }));

  return (
    <Grid columns="minmax(220px, 300px) minmax(0, 1fr)" gap={24} align="start">
      <div style={{ position: "sticky", top: 12 }}>
        <TodoListCard
          todos={todos}
          defaultExpanded
          onTodoClick={(todo) => {
            setSelId(todo.id);
            setShowAnswers(false);
          }}
        />
      </div>

      <Stack gap={16}>
        <Row justify="space-between" align="start" gap={12}>
          <Stack gap={2}>
            <Text size="small" tone="tertiary">{`阶段 ${stageIndex} · 预计 ${stage.time}`}</Text>
            <H3>{stage.name}</H3>
          </Stack>
          <Select
            value={status[stage.id] ?? "pending"}
            onChange={(v) => setStatus((prev) => ({ ...prev, [stage.id]: v as TodoStatus }))}
            options={STATUS_OPTIONS}
            style={{ width: 104 }}
          />
        </Row>

        <Text>{stage.goal}</Text>

        {stage.reading ? (
          <Stack gap={4}>
            <Label>先读</Label>
            {stage.reading.map((r) => (
              <Text key={r} size="small">
                {r}
              </Text>
            ))}
          </Stack>
        ) : null}

        <Stack gap={2}>
          <Label>要读的文件（点击打开）</Label>
          {stage.files.map((f) => (
            <FileRef key={f} spec={f} />
          ))}
        </Stack>

        <Stack gap={10}>
          <Row justify="space-between" align="center">
            <Label>自测题（先自己答，再看参考答案）</Label>
            <Button variant="ghost" onClick={() => setShowAnswers((v) => !v)}>
              {showAnswers ? "隐藏参考答案" : "显示参考答案"}
            </Button>
          </Row>
          {stage.questions.map((qa, i) => (
            <Stack key={qa.q} gap={4}>
              <Text>{`${i + 1}. ${qa.q}`}</Text>
              {showAnswers ? (
                <div style={{ borderLeft: `2px solid ${t.stroke.secondary}`, paddingLeft: 10 }}>
                  <Text size="small" tone="secondary">
                    {qa.a}
                  </Text>
                </div>
              ) : null}
            </Stack>
          ))}
        </Stack>

        <Stack gap={10}>
          <Label>动手实验</Label>
          {stage.labs.map((lab) => (
            <Stack key={lab.text} gap={6}>
              <Text>{lab.text}</Text>
              {lab.code ? <CodeBlock code={lab.code} /> : null}
            </Stack>
          ))}
        </Stack>

        <Row gap={8}>
          <Button
            variant="secondary"
            disabled={stageIndex === 0}
            onClick={() => {
              setSelId(STAGES[stageIndex - 1].id);
              setShowAnswers(false);
            }}
          >
            上一阶段
          </Button>
          <Button
            variant="secondary"
            disabled={stageIndex === STAGES.length - 1}
            onClick={() => {
              setSelId(STAGES[stageIndex + 1].id);
              setShowAnswers(false);
            }}
          >
            下一阶段
          </Button>
        </Row>
      </Stack>
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function MiniSglangLearningMap() {
  return (
    <Stack gap={40} style={{ padding: "8px 4px 48px", maxWidth: 1200 }}>
      <Stack gap={14}>
        <Stack gap={6}>
          <H1>mini-sglang 学习地图</H1>
          <Text tone="secondary">
            按一个请求的生命周期来组织。内容来自对 8,092 行 Python 源码和 171 个提交的通读；文中提到的实验结果，都是在你的 RTX 5070 Ti 上实际跑出来的。
          </Text>
        </Stack>
        <Callout tone="info" title="学习方法">
          <Stack gap={4}>
            <Text>1. 跟着数据走：每读一个函数都问三个问题——这个张量从哪来、形状是什么、接下来谁读它。</Text>
            <Text>2. 先单进程，后多进程：先用 `LLM` 类在一个进程里调试，最后才碰 ZMQ 和进程拓扑。</Text>
            <Text>3. 先关 overlap，再开：设 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 看懂顺序执行的版本，再去理解错开一拍的版本。</Text>
            <Text>4. 每个阶段答完自测题、做完一个实验才算过关；答不上来就回去读代码，别急着看参考答案。</Text>
          </Stack>
        </Callout>
      </Stack>

      <Stack gap={12}>
        <Stack gap={4}>
          <H2>一个请求的完整旅程</H2>
          <Text tone="secondary" size="small">
            点击任意一步查看细节，文件名可以直接点开并跳到对应行。虚线是 decode 循环：第 4 到第 10 步每生成一个 token 走一遍。
          </Text>
        </Stack>
        <Journey />
      </Stack>

      <Stack gap={12}>
        <Stack gap={4}>
          <H2>五个核心心智模型</H2>
          <Text tone="secondary" size="small">
            读代码卡住时，先回到这五条。几乎每一段难懂的代码都是其中某一条的直接后果。
          </Text>
        </Stack>
        <MentalModels />
      </Stack>

      <Stack gap={12}>
        <Stack gap={4}>
          <H2>学习路线</H2>
          <Text tone="secondary" size="small">
            按顺序走，大约 2–3 周。点左边的阶段看目标、文件、自测题和实验；右上角的状态会自动保存，左边的进度随之更新。
          </Text>
        </Stack>
        <Roadmap />
      </Stack>

      <Stack gap={12}>
        <Stack gap={4}>
          <H2>在你机器上的实测</H2>
          <Text tone="secondary" size="small">
            RTX 5070 Ti（算力 12.0），Qwen3-0.6B，单进程 `LLM` 类。
          </Text>
        </Stack>
        <Table
          headers={["实验", "结果", "说明"]}
          rows={[
            [
              <>
                单进程离线推理，<Code>{'attention_backend="fi"'}</Code>
              </>,
              "成功",
              <>
                <Code>memory_ratio=0.5</Code> 时 KV 池 56,909 个 token、6.08 GiB；CUDA graph 录了 bs 1、2、4、8
              </>,
            ],
            ["首次和第二次 generate", "15.87 s → 0.07 s", "首次在 JIT 编译 kernel；有缓存后，新进程第一次也只要 0.10 s"],
            [
              "两次贪心生成后的 radix 树",
              <Code key="ev">evictable_size = 30</Code>,
              "两次输出共享 10 个 token，两个分支各 10 个",
            ],
            ["radix 和 naive 的确定性", "radix 下第二次输出变了；naive 三次一致", "前缀复用在 bf16 下不是逐位一致，接近平局的 argmax 会翻转"],
            [
              <>
                默认的 <Code>--attn auto</Code>
              </>,
              "一提问就崩溃",
              "算力 12.0 被「≥ 10.0」的判断误选成 trtllm",
            ],
            [<Code key="fa">{'attention_backend="fa,fi"'}</Code>, "报错", "同样按「≥ 10.0」选了 FA4，报 cutlass 的 AttributeError"],
            [<Code key="cvd">CUDA_VISIBLE_DEVICES=0,0</Code>, "可见 GPU 数为 0", "单卡模拟不了 TP；NCCL 也不允许两个 rank 共用一张卡"],
          ]}
          rowTone={["success", "info", "info", "warning", "danger", "danger", "danger"]}
        />
      </Stack>

      <Stack gap={12}>
        <Stack gap={4}>
          <H2>用 git 历史重走作者的路</H2>
          <Text tone="secondary" size="small">
            一次读完 8,000 行很难，按作者加功能的顺序读就容易得多。你本地的 `server/launch.py` 有未提交的修改，所以用 worktree 检出旧版本，不要直接 checkout。
          </Text>
        </Stack>
        <CodeBlock
          code={`git worktree add ../minisgl-v1 6d6df1c   # 最初只有 engine 的版本，约 2,500 行
git show 8c560d1 --stat                  # radix 树是怎么加进来的
git show 690f55a                         # 先看标题猜怎么修，再看 diff
git worktree remove ../minisgl-v1`}
        />
        <H3>功能是按什么顺序长出来的</H3>
        <Table
          headers={["提交", "加了什么", "改动行数"]}
          columnAlign={["left", "left", "right"]}
          rows={BUILD_UP.map(([c, what, delta]) => [<Code key={c}>{c}</Code>, what, delta])}
          striped
        />
        <H3>拿修 bug 的提交当考题</H3>
        <Text tone="secondary" size="small">
          只看标题，先写下你认为 bug 在哪、该怎么修，再 `git show` 对答案。它们都很短，却分别戳中了对应阶段最容易误解的地方。
        </Text>
        <Table
          headers={["提交", "修的是什么", "PR", "对应阶段"]}
          rows={FIX_EXAMS.map(([c, what, pr, stage]) => [
            <Code key={c}>{c}</Code>,
            what,
            <Link key={pr} href={PR + pr}>
              {`#${pr}`}
            </Link>,
            stage,
          ])}
        />
      </Stack>

      <Stack gap={12}>
        <Stack gap={4}>
          <H2>毕业项目</H2>
          <Text tone="secondary" size="small">
            融会贯通的标准：合上代码，能在白纸上画出上面的请求旅程和两张表的读写关系，并独立做完至少一个「中」难度的项目。
          </Text>
        </Stack>
        <Table
          headers={["项目", "要动的层", "难度", "从这里开始"]}
          rows={[
            ["补上 usage 统计：非流式响应里的 token 数现在固定返回 0", "API server", "低", <FileRef key="u" spec="server/api_server.py" />],
            ["修好 SM120 的后端自动选择（阶段 7 的实验）", "engine", "低", <FileRef key="a" spec="engine/engine.py:218" />],
            [
              <>
                实现 <Code>stop</Code> 字符串：请求里有这个字段，但从没被用到
              </>,
              "API、detokenizer、scheduler",
              "中",
              <FileRef key="s" spec="tokenizer/detokenize.py:70" />,
            ],
            [
              <>
                实现 <Code>{"n > 1"}</Code>：同一个 prompt 生成多个回答，让 radix cache 自动共享 prompt 的 KV
              </>,
              "API、scheduler",
              "中",
              <FileRef key="n" spec="server/api_server.py:256" />,
            ],
            ["实现 frequency / presence penalty：要在 GPU 上维护每个请求的 token 计数", "sampler、scheduler", "中高", <FileRef key="p" spec="engine/sample.py:71" />],
            [
              <>
                实现抢占：显存不够时把 decode 请求退回队列，而不是按 <Code>max_tokens</Code> 保守预留
              </>,
              "scheduler、cache manager",
              "高",
              <FileRef key="r" spec="scheduler/prefill.py:33" />,
            ],
            ["不看代码，从零复刻一个最小版：单进程、naive cache、无 CUDA graph、无 overlap", "全部", "高", <FileRef key="c" spec="core.py:29" />],
          ]}
        />
      </Stack>
    </Stack>
  );
}
