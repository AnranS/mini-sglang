// Single source of truth for the learning route. Page titles in
// src/content/docs/stages/*.mdx must match `title` here.

/** @typedef {{ id: string, n: number, slug: string, title: string, time: string, summary: string, prereq: string[] }} Stage */

/** @type {Stage[]} */
export const STAGES = [
	{
		id: 's0',
		n: 0,
		slug: 'stages/00-prerequisites',
		title: '预备知识',
		time: '按需 1–3 天',
		summary: 'KV cache 为什么要分页，prefill 和 decode 有什么不同，为什么推理服务要拼命组批。',
		prereq: [],
	},
	{
		id: 's1',
		n: 1,
		slug: 'stages/01-run-and-debug',
		title: '跑起来，单步调试',
		time: '半天',
		summary: '用单进程的 LLM 类跑通一个请求，在调试器里把它从头跟到尾。',
		prereq: ['s0'],
	},
	{
		id: 's2',
		n: 2,
		slug: 'stages/02-data-structures',
		title: '核心数据结构',
		time: '半天',
		summary: 'Req 的三个长度、Batch、全局 Context，以及调度器维护的两张二维表。',
		prereq: ['s1'],
	},
	{
		id: 's3',
		n: 3,
		slug: 'stages/03-model-forward',
		title: '模型前向',
		time: '1–2 天',
		summary: '一个 decoder 层从加载权重到前向计算的每一行，并和 HF transformers 逐层对上。',
		prereq: ['s2'],
	},
	{
		id: 's4',
		n: 4,
		slug: 'stages/04-kv-memory',
		title: '显存与 KV 池',
		time: '1 天',
		summary: '显存怎么分给权重和 KV 池，一个 token 的 K/V 最终写进哪个地址。',
		prereq: ['s2', 's3'],
	},
	{
		id: 's5',
		n: 5,
		slug: 'stages/05-scheduler',
		title: '调度器',
		time: '2 天',
		summary: '准入控制、chunked prefill、prefill 优先，以及 overlap 模式下 CPU 领先 GPU 一拍。',
		prereq: ['s2', 's4'],
	},
	{
		id: 's6',
		n: 6,
		slug: 'stages/06-radix-cache',
		title: 'Radix cache',
		time: '1–2 天',
		summary: '前缀树怎么复用 KV：匹配、加锁、插入、分裂、驱逐，以及哪些槽位必须还回去。',
		prereq: ['s4', 's5'],
	},
	{
		id: 's7',
		n: 7,
		slug: 'stages/07-attention-cuda-graph',
		title: 'Attention 后端与 CUDA graph',
		time: '1–2 天',
		summary: '后端接口的 5 个方法、FlashInfer 先 plan 再 run，以及 decode 的 CUDA graph 录制与 replay。',
		prereq: ['s3', 's4'],
	},
	{
		id: 's8',
		n: 8,
		slug: 'stages/08-serving',
		title: '多进程服务化',
		time: '1 天',
		summary: 'API server、tokenizer、scheduler 怎么启动、怎么用 ZeroMQ 传消息、怎么取消请求。',
		prereq: ['s1', 's5'],
	},
	{
		id: 's9',
		n: 9,
		slug: 'stages/09-tensor-parallel',
		title: '张量并行',
		time: '1–2 天',
		summary: '权重怎么切、每层在哪里通信、多个 rank 怎么保持步调完全一致。',
		prereq: ['s3', 's8'],
	},
	{
		id: 's10',
		n: 10,
		slug: 'stages/10-kernels-performance',
		title: '进阶：kernel 与性能',
		time: '按兴趣',
		summary: 'JIT kernel 怎么编译和调用；用 benchmark 量出 overlap 和 CUDA graph 各省了多少，再用 profiler 看时间花在哪里。',
		prereq: ['s4', 's7'],
	},
];

export const STAGE_BY_ID = Object.fromEntries(STAGES.map((s) => [s.id, s]));
