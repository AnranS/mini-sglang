// The request journey on the home page. `refs` are resolved to GitHub links
// with current line numbers at build time (see src/lib/source.mjs).

export const LANES = [
	{ name: 'API Server', sub: '主进程 · FastAPI / shell' },
	{ name: 'Tokenizer', sub: '独立进程 · 分词 + 反分词' },
	{ name: 'Scheduler', sub: '每卡一个进程 · CPU 侧' },
	{ name: 'Engine', sub: '同一进程 · GPU 侧' },
];

/** Indices (0-based) of the steps that the dashed decode loop connects. */
export const DECODE_LOOP = { from: 9, to: 3, label: 'decode 循环：每轮一个 token' };

export const STEPS = [
	{
		lane: 0,
		title: '接收请求',
		box: 'v1_completions',
		fn: 'v1_completions()  /  shell_completion()',
		detail:
			'`FrontendManager.new_user` 分配一个自增的 uid，同时为它建一个 `asyncio.Event` 和一个回包列表。messages（或 prompt）和采样参数被打包成 `TokenizeMsg`，经 ZeroMQ 发给 tokenizer 进程；这个协程随后挂在 `wait_for_ack` 上等回包。',
		edge: 'ZMQ · TokenizeMsg',
		refs: [
			{ path: 'server/api_server.py', sym: 'v1_completions' },
			{ path: 'server/api_server.py', sym: 'shell_completion' },
			{ path: 'server/api_server.py', sym: 'FrontendManager.new_user' },
		],
	},
	{
		lane: 1,
		title: '分词',
		box: 'tokenize_worker',
		fn: 'tokenize_worker()  →  TokenizeManager.tokenize()',
		detail:
			'messages 是列表时先 `apply_chat_template`（带 `add_generation_prompt=True`），再 `encode` 成一维 int32 的 CPU tensor，包成 `UserMsg` 发给 scheduler。默认 `--num-tokenizer 0`，分词和反分词在同一个进程里：它按消息类型分拣，`TokenizeMsg` 去分词，`DetokenizeMsg` 去反分词。',
		edge: 'ZMQ · UserMsg',
		refs: [
			{ path: 'tokenizer/server.py', sym: 'tokenize_worker' },
			{ path: 'tokenizer/tokenize.py', sym: 'TokenizeManager.tokenize' },
		],
	},
	{
		lane: 2,
		title: '入队',
		box: '_process_one_msg',
		fn: 'receive_msg()  →  Scheduler._process_one_msg()',
		detail:
			'rank 0 从 ZeroMQ 收消息，多卡时再转发给其他 rank。prompt 长度达到 `max_seq_len` 的请求直接丢弃；`max_tokens` 超过 `max_seq_len − 输入长度` 时被截短。之后请求变成 `PendingReq`，进入 `PrefillManager.pending_list` 排队。',
		refs: [
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._process_one_msg' },
			{ path: 'scheduler/io.py', sym: 'SchedulerIOMixin._recv_msg_single_rank' },
			{ path: 'scheduler/prefill.py', sym: 'PrefillManager.add_one_req' },
		],
	},
	{
		lane: 2,
		title: '选下一批',
		box: '_schedule_next_batch',
		fn: 'Scheduler._schedule_next_batch()',
		detail:
			'prefill 优先：只要等待队列里有请求放得进来，这一批就全是 prefill。`PrefillAdder` 先在前缀缓存里匹配 `input_ids[:-1]`，再检查「要算的 token + `max_tokens` + 已经预留的量」能否放进「空闲 + 可驱逐」的 KV 空间；放得下就锁住前缀、分一行 table，超出 token 预算的部分变成 `ChunkedReq`。遇到第一个放不下的请求就停（先来先服务）。一个 prefill 都排不出来时，才由 `DecodeManager` 把所有在跑的请求按 uid 排序组成 decode 批。',
		refs: [
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._schedule_next_batch' },
			{ path: 'scheduler/prefill.py', sym: 'PrefillAdder._try_allocate_one' },
			{ path: 'scheduler/prefill.py', sym: 'PrefillManager.schedule_next_batch' },
			{ path: 'scheduler/decode.py', sym: 'DecodeManager.schedule_next_batch' },
		],
	},
	{
		lane: 2,
		title: '准备 batch',
		box: '_prepare_batch',
		fn: 'Scheduler._prepare_batch()',
		detail:
			'先用 dummy 请求把 decode 批补齐到录好的 CUDA graph 尺寸；再按页给这批新 token 分配 KV 槽位并写进 `page_table`；算出 positions、`out_loc`（每个新 token 的 K/V 要写到哪个槽位）和读写 `token_pool` 用的下标；让 attention 后端准备 metadata；把采样参数搬上 GPU。这些都在 scheduler 自己的 CUDA stream 上异步发射。',
		edge: '发射到 engine stream',
		refs: [
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._prepare_batch' },
			{ path: 'engine/graph.py', sym: 'GraphRunner.pad_batch' },
			{ path: 'scheduler/cache.py', sym: 'CacheManager.allocate_paged' },
			{ path: 'attention/fi.py', sym: 'FlashInferBackend.prepare_metadata' },
		],
	},
	{
		lane: 3,
		title: '前向',
		box: 'forward_batch',
		fn: 'Scheduler._forward()  →  Engine.forward_batch()',
		detail:
			'`_forward` 先按下标从 GPU 上的 `token_pool` 取出这批的输入 token。batch 通过全局 `Context` 交给模型，所以 `model.forward()` 没有参数。decode 批且 batch size 不超过录制上限时 replay CUDA graph，否则直接调 `model.forward()`。kernel 一发射完，就对每个请求调用一次 `complete_one()` 推进长度（这时 GPU 多半还没算完），然后发射采样。',
		refs: [
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._forward' },
			{ path: 'engine/engine.py', sym: 'Engine.forward_batch' },
			{ path: 'core.py', sym: 'Context.forward_batch' },
		],
	},
	{
		lane: 3,
		title: '逐层计算',
		box: 'Qwen3.forward',
		fn: 'Qwen3ForCausalLM.forward()',
		detail:
			'embedding → N 个 decoder 层（Qwen3-0.6B 是 28 层）→ 最后一个 RMSNorm → LM head。每层是「残差相加 + RMSNorm → QKV 投影 → attention → O 投影 → 残差相加 + RMSNorm → MLP」，残差相加和 RMSNorm 融合在一个 kernel 里。prefill 批只取每个请求最后一个 token 的 hidden state 去算 logits。',
		refs: [
			{ path: 'models/qwen3.py', sym: 'Qwen3ForCausalLM.forward' },
			{ path: 'models/qwen3.py', sym: 'Qwen3DecoderLayer.forward' },
			{ path: 'layers/embedding.py', sym: 'ParallelLMHead.forward' },
		],
	},
	{
		lane: 3,
		title: 'Attention',
		box: 'AttentionLayer',
		fn: 'AttentionLayer.forward()  →  attn_backend.forward()',
		detail:
			'先对 q、k 逐头做 RMSNorm（Qwen3 特有）和 RoPE，再交给后端。以 FlashInfer 后端为例：第一层进来时先 `plan()` 一次；每层都用 `store_kv` 按 `out_loc` 把新 token 的 K/V 写进显存池，再 `run()`，通过 `page_table` 查出的槽位读出全部历史 KV 做 attention。',
		refs: [
			{ path: 'layers/attention.py', sym: 'AttentionLayer.forward' },
			{ path: 'attention/fi.py', sym: 'FlashInferBackend.forward' },
			{ path: 'kvcache/mha_pool.py', sym: 'MHAKVCache.store_kv' },
			{ path: 'kernel/csrc/jit/store.cu' },
		],
	},
	{
		lane: 3,
		title: '采样',
		box: 'Sampler.sample',
		fn: 'Sampler.sample()  →  token_pool[write_tuple] = next_tokens',
		detail:
			'整批都是贪心就直接 argmax，否则走 FlashInfer 的 softmax 和 top-k / top-p 采样。新 token 写回 GPU 上的 `token_pool`，下一轮 decode 直接从那里读输入；同时非阻塞地拷一份到 CPU，并记一个 CUDA event 标记拷贝完成。',
		edge: 'next token 异步拷回 CPU',
		refs: [
			{ path: 'engine/sample.py', sym: 'Sampler.sample' },
			{ path: 'engine/sample.py', sym: 'Sampler.prepare' },
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._forward', text: 'self.token_pool[output_mapping]' },
		],
	},
	{
		lane: 2,
		title: '收尾（下一轮）',
		box: '_process_last_data',
		fn: 'Scheduler._process_last_data()',
		detail:
			'overlap 模式下，这一步发生在下一批已经发射之后。先等拷贝事件完成，把 token 追加到 CPU 侧的 `input_ids`，判断是否遇到 EOS 或写满 `max_tokens`；后者看的是已经被下一批推进过的长度，所以会早一个 token 判定结束。结束的请求释放 table 行、把 KV 插回前缀缓存，并记进 `finished_reqs`，防止下一轮重复释放；prefill 批里没结束的请求也会先把 prompt 的 KV 插进前缀缓存。',
		edge: 'ZMQ · DetokenizeMsg',
		refs: [
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._process_last_data' },
			{ path: 'scheduler/cache.py', sym: 'CacheManager.cache_req' },
		],
	},
	{
		lane: 1,
		title: '增量反分词',
		box: 'detokenize',
		fn: 'DetokenizeManager.detokenize()',
		detail:
			'每个请求维护一份解码状态。每次把「已确认文本末尾的几个 token + 新 token」一起解码，减掉已确认的部分，得到新增文本。结果以 � 结尾说明一个 UTF-8 字符还没拼完，就先扣住不发；中文流式输出不乱码靠的就是这里。',
		edge: 'ZMQ · UserReply',
		refs: [{ path: 'tokenizer/detokenize.py', sym: 'DetokenizeManager.detokenize' }],
	},
	{
		lane: 0,
		title: '流式返回',
		box: 'wait_for_ack',
		fn: 'FrontendManager.listen()  →  wait_for_ack()',
		detail:
			'后台协程 `listen` 收到 `UserReply` 后，按 uid 放进回包列表并唤醒对应的 Event；`wait_for_ack` 把增量文本逐条交给 SSE 响应。流式请求的客户端断开时，`stream_with_cancellation` 发出 `AbortMsg`，一路传到 scheduler 释放资源。',
		refs: [
			{ path: 'server/api_server.py', sym: 'FrontendManager.listen' },
			{ path: 'server/api_server.py', sym: 'FrontendManager.wait_for_ack' },
			{ path: 'server/api_server.py', sym: 'FrontendManager.stream_with_cancellation' },
		],
	},
];
