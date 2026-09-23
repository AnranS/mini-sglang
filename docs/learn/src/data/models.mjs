// The five mental models on the home page.

export const MODELS = [
	{
		title: '一个请求就是三个长度',
		body:
			'`cached_len` 是 KV 已经算好的 token 数，`device_len` 是 token id 已经确定的 token 数，`max_device_len` 是 prompt 长度加 `max_tokens`。每轮要算 `extend_len = device_len − cached_len` 个 token。prefill、chunked prefill、decode 用的是同一套公式，decode 只是 `extend_len` 等于 1 的特例。',
		refs: [
			{ path: 'core.py', sym: 'Req' },
			{ path: 'core.py', sym: 'Req.complete_one' },
		],
	},
	{
		title: '两张二维表',
		body:
			'`token_pool[table_idx, pos]` 存 token id，`page_table[table_idx, pos]` 存这个 token 的 KV 在显存池里的槽位，每个在跑的请求占一行。调度器只维护这两张表；模型看到的只是从表里查出来的 `input_ids`、`positions` 和 `out_loc`，根本不知道「请求」是什么。',
		refs: [
			{ path: 'scheduler/table.py', sym: 'TableManager' },
			{ path: 'engine/engine.py', text: 'self.ctx.page_table = self.page_table' },
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._prepare_batch' },
		],
	},
	{
		title: '显存和归属分离',
		body:
			'`MHAKVCache` 是一整块张量，自己不记录哪个槽位属于谁。归属由 `CacheManager` 管：空闲页放在 `free_slots` 里，已经算好的前缀挂在 radix 树上（「token 前缀 → 槽位」）。树用引用计数防止正在用的前缀被驱逐，空间不够时按访问时间淘汰叶子。',
		refs: [
			{ path: 'kvcache/mha_pool.py', sym: 'MHAKVCache' },
			{ path: 'scheduler/cache.py', sym: 'CacheManager' },
			{ path: 'kvcache/radix_cache.py', sym: 'RadixPrefixCache' },
		],
	},
	{
		title: '全局 Context 代替传参',
		body:
			'`model.forward()` 没有参数，每一层通过 `get_global_ctx().batch` 取 positions、attention metadata 和 KV 池。输入都放在地址固定的 buffer 里，decode 才能录成 CUDA graph 反复 replay；换一个 attention 后端也只需要实现 5 个方法。',
		refs: [
			{ path: 'core.py', sym: 'Context' },
			{ path: 'engine/graph.py', sym: 'GraphCaptureBuffer' },
			{ path: 'attention/base.py', sym: 'BaseAttnBackend' },
		],
	},
	{
		title: 'Overlap：CPU 永远领先 GPU 一拍',
		body:
			'CPU 每一轮先排好并发射第 N 批，再回头处理第 N−1 批的结果。于是 GPU 算第 N 批的同时，CPU 在做第 N−1 批的收尾和第 N+1 批的调度，GPU 几乎不用等。代价有两个：采样出的 token 必须先写进 GPU 上的 `token_pool`，下一批直接从那里读；「请求结束了」的判断和 GPU 上的真实进度错开一拍，以 EOS 结束的请求会被照常排进下一批多算一步，以 `max_tokens` 结束的请求则早一个 token 被判定结束，代码用 `finished_reqs` 防止重复释放。',
		refs: [
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler.overlap_loop' },
			{ path: 'scheduler/scheduler.py', sym: 'Scheduler._process_last_data', text: 'req not in self.finished_reqs' },
		],
	},
];
