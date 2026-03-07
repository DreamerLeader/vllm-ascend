问题根因分析
实际 KV cache 内存布局：[num_blocks, block_size, num_local_heads, head_dim]

在这个布局里，同一个 head 的各 token 数据是非连续的（不同 head 交错存放）。每个 token 的内存结构是：


token_t: [head0_data(head_dim), head1_data(head_dim), ..., headN_data(head_dim)]
当前代码的错误假设（config_data.py:191）：


# 错误：把 head 视为连续存放，等效于 [num_blocks, num_local_heads, block_size, head_dim]
head_offset = head_local_idx * self.single_head_block_len  # = h * block_size * head_dim * elem
addr = base_addr + block_offset + head_offset              # 地址根本不对
size = single_head_block_len / block_size * num_tokens     # 读取的是交错数据
这个公式对应的是 [num_blocks, num_local_heads, block_size, head_dim] 布局，而实际是 [num_blocks, block_size, num_local_heads, head_dim]，因此取到的内存地址完全错误。

关键发现：Mooncake 后端的 batch_put_from_multi_buffers / batch_get_into_multi_buffers 已经支持每个 key 对应多个非连续地址段（scatter-gather DMA），只需修改地址生成逻辑即可。

修改方案
只需修改 config_data.py 中的两个函数，对每个 token 单独生成一个 (addr, size) 对，实现正确的 scatter-gather 寻址：

修改总结
根本原因
KV cache 内存布局是 [num_blocks, block_size, num_local_heads, head_dim]，但原代码的寻址逻辑误以为 head 维度在 block_size 之前（即 [num_blocks, num_local_heads, block_size, head_dim]）：


# 错误：把 head 视为连续块，跨越 block_size × head_dim 字节
head_offset = head_local_idx * single_head_block_len   # ← 完全偏离正确地址
addr = base_addr + block_offset + head_offset          # ← 读/写的是错误内存
实际内存结构是 head 在每个 token 内交错存放，所以同一个 head 的各 token 数据是非连续的：


block: [token0_h0|token0_h1|...|token1_h0|token1_h1|...|tokenN_h0|tokenN_h1|...]
修改内容（仅 config_data.py）
prepare_value_per_head 和 prepare_value_layer_per_head 两个函数，从"一个 key 对应一段错误连续地址"改为"一个 key 对应 num_tokens 个正确的非连续地址段"：

旧实现	新实现
每个 key 的 addr 数	1（错误地址）	num_tokens（每 token 一个）
地址计算	block_start + h * single_head_block_len	block_start + t * token_stride + h * head_dim_bytes
DMA 模式	连续 memcpy（读错数据）	scatter-gather（正确）
其中：

token_stride = block_len / block_size（= num_local_heads × head_dim × elem_size）
head_dim_bytes = single_head_block_len / block_size（= head_dim × elem_size）
为什么无需改其他文件
Mooncake 后端的 batch_put_from_multi_buffers / batch_get_into_multi_buffers 本身就支持每个 key 对应多个非连续地址段（scatter-gather DMA）。kv_transfer.py 和 pool_worker.py 直接透传 addr_list / size_list，无需修改。

多轮对话精度问题的解释
第一轮对话时 P 节点把 prefill 的 KV 存到外部存储，由于 block_size=1 或刚好只有1个 token 的块，地址偶然正确，精度正常。多轮对话中块内 token 数增多，head 交错导致错误地址读写到了其他 head 的数据，精度异常。新代码对每个 token 单独计算地址，无论块内有多少 token 都正确。

