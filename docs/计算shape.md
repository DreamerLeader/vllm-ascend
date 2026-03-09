针对形状为 `[num_blocks, block_size, num_local_heads, head_dim]` 的 KV Cache，要实现“按照模型的头取地址”，并获取每个 token 按头连续的地址，核心在于计算内存偏移量。

这种布局通常用于 PagedAttention（如 vLLM）或推理框架中。在这种布局下，**对于一个特定的 Head，其对应的 Token 数据在物理内存上并不是连续的，而是“跨度”分布的**。

以下是详细的计算方法和逻辑分析：

### 1. 内存布局与步长分析

首先定义各个维度的含义：
*   `num_blocks` ($B$): 物理块的数量。
*   `block_size` ($T$): 每个块内的 token 数量。
*   `num_local_heads` ($H$): 注意力头的数量。
*   `head_dim` ($D$): 每个头的维度大小。

在 C/C++ 或 CUDA 编程中，内存是线性排列的。对于行优先存储，各维度的步长如下：
*   **块步长**: 一个完整块的大小。
    $$Stride_{block} = T \times H \times D$$
*   **Token 步长**: 块内相邻 Token 之间的距离。
    $$Stride_{token} = H \times D$$
    *(注：这意味着 Token $t$ 和 Token $t+1$ 之间隔了 $H$ 个 Head 的数据)*
*   **头步长**: 同一个 Token 内，相邻 Head 之间的距离。
    $$Stride_{head} = D$$
    *(注：这意味着 Head $h$ 和 Head $h+1$ 的数据是紧挨着的)*
*   **维度步长**: Head 内相邻元素的距离。
    $$Stride_{dim} = 1$$

### 2. 地址计算公式

假设 KV Cache 的基地址为 `base_ptr`，我们要获取 **第 `b` 个块**、**第 `t` 个 token**、**第 `h` 个头** 的地址。

计算公式如下：

$$
\text{Addr} = \text{base\_ptr} + (b \times Stride_{block}) + (t \times Stride_{token}) + (h \times Stride_{head})
$$

代入具体数值：

$$
\text{Addr} = \text{base\_ptr} + (b \times T \times H \times D) + (t \times H \times D) + (h \times D)
$$

### 3. 如何实现“按头取地址”

你提到的“按照模型的头取地址”，通常是指在处理 Attention 时，我们固定一个 Head，然后遍历该 Head 下的所有 Token。

由于内存布局是 `[Block, Token, Head, Dim]`，**对于特定的 Head $h$，其所有 Token 的数据在内存中是不连续的**。

如果你要获取 Head $h$ 在 Block $b$ 中所有 Token 的地址序列，计算方式如下：

*   **该 Head 在该 Block 的起始地址**:
    $$P_{start} = \text{base\_ptr} + b \times (T \times H \times D) + h \times D$$

*   **Token 之间的跳跃步长**:
    $$\text{Jump} = H \times D$$

这意味着，如果你想顺序读取 Head $h$ 的 Token $0, 1, 2...$，你的指针每次需要增加 `num_local_heads * head_dim`。

### 4. 代码示例 (Python/CUDA 风格伪代码)

假设你需要编写一个内核或函数来获取地址：

```python
# 定义维度变量
# num_blocks, block_size, num_local_heads, head_dim

def get_head_address(base_ptr, block_idx, token_idx, head_idx):
    # 计算各维度步长
    stride_block = block_size * num_local_heads * head_dim
    stride_token = num_local_heads * head_dim
    stride_head = head_dim
    
    # 计算线性偏移量
    offset = (block_idx * stride_block) + \
             (token_idx * stride_token) + \
             (head_idx * stride_head)
             
    return base_ptr + offset

# 如果你需要获取某个头(Head h)在某个块(Block b)下，所有token的"连续逻辑地址"（实际上物理不连续）：
# 这是一个遍历逻辑
block_idx = 5  # 假设取第5个块
head_idx = 0   # 假设取第0个头

# 该头在该块中的基地址
head_base = base_ptr + (block_idx * block_size * num_local_heads * head_dim) + (head_idx * head_dim)

# 遍历该块内的 token
for t in range(block_size):
    # 当前 token 的地址
    # 注意：这里每次循环指针增加 num_local_heads * head_dim
    current_addr = head_base + (t * num_local_heads * head_dim)
    # ... 读取 current_addr 处的 head_dim 个数据 ...
```

### 5. 进阶：结合 Block Table 的实际推理场景

在实际的 LLM 推理（如 vLLM）中，逻辑上的 Sequence 并不是连续的 Block，而是通过 `Block Table` 映射的。此时计算“每个 Token 按头连续的地址”稍微复杂一点：

假设有一个 Sequence，由逻辑 Block ID 列表组成：`[block_0, block_1, ...]`。
要获取逻辑 Token 序列下，Head $h$ 的数据地址：

```cpp
// C++/CUDA 伪代码
// block_table: 存储逻辑 token 对应的物理 block id
// logical_token_idx: 逻辑上的第几个 token (0 ~ seq_len)

int physical_block_id = block_table[logical_token_idx / block_size];
int offset_in_block = logical_token_idx % block_size;

// 计算物理地址
// 注意：这里依然使用了 stride_token = num_local_heads * head_dim
// 因为在物理块内，token 依然是交错存储的
int addr = base_ptr 
         + physical_block_id * (block_size * num_local_heads * head_dim) 
         + offset_in_block * (num_local_heads * head_dim) 
         + head_idx * head_dim;
```

### 总结

对于 `[num_blocks, block_size, num_local_heads, head_dim]` 形状的 KV Cache：

1.  **地址计算**：使用公式 `offset = b*T*H*D + t*H*D + h*D`。
2.  **数据连续性**：
    *   **Head 维度**：同一个 Token 的同一个 Head 内部数据（维度方向）是**连续**的（长度为 `head_dim`）。
    *   **Token 维度**：同一个 Head 的不同 Token 数据是**不连续**的，每隔 `num_local_heads * head_dim` 分布一次。

如果你希望“按头连续”访问（即读头0的所有token，再读头1的所有token），这种内存布局是不友好的，你需要通过 stride 访问（跳跃读取）。如果硬件支持，可以使用向量化加载（如 CUDA 的 `ld.global.nc.vector`）结合步长访问来优化性能。
