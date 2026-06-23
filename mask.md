toolName: Skill
            
status: success
          
          
我来分析这个 bug。让我先确认本地仓库的状态，并获取 PR 中 `coordinator.py` 的完整代码以及 `kv_transfer.py` / `config_data.py` 的修改部分。
        
            
toolName: file_search
            
status: success
          
file_pattern: **/ascend_store/coordinator.py
          

        
            
toolName: web_search
            
status: failed
          
query: https://raw.githubusercontent.com/Pz1116/vllm-ascend/refs/heads/ascend_store_retention/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py
          
本地仓库没有 `coordinator.py`（说明本地是 PR 合入前的状态）。但我已从 PR diff 中获取了完整的 `coordinator.py` 代码。让我再确认 PR 对 `config_data.py` 中 `mask_allows_chunk` 的修改，以及 `kv_transfer.py` 中 mask 的应用方式，以完整定位根因。
        
            
toolName: web_search
            
status: success
          
query: https://github.com/vllm-project/vllm-ascend/pull/10439/files?file=vllm_ascend%2Fdistributed%2Fkv_transfer%2Fkv_pool%2Fascend_store%2Fconfig_data.py
          
现在让我获取 `config_data.py` 中 `store_mask`/`load_mask`/`mask_allows_chunk` 的实现，以及发送线程如何应用 `store_mask`。
        
            
toolName: web_search
            
status: success
          
query: https://github.com/vllm-project/vllm-ascend/pull/10439/files?file=vllm_ascend%2Fdistributed%2Fkv_transfer%2Fkv_pool%2Fascend_store%2Fconfig_data.py&show-viewed-files=false
          

        
            
toolName: CompactFake
            
status: success
          
          
我来分析这个 bug：KV cache 能 put 进去但 get 不出来。让我先查看相关代码，重点关注 load 路径上的 mask 应用逻辑。
        
            
toolName: view_files
            
status: failed
          
            
filePath: /Users/fangjianwei/fjw/0623/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/fangjianwei/fjw/0623/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py
          
coordinator.py 在本地不存在，说明 PR 还未合并到本地。让我查看 config_data.py 的剩余部分以及 kv_transfer.py，并确认本地仓库的状态。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/fangjianwei/fjw/0623/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py
          

        
            
toolName: view_folder
            
status: success
          
/Users/fangjianwei/fjw/0623/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store
          
本地没有 coordinator.py，说明 PR #10439 还未合并到本地。让我检查 git 状态，并查看 PR 的实际改动内容。
        
            
toolName: run_command
            
status: success
          
command: git log -n 5 --oneline undefined
          

        
            
toolName: run_command
            
status: success
          
command: git status undefined
          

        
            
toolName: run_command
            
status: success
          
command: git log -n 5 --oneline 2>&1; echo "---STATUS---"; git status 2>&1; echo "---BRANCH---"; git branch --show-current 2>&1 undefined
          
本地仓库在 main 分支且是干净的，PR #10439 还未合并。让我获取 PR 的实际改动内容来分析这个 bug。
        
            
toolName: web_search
            
status: success
          
query: https://github.com/vllm-project/vllm-ascend/pull/10439.diff
          
diff 被截断了。让我直接获取 PR 分支上的关键文件来分析 bug。
        
            
toolName: web_search
            
status: failed
          
query: https://raw.githubusercontent.com/vllm-project/vllm-ascend/refs/pull/10439/head/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py
          

        
            
toolName: web_search
            
status: failed
          
query: https://raw.githubusercontent.com/vllm-project/vllm-ascend/refs/pull/10439/head/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py
          
让我通过 GitHub API 获取 PR 的关键文件内容。
        
            
toolName: web_search
            
status: success
          
query: https://api.github.com/repos/vllm-project/vllm-ascend/pulls/10439/files?per_page=100
          
API 返回也被截断了。让我直接获取关键文件的内容。
        
            
toolName: web_search
            
status: success
          
query: https://api.github.com/repos/vllm-project/vllm-ascend/contents/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py?ref=3b15882f356e3d372ab0c11c0720de290faac86e
          
base64 内容被截断了。让我用 git 直接拉取 PR 分支到本地，这样能完整读取文件。
        
            
toolName: run_command
            
status: success
          
command: git fetch origin pull/10439/head:pr-10439 2>&1 | tail -5 undefined
          

        
            
toolName: run_command
            
status: success
          
command: git remote -v 2>&1; echo "---"; git config --get-regexp 'remote' 2>&1 | head -20 undefined
          
remote 指向的是 fork 仓库。让我添加 upstream 并获取 PR。
        
            
toolName: run_command
            
status: success
          
command: git remote add upstream https://github.com/vllm-project/vllm-ascend.git 2>&1; git fetch upstream pull/10439/head:pr-10439 2>&1 | tail -10 undefined
          

        
            
toolName: run_command
            
status: success
          
command: git branch | grep pr-10439 2>&1; echo "---"; git log pr-10439 -n 3 --oneline 2>&1 undefined
          
PR 分支已获取。现在让我读取关键文件来分析 bug。
        
            
toolName: run_command
            
status: success
          
command: git show pr-10439:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py undefined
          
输出被截断了。让我把文件保存到临时位置再完整读取。
        
            
toolName: run_command
            
status: success
          
command: git show pr-10439:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py > /tmp/pr10439_coordinator.py 2>&1; wc -l /tmp/pr10439_coordinator.py undefined
          

        
            
toolName: run_command
            
status: success
          
command: git show pr-10439:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py > /tmp/pr10439_kv_transfer.py 2>&1; wc -l /tmp/pr10439_kv_transfer.py undefined
          

        
            
toolName: run_command
            
status: success
          
command: git show pr-10439:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py > /tmp/pr10439_config_data.py 2>&1; wc -l /tmp/pr10439_config_data.py undefined
          

        
            
toolName: run_command
            
status: success
          
command: git show pr-10439:vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py > /tmp/pr10439_pool_worker.py 2>&1; wc -l /tmp/pr10439_pool_worker.py undefined
          
现在让我读取这些关键文件来分析 bug。先看 coordinator.py 的核心逻辑。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_coordinator.py
          
现在我看到了 coordinator.py 的完整逻辑。关键发现：`load_mask` 调用 `find_longest_cache_hit` 时传入了一个**空的** `ExternalCachedBlockPool()`（`exists=None`）。让我继续看 `kv_transfer.py` 中如何使用 `load_mask`。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_kv_transfer.py
          
现在我看到了关键逻辑。让我读取 `pool_worker.py` 来理解 `load_mask` 的调用上下文，特别是 `_lookup_with_coordinator` 方法。
        
            
toolName: search_by_regex
            
status: success
          
query: _lookup_with_coordinator|load_mask|load_spec|find_longest_cache_hit|cache_coordinator
          
让我读取 `_lookup_with_coordinator` 方法和相关的 load 逻辑。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_pool_worker.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_pool_worker.py
          
让我查看 `config_data.py` 中 `mask_allows_chunk` 和 `load_mask`/`store_mask` 的完整实现。
        
            
toolName: search_by_regex
            
status: success
          
query: mask_allows_chunk|def store_mask|def load_mask|set_cache_coordinator|cache_coordinator
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_config_data.py
          
让我查看 `_chunk_hash_to_bytes` 和 `_block_hash_to_bytes` 的实现，确认 lookup 和 load 之间的 key 一致性。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_pool_worker.py
          

        
            
toolName: search_by_regex
            
status: success
          
query: _block_hash_to_bytes
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_config_data.py
          
让我查看 `block_hashes_for_spec` 和 `find_longest_cache_hit` 的实现，确认 hash 一致性。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_coordinator.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_coordinator.py
          
让我查看 `get_block_size` 的实现和 `process_tokens_with_block_ids` 的 chunk 生成逻辑。
        
            
toolName: search_by_regex
            
status: success
          
query: def get_block_size|def process_tokens_with_block_ids|def process_tokens|group_block_size|cache_family_ratio|def _uses_reachable_mask|def _num_chunks|def _cache_family_granularity
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_config_data.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_config_data.py
          
让我查看测试文件和 `process_tokens_with_block_ids` 的完整实现。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_config_data.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_coordinator.py
          
让我查看 `_reachable_block_mask` 和 `_uses_reachable_mask` 的实现，以及测试文件。
        
            
toolName: search_by_regex
            
status: success
          
query: def _reachable_block_mask|def _uses_reachable_mask|def _num_chunks|def _cache_family_granularity|def _find_longest_cache_hit
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_coordinator.py
          
现在我已经定位到根本原因了。让我查看接收线程中的 load 路径确认问题。
        
            
toolName: search_by_regex
            
status: success
          
query: _load_mask|load_mask|_chunk_mask_allows|mask_allows_chunk
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_kv_transfer.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_kv_transfer.py
          
现在我找到了根本原因。让我确认 `num_prompt_tokens` 在调用点是否可用。
        
            
toolName: search_by_regex
            
status: success
          
query: num_prompt_tokens
          

        
            
toolName: search_by_regex
            
status: success
          
query: num_prompt_tokens
          

        
            
toolName: search_by_regex
            
status: success
          
query: class.*Request|num_prompt|token_len_chunk|block_hashes|block_ids_by_group
          

        
            
toolName: search_by_regex
            
status: success
          
query: request\.|class.*Req|token_len_chunk|can_save|load_spec
          

        
            
toolName: search_by_regex
            
status: success
          
query: num_prompt|num_tokens|prompt_token
          

        
            
toolName: search_by_regex
            
status: success
          
query: class ReqMeta|num_prompt_tokens|token_len_chunk|block_hashes|block_ids_by_group
          

        
            
toolName: search_by_regex
            
status: success
          
query: class ReqMeta|num_prompt_tokens
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_config_data.py
          
我已经定位到根本原因了。让我查看实际仓库中的文件位置。
        
            
toolName: file_search
            
status: success
          
file_pattern: **/ascend_store/coordinator.py
          

        
            
toolName: file_search
            
status: success
          
file_pattern: **/ascend_store/config_data.py
          

        
            
toolName: file_search
            
status: success
          
file_pattern: **/ascend_store/kv_transfer.py
          

        
            
toolName: file_search
            
status: success
          
file_pattern: **/ascend_store/pool_worker.py
          

        
            
toolName: run_command
            
status: success
          
command: cd /Users/fangjianwei/fjw/0623/vllm-ascend && git status && git branch --show-current undefined
          
现在我已经完成了完整的分析。让我确认一下 `store_mask` 和 `load_mask` 的关键差异。
        
            
toolName: view_files
            
status: success
          
            
filePath: /tmp/pr10439_coordinator.py
          
我已经完成了完整的根因分析。下面给你详细讲解这个 bug 的根因和修复方案。

---

## 根因分析

问题的核心在于 **`store_mask` 和 `load_mask` 之间存在不对称**：store 路径应用了 retention mask（保留掩码），但 load 路径没有应用，导致 get 时尝试加载从未被 store 的 chunk。

### 1. Store 路径（PUT）— 正确应用了 retention mask

在 [coordinator.py:155-186](file:///tmp/pr10439_coordinator.py#L155-L186) 的 `store_mask` 中，对非压缩组（`_uses_reachable_mask = True`，即 `None`/`"default"`/`"c1"`）调用了 `_reachable_block_mask`：

```python
mask = _reachable_block_mask(
    manager_cls,
    start_block=0,
    end_block=num_chunks,
    alignment_tokens=self.lcm_block_size,
    kv_cache_spec=spec,
    use_eagle=group_id in self.eagle_group_ids,
    retention_interval=self.retention_interval,
    num_prompt_tokens=num_prompt_tokens,
)
masks.append([True] * num_chunks if mask is None else mask)
```

这个 retention mask 会将一些 block 标记为 `False`（"不可达、不值得传输"），**只有 `True` 的 block 才会被实际 `m_store.put()` 存储**。

### 2. Load 路径（GET）— 没有应用 retention mask

在 [coordinator.py:140-153](file:///tmp/pr10439_coordinator.py#L140-L153) 的 `load_mask` 中：

```python
def load_mask(self, block_hashes, token_len):
    masks, _ = self.find_longest_cache_hit(
        block_hashes,
        token_len,
        ExternalCachedBlockPool(),   # ← exists=None
        apply_eagle=False,
    )
    return tuple(
        [True] * _num_chunks(token_len, ...)
        if not _uses_reachable_mask(...)
        else mask                     # ← 非压缩组直接用 mask
        for group_id, mask in enumerate(masks)
    )
```

关键问题在 `ExternalCachedBlockPool()` 的 [实现](file:///tmp/pr10439_coordinator.py#L24-L39)：

```python
class ExternalCachedBlockPool:
    def __init__(self, exists=None):
        # exists=None is used for load/store masks where hit length has already
        # been decided and each manager only needs to apply its own reachability.
        self._exists = exists

    def get_cached_block(self, block_hash, group_ids):
        if self._exists is None:
            return [self._present_block] * len(group_ids)  # ← 永远返回"存在"
        ...
```

`exists=None` 时，`get_cached_block` **永远返回 present_block**。因此 `find_longest_cache_hit` 认为所有 block 都是命中的，返回的 `mask` **全为 `True`**，**完全没有应用 retention mask**。

### 3. 不对称导致的后果

| 路径 | 行为 | 结果 |
|------|------|------|
| **Store (PUT)** | 应用 `_reachable_block_mask`，部分 block 为 `False` | 只有可达 block 被 `put` 存储 |
| **Load (GET)** | 用 `find_longest_cache_hit(exists=None)`，全为 `True` | 尝试 `get` **所有** block，包括从未存储的 |

在 [pool_worker.py:523](file:///tmp/pr10439_pool_worker.py#L523) 和 [kv_transfer.py:446](file:///tmp/pr10439_kv_transfer.py#L446) 的 load 路径中：

```python
load_masks = self.token_database.load_mask(request.block_hashes, token_len)
...
for start, end, key, _ in self.token_database.process_tokens_with_block_ids(...):
    if not self.token_database.mask_allows_chunk(load_masks, group_id, start):
        continue  # ← mask 全 True，不会跳过任何 chunk
    ...
    key_list.append(key.to_string())
...
self.m_store.get(key_list_c, addr_list_c, size_list_c)  # ← 对不存在的 key 执行 get，失败
```

由于 `load_mask` 返回全 `True`，`mask_allows_chunk` 不会跳过任何 chunk。于是 `m_store.get()` 被要求获取那些 **store 时被 retention mask 过滤掉、从未存储的 key**，导致 get 失败或返回空。

---

## 修复方案

让 `load_mask` 应用与 `store_mask` 相同的 retention mask，确保只加载实际被存储的 chunk。需要修改 4 个文件：

### 修改 1: `coordinator.py` — `load_mask` 应用 `_reachable_block_mask`

```python
def load_mask(
    self,
    block_hashes: list[BlockHash],
    token_len: int,
    num_prompt_tokens: int | None = None,
) -> tuple[list[bool], ...]:
    masks: list[list[bool]] = []
    for group_id, spec in enumerate(self.group_effective_specs):
        num_chunks = _num_chunks(token_len, self.group_effective_block_sizes[group_id])
        if not _uses_reachable_mask(self.group_cache_families[group_id]):
            masks.append([True] * num_chunks)
            continue
        manager_cls = _get_manager_class(_unwrap_spec(self.kv_cache_groups[group_id].kv_cache_spec))
        mask = _reachable_block_mask(
            manager_cls,
            start_block=0,
            end_block=num_chunks,
            alignment_tokens=self.lcm_block_size,
            kv_cache_spec=spec,
            use_eagle=group_id in self.eagle_group_ids,
            retention_interval=self.retention_interval,
            num_prompt_tokens=num_prompt_tokens,
        )
        masks.append([True] * num_chunks if mask is None else mask)
    return tuple(masks)
```

这样 `load_mask` 和 `store_mask` 使用**完全相同的 retention mask 逻辑**，保证 store/load 对称。

### 修改 2: `config_data.py` — `load_mask` 传递 `num_prompt_tokens`

```python
def load_mask(
    self,
    block_hashes: list[BlockHash],
    token_len: int,
    num_prompt_tokens: int | None = None,
) -> tuple[list[bool], ...] | None:
    if self.cache_coordinator is None:
        return None
    return self.cache_coordinator.load_mask(block_hashes, token_len, num_prompt_tokens)
```

### 修改 3: `kv_transfer.py` — `_load_mask` 传递 `num_prompt_tokens`

```python
def _load_mask(self, req_meta, token_len):
    load_mask = getattr(self.token_database, "load_mask", None)
    if load_mask is None:
        return None
    return load_mask(req_meta.block_hashes, token_len, req_meta.num_prompt_tokens)
```

### 修改 4: `pool_worker.py` — `start_load_kv` 传递 `num_prompt_tokens`

```python
load_masks = self.token_database.load_mask(
    request.block_hashes, token_len, request.num_prompt_tokens
)
```

`ReqMeta` 已经有 `num_prompt_tokens` 字段（[config_data.py:670](file:///tmp/pr10439_config_data.py#L670)），`_store_mask` 也已经在用它（[kv_transfer.py:228](file:///tmp/pr10439_kv_transfer.py#L228)），所以调用点数据是现成的。

---

## 总结

| 项目 | 说明 |
|------|------|
| **根因** | `load_mask` 用 `ExternalCachedBlockPool(exists=None)` 调 `find_longest_cache_hit`，返回全 `True` mask，**未应用 retention mask**；而 `store_mask` 应用了 `_reachable_block_mask`，部分 block 为 `False` |
| **现象** | store 只存了可达 block，但 load 尝试 get 所有 block（含未存储的），导致 get 失败 |
| **修复** | 让 `load_mask` 复用 `store_mask` 的 `_reachable_block_mask` 逻辑，保证 store/load mask 对称 |
| **影响范围** | `coordinator.py`、`config_data.py`、`kv_transfer.py`、`pool_worker.py` 四个文件 |

需要我直接把这些修改应用到 PR 分支上吗？
