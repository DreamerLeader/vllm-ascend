if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                import os
                _dump_dir = os.environ.get(
                    "DSA_SAS_METADATA_DUMP_DIR",
                    "/tmp/dsa_sas_metadata_dump_c4",
                )
                os.makedirs(_dump_dir, exist_ok=True)
                _tensor_args = {
                    "cu_seqlens_q": prefill_query_start_loc,
                    "cu_seqlens_ori_kv": prefill_query_start_loc,
                    "cu_seqlens_cmp_kv": cu_c4_cmp_seqlen_list,
                    "seqused_q": self.seqused_q,
                    "seqused_kv": self.seq_lens[reqs_start:],
                    "max_seqlen_q": seq_lens_q.max(),
                    "max_seqlen_kv": self.seq_lens[reqs_start:].max(),
                }
                _scalar_args = {
                    "num_heads_q": n_local_heads,
                    "num_heads_kv": 1,
                    "head_dim": self.model_config.get_head_size(),
                    "batch_size": len(self.seq_lens[reqs_start:]),
                    "cmp_topk": index_topk,
                    "cmp_ratio": 4,
                    "ori_mask_mode": 4,
                    "cmp_mask_mode": 3,
                    "ori_win_left": self.model_config.hf_config.sliding_window - 1,
                    "ori_win_right": 0,
                    "layout_q": "TND",
                    "layout_kv": "PA_ND",
                    "has_ori_kv": True,
                    "has_cmp_kv": True,
                    "device": str(self.seqused_q.device),
                }
                print("[DSA_DUMP] npu_sparse_attn_sharedkv_metadata (compressor_ratio=4) inputs:")
                for _name, _t in _tensor_args.items():
                    if _t is None:
                        print(f"[DSA_DUMP]   tensor {_name}: None")
                        continue
                    _save_path = os.path.join(_dump_dir, f"{_name}.pt")
                    torch.save(_t.detach().cpu(), _save_path)
                    print(
                        f"[DSA_DUMP]   tensor {_name}: shape={tuple(_t.shape)}, "
                        f"dtype={_t.dtype}, device={_t.device} -> saved to {_save_path}"
                    )
                _scalar_path = os.path.join(_dump_dir, "scalar_args.txt")
                with open(_scalar_path, "w") as _f:
                    for _name, _v in _scalar_args.items():
                        _line = f"{_name} (type={type(_v).__name__}): {_v}"
                        _f.write(_line + "\n")
                        print(f"[DSA_DUMP]   scalar {_line}")
                print(f"[DSA_DUMP] scalar args saved to {_scalar_path}")
