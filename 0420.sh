pytest --cov=vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store \
         --cov-report=term-missing \
         tests/ut/distributed/ascend_store/ \
         -v

PYTHONPATH="$PWD:$PYTHONPATH" pytest --noconftest \
    --cov=vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store \                                                                                                                                           
    --cov-report=json:cov_ascend_store.json \                                                                                                                                                                
    tests/ut/distributed/ascend_store/ -q    
