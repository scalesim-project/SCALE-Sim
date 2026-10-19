"""Independent Hugging Face reference and exact integer arithmetic checks."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import Gemma3TextConfig,Gemma3ForCausalLM
from numerical import Decoder,quantize,int_gemm

def main():
    torch.set_num_threads(4);torch.manual_seed(18)
    checks=[]
    for shape in [(1,1152,32),(4,513,256),(17,33,7)]:
        m,k,n=shape;a=torch.randint(-127,128,(m,k),dtype=torch.int8);b=torch.randint(-127,128,(k,n),dtype=torch.int8)
        got=int_gemm(a,b);expected=a.numpy().astype(np.int64)@b.numpy().astype(np.int64)
        np.testing.assert_array_equal(got.numpy(),expected);checks.append(dict(check='exact_int32_gemm',shape=shape))
    zeros,scale=quantize(torch.zeros(3,19));assert not zeros.any() and torch.equal(scale,torch.ones_like(scale))
    cfg=Gemma3TextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=6,num_attention_heads=4,
                        num_key_value_heads=1,head_dim=16,vocab_size=256,sliding_window=8,
                        query_pre_attn_scalar=16,max_position_embeddings=128)
    cfg._attn_implementation='eager';hf=Gemma3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name,p in hf.named_parameters():
            if 'norm' in name:p.copy_(0.1*torch.randn_like(p))
    state=hf.state_dict();tokens=torch.randint(0,256,(1,21))
    model=Decoder(cfg.to_dict(),state,False)
    with torch.inference_mode():
        expected=hf(tokens,logits_to_keep=1,use_cache=True)
        actual=model.forward(tokens[0].tolist())
        error=float((actual-expected.logits[0,-1]).abs().max());assert error<2e-5,error
        checks.append(dict(check='fp32_prefill_matches_hf_local_global_masks',max_abs_error=error))
        past=expected.past_key_values
        for token in [37,19,241]:
            expected=hf(torch.tensor([[token]]),past_key_values=past,logits_to_keep=1,use_cache=True)
            past=expected.past_key_values;actual=model.forward([token])
            error=float((actual-expected.logits[0,-1]).abs().max());assert error<2e-5,error
            checks.append(dict(check='fp32_cached_decode_matches_hf',max_abs_error=error))
    q=Decoder(cfg.to_dict(),state,True);got=q.forward(tokens[0].tolist())
    assert torch.isfinite(got).all() and all(v[0].dtype==torch.int8 and v[2].dtype==torch.int8 for v in q.cache.values())
    float_ref=Decoder(cfg.to_dict(),state,False).forward(tokens[0].tolist())
    cosine=float(torch.nn.functional.cosine_similarity(got,float_ref,dim=0));assert cosine>.98,cosine
    for token in [37,19,241]:assert torch.isfinite(q.forward([token])).all()
    kinds={e['kind'] for e in q.events}
    assert {'embedding','rmsnorm','gemm','quantize','dequantize','rope_apply','kv_write','softmax','attention_mask','gelu','gating','residual'}<=kinds
    checks.append(dict(check='complete_w8a8_kv8_graph_finite',cosine_to_fp32=cosine,operator_kinds=sorted(kinds)))
    result=dict(status='passed',checks=checks)
    out=Path(__file__).parent/'build/numerical_tests.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
