"""Execute the bounded fusion on real KV caches and compare saved logits exactly."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import numerical
from numerical import Decoder
from fused_value_cache import requantize_value_cache
from tiled_gemm_reference import execute_tiled
from config import Architecture

HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    folder=HERE/'build/numerical_512'
    baseline=json.loads((folder/'result.json').read_text())
    checkpoint=Path(baseline['checkpoint'])/'model.safetensors'
    digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    assert digest==baseline['checkpoint_sha256']
    original_quantize=numerical.quantize
    checks=[];matrix_checks=[]
    arch=Architecture.read(HERE/'configs/decode_100tps.json')
    original_gemm=numerical.int_gemm
    class FusedDecoder(Decoder):
        def attention(self,p,q,k,v,positions,layer):
            self.fused_layer=layer
            self.matrix_layout='attention';self.attention_matrix_index=0
            return super().attention(p,q,k,v,positions,layer)
        def linear(self,name,x,key):
            self.matrix_layout='packed'
            return super().linear(name,x,key)
    print('Loading real checkpoint for fused-cache validation',flush=True)
    model=FusedDecoder.load(checkpoint.parent,True,False)
    def quantize(x,dim=-1):
        if dim!=0:return original_quantize(x,dim)
        _,_,values,scales=model.cache[model.fused_layer]
        values=values[-len(x):,0,:];scales=scales[-len(x):,0,:]
        actual,actual_scale=requantize_value_cache(values,scales,83)
        expected,expected_scale=original_quantize(x,0)
        assert torch.equal(actual,expected) and torch.equal(actual_scale,expected_scale)
        checks.append(dict(layer=model.fused_layer,context=len(x),bit_exact=True))
        return actual,actual_scale
    numerical.quantize=quantize
    logits=model.forward(json.loads((folder/'prompt_ids.json').read_text()))
    np.testing.assert_array_equal(logits.numpy(),np.load(folder/'prefill_logits_int8.npy'))
    print('Fused numerical prefill logits match saved baseline bit for bit',flush=True)
    def checked_gemm(a,b):
        layout=model.matrix_layout
        if layout=='attention':
            layout='key_transpose' if model.attention_matrix_index%2==0 else 'row_major'
            model.attention_matrix_index+=1
        actual=execute_tiled(a,b,arch,layout)
        expected=original_gemm(a,b)
        assert torch.equal(actual,expected)
        matrix_checks.append(dict(m=a.shape[0],k=a.shape[1],n=b.shape[1],layout=layout,bit_exact=True))
        return actual
    numerical.int_gemm=checked_gemm
    for step,token in enumerate(baseline['decode_input_token_ids']):
        logits=model.forward([token])
        np.testing.assert_array_equal(logits.numpy(),np.load(folder/f'decode_{step}_logits_int8.npy'))
        assert int(logits.argmax())==baseline['decode_output_token_ids'][step]
        print(f'Decode {step}: all 262144 logits match exactly',flush=True)
    output=dict(status='passed',checkpoint_sha256=digest,cache_checks=checks,
                prefill_logits_bit_exact=True,decode_logits_bit_exact=baseline['decode_steps'],
                output_tokens=baseline['decode_output_token_ids'],
                tiled_gemm_checks=matrix_checks,configuration=arch.to_dict(),
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (HERE/'numerical.py',HERE/'fused_value_cache.py',HERE/'tiled_gemm_reference.py',HERE/'workload.py',HERE/'config.py',Path(__file__))})
    (HERE/'build/decode_fusion_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print('PASS real-checkpoint fused numerical execution',flush=True)

if __name__=='__main__':main()
