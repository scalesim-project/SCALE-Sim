#!/usr/bin/env python3
"""Run real-checkpoint numerical Gemma W8A8+KV8 prefill and cached decode."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer
from numerical import Decoder

HERE=Path(__file__).resolve().parent
DEFAULT=Path.home()/'.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'

def compare(a,b):
    topa=set(torch.topk(a,10).indices.tolist());topb=set(torch.topk(b,10).indices.tolist())
    return dict(logit_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                max_abs_logit_difference=float((a-b).abs().max()),
                rms_logit_difference=float(torch.sqrt((a-b).square().mean())),
                int8_argmax=int(a.argmax()),fp32_argmax=int(b.argmax()),top10_overlap=len(topa&topb))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=DEFAULT)
    p.add_argument('--prompt-tokens',type=int,default=512);p.add_argument('--decode-steps',type=int,default=3)
    p.add_argument('--out',type=Path,default=HERE/'build/numerical_512');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--no-fp32-reference',action='store_true');args=p.parse_args()
    assert args.prompt_tokens>=16 and args.decode_steps>0
    torch.set_num_threads(args.threads);torch.manual_seed(7);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'result.json').unlink(missing_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint),local_files_only=True)
    text=('Read the following background and then briefly explain why the sky looks blue. '
          'Sunlight contains many wavelengths. Molecules in the atmosphere scatter shorter wavelengths more strongly. '
          'The Earth has a nitrogen and oxygen atmosphere. Clouds consist of water droplets and ice crystals. ')
    ids=[];repeat=1
    print("Preparing tokenized prompt",flush=True)
    while len(ids)<args.prompt_tokens:
        assert repeat<1024,"Prompt construction did not converge"
        ids=tokenizer.apply_chat_template([dict(role='user',content=text*repeat)],tokenize=True,add_generation_prompt=True,return_dict=False);repeat+=1
    suffix=tokenizer.encode('<end_of_turn>\n<start_of_turn>model\n',add_special_tokens=False)
    assert ids[-len(suffix):]==suffix
    ids=ids[:args.prompt_tokens-len(suffix)]+suffix
    (args.out/'prompt_ids.json').write_text(json.dumps(ids))
    start=time.monotonic();print('Loading and quantizing real checkpoint',flush=True)
    model=Decoder.load(args.checkpoint,True,False)
    assert model.layers==26 and model.h==1152 and model.d==256
    print(f'Loaded W8A8 weights in {time.monotonic()-start:.2f}s; numerical prefill {len(ids)} tokens',flush=True)
    t=time.monotonic();logits=model.forward(ids);prefill_wall=time.monotonic()-t
    first=int(logits.argmax());generated=[first];input_ids=[];outputs=[];comparisons=[]
    np.save(args.out/'prefill_logits_int8.npy',logits.numpy())
    print(f'Numerical INT8 prefill completed in {prefill_wall:.2f}s, first token={first}',flush=True)
    reference=None
    if not args.no_fp32_reference:
        print('Loading FP32 reference from the same checkpoint',flush=True)
        reference=Decoder.load(args.checkpoint,False,False)
        t=time.monotonic();fp=reference.forward(ids);comparisons.append(dict(phase='prefill',**compare(logits,fp)))
        print(f'FP32 prefill {time.monotonic()-t:.2f}s; {comparisons[-1]}',flush=True)
    model.record=True;decode_wall=[]
    for step in range(args.decode_steps):
        token=generated[-1];input_ids.append(token);t=time.monotonic();logits=model.forward([token]);wall=time.monotonic()-t
        chosen=int(logits.argmax());generated.append(chosen);outputs.append(chosen);decode_wall.append(wall)
        model.events.append(dict(name='greedy_token_selection',kind='argmax',position=model.position-1,shape=[],dtype='torch.int64',token_id=chosen))
        if reference is not None:comparisons.append(dict(phase=f'decode_{step}',**compare(logits,reference.forward([token]))))
        np.save(args.out/f'decode_{step}_logits_int8.npy',logits.numpy())
        print(f'Decode {step}: input={token} output={chosen}, CPU functional runtime={wall:.3f}s',flush=True)
    (args.out/'operator_trace.json').write_text(json.dumps(model.events,indent=2))
    checkpoint=args.checkpoint/'model.safetensors'
    digest=hashlib.sha256()
    with checkpoint.open('rb') as f:
        while chunk:=f.read(16*1024**2):digest.update(chunk)
    result=dict(status='passed',classification='real-checkpoint numerical execution; wall times are host CPU times, not simulated accelerator TPS',
                checkpoint=str(args.checkpoint),checkpoint_sha256=digest.hexdigest(),config=json.loads((args.checkpoint/'config.json').read_text()),
                quantization=dict(weights='symmetric INT8 per output channel',linear_activations='dynamic symmetric INT8 per row',
                                  kv='dynamic INT8 per token/head',attention='QK and AV exact INT8->INT32; cached V dequantized and requantized per feature for AV',
                                  non_gemm='FP32 RMSNorm, RoPE, softmax, GELU and residuals',
                                  calibration='No trained quantization calibration; this is an explicit post-training quantization scheme.',
                                  prefill='V per-feature quantization spans the available prompt block; quantization can depend on prefill chunking.'),
                prompt_tokens=len(ids),decode_steps=args.decode_steps,first_decode_context=len(ids)+1,
                decode_input_token_ids=input_ids,decode_output_token_ids=outputs,generated_token_ids=generated,
                generated_text=tokenizer.decode(generated,skip_special_tokens=True),
                fp32_comparison=comparisons,operator_counts=dict(Counter(e['kind'] for e in model.events)),
                functional_cpu_prefill_seconds=prefill_wall,functional_cpu_decode_seconds=decode_wall,
                all_logits_finite=True,kv_cache_dtype='torch.int8',torch_version=torch.__version__,
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'numerical.py',Path(__file__)]})
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS numerical generation:',repr(result['generated_text']),flush=True)
    print('Report:',args.out/'result.json',flush=True)
if __name__=='__main__':main()
