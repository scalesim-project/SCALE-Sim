"""Validate the full real-weight FP32 decoder against Hugging Face on 512 tokens."""
import json,time
from pathlib import Path
import torch
from transformers import Gemma3ForCausalLM
from numerical import Decoder
from run_numerical import DEFAULT

HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    ids=json.loads((HERE/'build/numerical_512/prompt_ids.json').read_text())
    print('Loading official Hugging Face numerical reference',flush=True)
    hf=Gemma3ForCausalLM.from_pretrained(str(DEFAULT),local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    own=Decoder.load(DEFAULT,False,False);checks=[]
    with torch.inference_mode():
        ref=hf(torch.tensor([ids]),use_cache=True,logits_to_keep=1)
        got=own.forward(ids);want=ref.logits[0,-1]
        torch.testing.assert_close(got,want,atol=2e-4,rtol=2e-4)
        checks.append(dict(phase='prefill_512',max_abs_error=float((got-want).abs().max()),argmax_equal=int(got.argmax())==int(want.argmax())))
        # Use the same actual INT8-generated input token for both FP32 references.
        token=json.loads((HERE/'build/numerical_512/result.json').read_text())['decode_input_token_ids'][0]
        ref=hf(torch.tensor([[token]]),past_key_values=ref.past_key_values,use_cache=True,logits_to_keep=1)
        got=own.forward([token]);want=ref.logits[0,-1]
        torch.testing.assert_close(got,want,atol=2e-4,rtol=2e-4)
        checks.append(dict(phase='decode_context_513',max_abs_error=float((got-want).abs().max()),argmax_equal=int(got.argmax())==int(want.argmax())))
    report=dict(status='passed',checkpoint=str(DEFAULT),checks=checks)
    (HERE/'build/checkpoint_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
