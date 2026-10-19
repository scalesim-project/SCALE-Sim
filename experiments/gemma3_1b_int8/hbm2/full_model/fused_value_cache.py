"""Bounded two-pass reference for the fused V-cache timing operation."""
import torch


def requantize_value_cache(values, token_scales, chunk_rows=83):
    """Preserve FP32 multiplication, division and torch round-to-even exactly.

    Only a row chunk is dequantized at once. Feature maxima/scales persist
    across chunks. The INT8 output represents the HBM destination.
    """
    if values.dtype != torch.int8 or values.ndim != 2 or values.shape[0] == 0:
        raise ValueError('Expected a nonempty INT8 context-by-feature tensor')
    if token_scales.shape != (values.shape[0], 1) or token_scales.dtype != torch.float32:
        raise ValueError('Expected one FP32 scale per context row')
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    maximum=torch.zeros((1,values.shape[1]),dtype=torch.float32)
    for start in range(0,len(values),chunk_rows):
        block=values[start:start+chunk_rows].float()*token_scales[start:start+chunk_rows]
        maximum=torch.maximum(maximum,block.abs().amax(0,keepdim=True))
    scales=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    output=torch.empty_like(values)
    for start in range(0,len(values),chunk_rows):
        block=values[start:start+chunk_rows].float()*token_scales[start:start+chunk_rows]
        output[start:start+chunk_rows]=torch.round(block/scales).clamp(-127,127).to(torch.int8)
    return output,scales
