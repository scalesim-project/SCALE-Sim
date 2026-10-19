"""Expose remaining differences from the unmodified SCALE-Sim memory policy.

These are deliberately NOT presented as equivalent hardware configurations:
CALC/USER have abstract word buffers/prefetch and no serial AXI DMA. Comparing
normalized demand-row commit times identifies actual policy mismatches even
after removing the initial DMA load interval.
"""
import numpy as np
from scalesim.scale_config import scale_config
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad


def compare(shape,rows,cols,rtl_commits):
    m,k,n=shape
    cfg=scale_config();cfg.valid_conf_flag=True;cfg.array_rows=rows;cfg.array_cols=cols
    engine=systolic_compute_ws()
    engine.set_params(config_obj=cfg,ifmap_op_mat=np.arange(m*k).reshape(m,k),
                      filter_op_mat=np.arange(k*n).reshape(k,n),ofmap_op_mat=np.arange(m*n).reshape(m,n))
    demands=engine.get_demand_matrices();prefetch=engine.get_prefetch_matrices()
    results={}
    for mode in ('CALC','USER'):
        memory=double_buffered_scratchpad()
        memory.set_params(verbose=False,estimate_bandwidth_mode=mode=='CALC',word_size=1,
                          ifmap_buf_size_bytes=65536,filter_buf_size_bytes=65536,ofmap_buf_size_bytes=16384,
                          rd_buf_active_frac=0.5,wr_buf_active_frac=0.5,
                          ifmap_backing_buf_bw=32,filter_backing_buf_bw=32,ofmap_backing_buf_bw=8,
                          ifmap_sram_bank_num=rows,filter_sram_bank_num=cols,
                          ifmap_sram_bank_port=1,filter_sram_bank_port=1,config=cfg)
        if mode=='USER':memory.set_read_buf_prefetch_matrices(*prefetch)
        memory.service_memory_requests(*demands)
        a,b,c=memory.get_sram_trace_matrices()
        # This is the joint service edge used by native stall accumulation,
        # with nominal read hit latency removed, exactly as upstream does.
        joint=np.maximum.reduce([a[:,0]-memory.ifmap_buf.get_hit_latency(),
                                 b[:,0]-memory.filter_buf.get_hit_latency(),c[:,0]]).astype(np.int64).tolist()
        assert len(joint)==len(rtl_commits)
        mismatches=[i for i,(x,y) in enumerate(zip(joint,rtl_commits)) if x!=y]
        first=mismatches[0] if mismatches else None
        results[mode]=dict(equivalent=not mismatches,demand_rows=len(joint),different_commit_edges=len(mismatches),
                           first_mismatch=None if first is None else dict(logical_row=first,native_edge=joint[first],rtl_edge=rtl_commits[first]),
                           native_report_last_compute_cycle=memory.get_total_compute_cycles(),
                           native_stall_cycles=memory.get_stall_cycles(),rtl_last_commit_edge=rtl_commits[-1],
                           native_joint_commit_edges=joint)
    return dict(shape=list(shape),array=[rows,cols],modes=results,
                interpretation='Compute-relative edges; initial RTL DMA load excluded. Native CALC/USER do not model this AXI/HBM system.',
                native_settings=dict(word_bytes=1,ifmap_filter_abstract_capacity=65536,ofmap_abstract_capacity=16384,
                                     active_fraction=0.5,ifmap_filter_backing_words_per_cycle=32,ofmap_backing_words_per_cycle=8,
                                     layout_evaluation=False,ramulator_trace=False),
                policy_gaps=['Native subtracts nominal SRAM read hit latency; RTL pays synchronous bank gather latency.',
                             'Native prefetch/active windows differ from whole-tile A-then-B DMA loading.',
                             'Native drains buffered ofmaps; RTL queues packets and writes final INT32 words with AXI B completion.',
                             'Native CALC/USER backing ports here are not causal Ramulator HBM2 controllers.'])
