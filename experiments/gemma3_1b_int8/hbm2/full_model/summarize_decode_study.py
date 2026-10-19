"""Verify completed stage artifacts and retain a compact, measured comparison."""
import hashlib
import json
from pathlib import Path
import unittest

from test_configuration import ConfigurationChecks
from test_decode_architecture import DecodeArchitectureChecks

HERE=Path(__file__).resolve().parent

def main():
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                             for cls in (ConfigurationChecks,DecodeArchitectureChecks))
    tested=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    fusion=json.loads((HERE/'build/decode_fusion_validation.json').read_text())
    assert fusion['status']=='passed'
    for name,digest in fusion['sources'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    reports=[]
    for name in ('fast_decode','decode_os','decode_overlap','decode_100tps'):
        folder=HERE/'build'/name;r=json.loads((folder/'result.json').read_text())
        assert r['status']=='passed' and r['assumed_core_clock_hz']==1000000000 and r['decode_steps']==3
        if name!='fast_decode':
            execution=json.loads((folder/'execution.json').read_text());assert execution['status']=='completed'
            for file,digest in execution['inputs_sha256'].items():
                assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==digest
            assert hashlib.sha256((folder/'timing.csv').read_bytes()).hexdigest()==execution['timing_sha256']
            assert json.loads((folder/'timing_tests.json').read_text())['status']=='passed'
            for path,digest in r['source_sha256'].items():
                assert hashlib.sha256((HERE.parents[3]/path).read_bytes()).hexdigest()==digest,path
        if reports:
            assert r['numerical_trace_sha256']==reports[0]['numerical_trace_sha256']
            assert r['hbm_config_sha256']==reports[0]['hbm_config_sha256']
        reports.append(r)
    rows=[]
    for r in reports:
        steps=r['decode_steps'];c=r['configuration']
        rows.append(dict(dataflow=c.get('dataflow','ws'),array=[c['array_rows'],c['array_cols']],
                         vector_lanes=c['vector_lanes'],gemm_prefetch=c.get('gemm_prefetch',False),
                         fuse_value_cache=c.get('fuse_value_cache',False),
                         total_staging_KiB=(2*c['operand_staging_bytes']+c['vector_staging_bytes'])/1024,
                         decode_tps=r['mean_decode_tps'],milliseconds=r['mean_decode_ms'],
                         compute_ms=sum(t['compute_cycles'] for t in r['tokens'])/steps/1e6,
                         exposed_memory_wait_ms=sum(t['memory_wait_cycles'] for t in r['tokens'])/steps/1e6,
                         physical_HBM_bytes_per_token=sum(t['dram_transaction_bytes'] for t in r['tokens'])/steps))
    final=reports[-1]
    assert final['mean_decode_tps']>100 and all(t['tps']>100 for t in final['tokens'])
    result=dict(status='passed',target_tps=100,configuration_and_architecture_tests=tested.testsRun,
                real_checkpoint_fusion_validation=fusion,measurements=rows,
                speedup_over_41tps=final['mean_decode_tps']/reports[0]['mean_decode_tps'],
                mac_count_reduction=256,vector_lane_reduction=4,
                scope='Batch 1, 512-token prompt, contexts 513-515, three cached greedy decode steps; simulated 1 GHz, native HBM2 timing, separate real numerical execution.',
                numerical_trace_sha256=final['numerical_trace_sha256'],hbm_config_sha256=final['hbm_config_sha256'])
    (HERE/'decode_100tps_study.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'decode_100tps_result.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
