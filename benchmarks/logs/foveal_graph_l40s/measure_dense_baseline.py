from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import json,sys,time,torch
from benchmarks.model import EvalModel
from benchmarks.serving import _prompt_ids
from baseline_inference.engine import NoPrefixBlockManager
core=sys.argv[1];out=Path(sys.argv[2])
entry=next(x for x in json.loads(Path('benchmarks/logs/foveal_verified/base_checkpoints.json').read_text()) if x['core']==core)
e=EvalModel(entry['path'],strict=True,quiet=True,max_num_seqs=1,max_model_len=2304,max_num_batched_tokens=4096,gpu_memory_utilization=.2)
e.load();engine=getattr(e._llm, "engine", e._llm)
# Identical prompts must use fresh recurrent and KV state, including Polar.
engine.scheduler.block_manager=NoPrefixBlockManager(engine.config.num_kvcache_blocks,engine.config.kvcache_block_size)
prompt=_prompt_ids(engine.tokenizer,2048)
report={'core':core,'model':entry,'backend':e.backend,'context_tokens':2048,'attn_window':e.hf_config.attn_window,'generated_tokens':130,'ignore_eos':True,'repetitions':[],'warmups':[]}
for i in range(4):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 e.generate([prompt],max_tokens=130,ignore_eos=True)
 torch.cuda.synchronize();r=dict(e.last_call_metrics)
 assert r['decode_tokens']==129 and r['prefill_tokens']==2048
 r['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
 report['warmups' if i==0 else 'repetitions'].append(r)
 print(core,i,r,flush=True)
out.write_text(json.dumps(report,indent=2)+'\n');e.close()
