from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import json,sys,hashlib,time
from baseline_inference.foveal_engine import FovealLLM
from benchmarks.verify_foveal_cache import verify,verify_greedy
model,out=sys.argv[1],Path(sys.argv[2]);e=FovealLLM(model,device='cuda');e._decode_step=e._decode_step_graph
start=time.perf_counter()
r={'model':model,'backend':'FovealLLM','decode_backend':'cuda_graph','decoder_sources_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['baseline_inference/foveal_engine.py','baseline_inference/foveal_decode.py','baseline_inference/foveal_triton.py']}}
r['cases']=verify(e,[65,641,8192],66,atol=1e-4,rtol=1e-4,controls=True,workload='prose',compare_every=16)
r['greedy_cases']=verify_greedy(e,[65,641],130)
r['elapsed_s']=time.perf_counter()-start
out.write_text(json.dumps(r,indent=2)+'\n');e.close()
