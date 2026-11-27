"""D missing-window and process-deadline checks on archived inputs, zero steps."""
import argparse,json,sys,time,platform,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.monitoring.b05_model import B05Model
from l3_system.monitoring.inference_service import InferenceService
from deliverables.D05.scripts.fit_b05_models import write

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.dataset/'manifest.json').read_text());freeze=json.loads((a.models/'model_freeze.json').read_text());missing=[];performance={}
 for env,receipt in freeze['models'].items():
  model=B05Model.load_trusted(a.models/receipt['path'],receipt['sha256']);example=None
  for e in manifest['episodes']:
   if e['environment_id']!=env:continue
   samples=[json.loads(s) for s in (a.dataset/e['episode_id']/'samples.jsonl').read_text().splitlines()]
   for end in range(3,len(samples)):
    window=[s['delivered_observation'] for s in samples[end-3:end+1]]
    if e['split']=='train' and example is None:example=window
    if any(v is None for row in window for v in row):
     result=model.score_window(window)
     if not result['automatic_execution_blocked'] or result['status']!='unavailable':raise ValueError('missing window escaped gate')
     missing.append({'episode_id':e['episode_id'],'end_record':end+1,'environment':env,'result':result})
  packet={'binding':model.binding,'episode_id':'archived-performance-fixture','state_versions':[1,2,3,4],'values':example}
  context={'binding':model.binding,'episode_id':'archived-performance-fixture','state_version':4,'observed_monotonic':0.}
  before=time.perf_counter()
  with InferenceService(a.models/receipt['path'],receipt['sha256'],timeout_seconds=.5) as service:
   startup=time.perf_counter()-before;times=[]
   for index in range(100):
    context['observed_monotonic']=time.monotonic();result=service.score(str(index),packet,context)
    if result['status']!='scored':raise ValueError('process inference failed: '+str(result))
    if result['execution_authorized'] is not False:raise ValueError('scorer authorized execution')
    times.append(result['wall_seconds'])
   duplicate=service.score('99',packet,context)
   expired=service.score('past',packet,context,absolute_deadline=time.monotonic()-1)
   context['observed_monotonic']=time.monotonic()-1;stale=service.score('stale',packet,context)
   if not all(x['automatic_execution_blocked'] for x in (duplicate,expired,stale)):raise ValueError('negative gate escaped')
  performance[env]={'model_worker_startup_seconds':startup,'hot_calls':len(times),'mean_s':float(np.mean(times)),
   'p95_s':float(np.percentile(times,95)),'max_s':max(times),'deadline_seconds':.5,'deadline_overruns':sum(t>.5 for t in times),
   'scope':'archived training window repeated, IPC plus numeric model; no B execution, no distributed network',
   'duplicate_blocked':True,'expired_blocked':True,'stale_blocked':True}
 if len(missing)!=56:raise ValueError('original missing-window denominator differs')
 a.output.mkdir(parents=True,exist_ok=False)
 write(a.output/'missing_window_ledger.json',missing);write(a.output/'budget_verification.json',{'schema_version':'DBudgetVerification/1.0',
  'performance':performance,'python':sys.version,'host':platform.platform(),'new_simulator_steps':0,
  'missing_windows_planned':56,'missing_windows_blocked':len(missing),'joint_budget_approved':False,'formal_blind_started':False})
 print(json.dumps(performance))
if __name__=='__main__':main()
