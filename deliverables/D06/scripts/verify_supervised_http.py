"""Real SMPL + D risk model + Flask HTTP contract; no browser-render claim."""
import argparse,json,re
from pathlib import Path
from l3_system.execution import EnvironmentAdapter
from l3_system.monitoring.residual_detector import ResidualDetector
from l3_system.monitoring.supervised_session import SupervisedSession
from l3_system.monitoring.supervision_console import create_app

def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 frozen=json.loads((a.frozen/'freeze.json').read_text());model=ResidualDetector.load_trusted(a.frozen/'model.pkl',frozen['model_sha256']);events=[]
 with EnvironmentAdapter(max_steps=8) as b:
  s=SupervisedSession(b,model,a.output/'session');s.start();client=create_app(s).test_client()
  def form():
   html=client.get('/').get_data(as_text=True)
   d={k:re.search('name="'+k+'" value="([^"]+)"',html).group(1) for k in ('token','revision')}
   return dict(d,operator='D simulated HTTP operator',reason='explicit simulator acceptance')
  def post(route,expected=303,data=None):
   before=b.step_calls;res=client.post(route,data=form() if data is None else data)
   assert res.status_code==expected,(route,res.status_code,expected)
   events.append({'route':route,'status':res.status_code,'step_delta':b.step_calls-before,'state':s.status()})
   return res
  post('/cycle',409);assert b.step_calls==0
  post('/sample',403,data={})
  for _ in range(6):post('/sample')
  assert b.step_calls==6
  confirmed=form();post('/confirm',data=confirmed);assert b.step_calls==6
  post('/confirm',409,data=confirmed);post('/confirm',409)
  post('/cycle');assert b.step_calls==7
  post('/probe');assert b.step_calls==7 and s.faulted
  post('/confirm',409);post('/cycle',409);post('/sample',409)
  old=s.episode;post('/restart');assert s.episode!=old and s.samples==0 and b.step_calls==7
  for _ in range(6):post('/sample')
  post('/confirm');assert b.step_calls==13
  post('/cycle');assert b.step_calls==14
  for _ in range(3):
   if s.faulted:break
   post('/cycle')
  assert s.faulted and b.done;ended=b.step_calls
  post('/confirm',409);post('/cycle',409);assert b.step_calls==ended
  report={'schema_version':'D05HTTPAcceptance/1.0','passed':True,'real_SMPL':True,'risk_model_sha256':frozen['model_sha256'],
          'events':events,'total_step_calls':b.step_calls,'browser_render_verified':False,'operator_simulated':True,
          'browser_blocker':'Local Chromium launch syscall denied; cloud localhost connection refused and subsequent browser URL policy rejected. No bypass attempted.',
          'physical_safety_certified':False,'new_episode_recovery_not_same_episode':True}
 (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'passed':True,'http_checks':len(events),'real_steps':ended,'browser_render_verified':False}))
if __name__=='__main__':main()
