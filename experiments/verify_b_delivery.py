"""Execute public B acceptance in fresh output paths. Never starts a holdout."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--development',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 root=Path(__file__).resolve().parents[1]
 jobs=[('unit',['-m','unittest','discover','-s','l3_system/tests']),('C08',['-m','unittest','discover','-s','deliverables/C08/tests']),('C10',['-m','unittest','discover','-s','deliverables/C10/tests']),('atropine',['-m','experiments.run_validation','--output',str(a.output/'atropine')]),('session',['-m','experiments.validate_session','--output',str(a.output/'session')]),('mab_receipts',['-m','experiments.verify_b_freeze_receipts','--output',str(a.output/'mab_receipts')]),('gateway',['-m','experiments.verify_joint_gateway','--output',str(a.output/'gateway')]),('C_native',['-m','deliverables.C10.live','--out',str(a.output/'C_native')]),('docs',['-m','experiments.check_docs'])]
 if a.development:jobs.append(('development',['-m','experiments.run_b08_development','--output',str(a.output/'development'),'--trusted-startup-development']))
 rows=[]
 for name,args in jobs:
  start=time.monotonic()
  with (a.output/(name+'.log')).open('w') as f:
   try:r=subprocess.run([sys.executable,*args],cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=240);code=r.returncode
   except subprocess.TimeoutExpired:code=124
  rows.append({'name':name,'returncode':code,'wall_seconds':time.monotonic()-start})
  (a.output/'summary.json').write_text(json.dumps({'checks':rows,'passed':all(r['returncode']==0 for r in rows),'holdout_started':False,'joint_signoff_complete':False},indent=2)+'\n')
  print(name,code,flush=True)
 return 0 if all(r['returncode']==0 for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
