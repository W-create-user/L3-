"""D08 paired episode diagnostics for the frozen sensor-only holdout."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import beta

def interval(k,n):
 return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def analyze(source):
 source=Path(source);r=json.loads(source.read_text());new=r['result']['episodes'];old=r['baseline']['episodes']
 assert [e['episode_id'] for e in new]==[e['episode_id'] for e in old]
 differences=[];rows=[]
 for a,b in zip(new,old):
  n=a['counts']['FP']+a['counts']['TN'];assert n==b['counts']['FP']+b['counts']['TN']
  nf=a['counts']['FP']/n if n else None;of=b['counts']['FP']/n if n else None
  if n:differences.append(of-nf)
  rows.append({'episode_id':a['episode_id'],'normal_windows':n,'baseline_fpr':of,'residual_fpr':nf,'fpr_reduction':of-nf if n else None})
 rng=np.random.default_rng(4810);d=np.array(differences)
 boot=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(5000)])
 ev=r['result']['events'];n=r['result']['whole_normal_episodes'];k=r['result']['whole_normal_with_alarm']
 return {'schema_version':'D08SensorEpisodeAnalysis/1.0','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'paired_episodes':len(rows),'rows':rows,'mean_episode_normal_window_fpr_reduction':float(d.mean()),
  'paired_episode_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_seed':4810,'bootstrap_samples':5000,
  'conditional_binomial_intervals':{'raw_event_detection':{'k':ev['detected_event_count'],'n':ev['scorable_event_count'],'interval_95':interval(ev['detected_event_count'],ev['scorable_event_count'])},
                                  'whole_normal_false_alarm':{'k':k,'n':n,'interval_95':interval(k,n)}},
  'independent_episode_assumption_verified':False,'population_confidence_claim':False,
  'limits':'Intervals condition on independent representative source episodes, which is not established here. Adaptive v1/v2 development is retained; v3 uses a fresh holdout. Correlated windows are not treated as independent trials. No production/energy/efficiency claims.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=analyze(a.source)
 with a.output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print('Completed paired episode diagnostics with explicit independence limits')
if __name__=='__main__':main()
