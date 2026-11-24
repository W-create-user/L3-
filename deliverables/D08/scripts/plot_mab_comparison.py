"""Plot paired mAb simulated goal times; never substitute software time for production."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 audit=json.loads(a.audit.read_text());rows=json.loads(a.runs.read_text())['runs'];arms=['traditional_controller','single_agent','four_agent'];conditions=['nominal','sensor_fault','process_disturbance']
 fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');colors=['#64748b','#42a5a5','#d59a35'];x=np.arange(3)
 for j,arm in enumerate(arms):
  values=[audit['by_condition'][c][arm]['mean_simulated_makespan_min_success_only'] for c in conditions]
  axes[0].bar(x+(j-1)*.25,values,.25,label=arm.replace('_',' '),color=colors[j])
 axes[0].set_xticks(x,['Nominal','Controller sensor bias','Initial state offset']);axes[0].set_ylabel('Simulated minutes to buffer goal');axes[0].set_ylim(0,450);axes[0].legend(fontsize=8);axes[0].set_title('30 matched groups, 90 real model runs')
 for j,arm in enumerate(arms):
  values=[r['wall_seconds'] for r in rows if r['arm']==arm]
  axes[1].scatter(np.full(len(values),j),values,color=colors[j],s=18,alpha=.5)
 axes[1].set_xticks(range(3),['Traditional','Single agent','Four roles']);axes[1].set_ylabel('Software wall seconds, including initialization');axes[1].set_title('Software cost only, not plant throughput')
 fig.suptitle('D24 mAb buffer-goal topology comparison',fontsize=15)
 fig.supxlabel('Same proportional law; one executor; 30/30 goals per arm; paired simulated-time improvement = 0%. Not production acceptance.',fontsize=9)
 fig.savefig(a.output/'mab_comparison.png',dpi=180);fig.savefig(a.output/'mab_comparison.pdf');plt.close(fig)
 print('Exported mAb comparison figures')
if __name__=='__main__':main()
