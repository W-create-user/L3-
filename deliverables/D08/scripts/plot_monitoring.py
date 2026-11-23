"""Export evidence-based D research figures; seed spread is not a confidence interval."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 root=a.evidence;reports=[json.loads((root/n).read_text()) for n in ['residual_v1_holdout.json','residual_v2_holdout.json','residual_final_holdout.json']]
 final=reports[-1];fusion=json.loads((root/'pretrained_fusion_study.json').read_text())
 fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
 colors=['#64748b','#087f8c'];ax=axes[0,0];keys=['TP','FP','FN','TN'];x=np.arange(4)
 ax.bar(x-.18,[final['baseline']['counts'][k] for k in keys],.36,label='Original IF',color=colors[0]);ax.bar(x+.18,[final['result']['counts'][k] for k in keys],.36,label='IF + relative residual',color=colors[1]);ax.set_xticks(x,keys);ax.set_ylabel('Correlated windows');ax.set_title('Final fresh holdout: 12 episodes');ax.legend(fontsize=8)
 ax=axes[0,1];recall=[r['window_recall']*100 for r in reports]
 ax.bar(['v1 / 28001+','v2 / 38001+','v3 / 48001+'],recall,color=['#94a3b8','#94a3b8','#087f8c']);ax.axhline(80,ls='--',color='#b45309',label='Prespecified engineering criterion');ax.set_ylim(0,110);ax.set_ylabel('Window recall (%)');ax.set_title('All development holdouts retained');ax.legend(fontsize=8)
 for i,y in enumerate(recall):ax.text(i,y+2,f'{y:.1f}%',ha='center')
 ax=axes[1,0];variants=list(fusion['runs'][0]['variants']);names=['Concat','Attention','Time','Image','Text','No image','No text']
 values=np.array([[r['variants'][v]['test']['f1'] for v in variants] for r in fusion['runs']])
 ax.bar(np.arange(7),values.mean(0),color='#cbd5e1')
 for seed,ys in zip([17,29,43],values):ax.scatter(np.arange(7),ys,s=22,label=f'Seed {seed}')
 ax.set_xticks(np.arange(7),names,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('F1, previously inspected test');ax.set_title('21 pretrained-visual models, fixed 20 epochs');ax.legend(fontsize=8)
 ax=axes[1,1];bins=final['result']['calibration_bins'];ax.plot([0,1],[0,1],ls='--',color='#64748b')
 ax.scatter([b['mean_probability'] for b in bins],[b['label_rate'] for b in bins],s=[20+2*np.sqrt(b['count']) for b in bins],color='#087f8c');ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02);ax.set_xlabel('Calibrated sensor-bias label estimate');ax.set_ylabel('Observed window label rate');ax.set_title(f"Final holdout calibration diagnostic\nBrier={final['result']['brier_score']:.6f}; ECE={final['result']['ece']:.6f}")
 fig.suptitle('L3 D monitoring evidence',fontsize=18)
 fig.supxlabel('Simulation sensor-copy bias only; correlated windows; derived image/text; no physical safety or production claim.',fontsize=9)
 fig.savefig(a.output/'monitoring_results.png',dpi=180);fig.savefig(a.output/'monitoring_results.pdf');plt.close(fig)
 print('Exported monitoring_results.png and monitoring_results.pdf')
if __name__=='__main__':main()
