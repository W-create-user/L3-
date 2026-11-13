"""Re-train D04 variants separately with their own validation-only thresholds.

Synthetic CLI is engineering verification, not evidence of process efficacy.
"""
from .multimodal_model import Batch,train_baseline,synthetic_batches
import torch

VARIANTS={'full_concat':('concat',(0,1,2)),'full_attention':('attention',(0,1,2)),
          'time_only':('concat',(0,)),'image_only':('concat',(1,)),'text_only':('concat',(2,)),
          'without_image':('attention',(0,2)),'without_text':('attention',(0,1))}


def subset(batch,columns):
    mask=torch.zeros_like(batch.mask)
    mask[:,list(columns)]=batch.mask[:,list(columns)]
    keep=mask.any(dim=1);indices=keep.nonzero().flatten().tolist()
    if not indices:raise ValueError('variant has no available samples')
    result=Batch(batch.series[keep].clone(),batch.images[keep].clone(),[batch.texts[i] for i in indices],
                 mask[keep].clone(),batch.labels[keep].clone(),[batch.groups[i] for i in indices])
    return result,{'input_samples':len(keep),'eligible_samples':len(indices),'abstained_samples':int((~keep).sum())}


def compare(batches,*,epochs=1,seed=17):
    reports={}
    for name,(mode,columns) in VARIANTS.items():
        reduced={};counts={}
        for split,batch in batches.items():reduced[split],counts[split]=subset(batch,columns)
        _,_,_,report=train_baseline(reduced,mode=mode,epochs=epochs,seed=seed)
        # Individual metrics use available-case denominators; no ranking across
        # different denominators is justified without a separate paired analysis.
        reports[name]={'training':'independent_retraining','modality_columns':list(columns),
            'sample_counts':counts,'threshold_source':report['threshold_source'],'threshold':report['threshold'],
            'test':report['test'],'losses':report['losses'],
            'visual_backbone_frozen':report['visual_backbone_frozen'],
            'text_naive_bayes':report['text_naive_bayes']}
    return {'schema_version':'D04Comparison/0.1','variants':reports,'epochs_per_variant':epochs,'seed':seed,
        'paired_comparison_established':False,'deployment_authorized':False,
        'limits':'Available-case sample sets may differ. Independent retraining, not inference-only masking. Random frozen visual backbone has no pretrained semantics. Numerical superiority is not established.'}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-demo',action='store_true',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    result=compare(synthetic_batches());result['scope']='synthetic_engineering_only'
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Executed 7 independently retrained synthetic variants; no real process efficacy conclusion.')
