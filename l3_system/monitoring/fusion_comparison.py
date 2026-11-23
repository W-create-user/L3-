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


def compare(batches,*,epochs=1,seed=17,common_test_cohort=False,artifact_writer=None,visual_state=None):
    if artifact_writer is not None and not callable(artifact_writer):raise ValueError('artifact writer must be callable')
    if type(common_test_cohort)is not bool:raise ValueError('explicit cohort mode required')
    # Selection depends only on pre-existing modality availability, not labels or scores.
    test=batches['test'];common_mask=test.mask.all(dim=1)
    common_indices=common_mask.nonzero().flatten().tolist()
    if common_test_cohort and not common_indices:raise ValueError('no complete common test cohort')
    if common_test_cohort:
        common=Batch(test.series[common_mask].clone(),test.images[common_mask].clone(),
            [test.texts[i] for i in common_indices],test.mask[common_mask].clone(),test.labels[common_mask].clone(),
            [test.groups[i] for i in common_indices])
        batches={**batches,'test':common}
    reports={}
    for name,(mode,columns) in VARIANTS.items():
        reduced={};counts={}
        for split,batch in batches.items():reduced[split],counts[split]=subset(batch,columns)
        model,vectorizer,nb,report=train_baseline(reduced,mode=mode,epochs=epochs,seed=seed,visual_state=visual_state)
        artifact=artifact_writer(name,model,vectorizer,nb,reduced['validation']) if artifact_writer else None
        # Individual metrics use available-case denominators; no ranking across
        # different denominators is justified without a separate paired analysis.
        reports[name]={'training':'independent_retraining','modality_columns':list(columns),
            'sample_counts':counts,'threshold_source':report['threshold_source'],'threshold':report['threshold'],
            'test':report['test'],'losses':report['losses'],
            'visual_backbone_frozen':report['visual_backbone_frozen'],
            'text_naive_bayes':report['text_naive_bayes']}
        if artifact is not None:reports[name]['private_artifact_receipt']=artifact
    return {'schema_version':'D04Comparison/0.1','variants':reports,'epochs_per_variant':epochs,'seed':seed,
        'paired_comparison_established':False,'deployment_authorized':False,
        'common_test_cohort':common_test_cohort,'original_test_samples':len(test.labels),
        'test_cohort_samples':len(batches['test'].labels),
        'test_cohort_excluded_samples':len(test.labels)-len(batches['test'].labels),
        'pretrained_visual_weights':visual_state is not None,
        'limits':('Shared complete-case test subset excludes missing-modality cases; selection limits generalization. ' if common_test_cohort else 'Available-case test sample sets may differ. ')+ 'Independent retraining, not inference-only masking. Train/validation availability may differ. Derived numeric images do not establish independent visual sensing. Numerical superiority is not established.'}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-test-cohort',action='store_true')
    parser.add_argument('--synthetic-demo',action='store_true',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    result=compare(synthetic_batches(),common_test_cohort=args.common_test_cohort);result['scope']='synthetic_engineering_only'
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Executed 7 independently retrained synthetic variants; no real process efficacy conclusion.')
