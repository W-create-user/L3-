"""Repeated initialization study; seeds do not create independent episodes."""
import math
import statistics
from .fusion_comparison import compare, VARIANTS, subset
from .multimodal_model import split_check


def study(batches,*,seeds=(17,29,43),epochs=1,scope):
    if scope not in ('synthetic_engineering_only','reviewed_sensor_fault_exploratory'):
        raise ValueError('explicit supported evidence scope required')
    if not isinstance(seeds,(list,tuple)) or len(seeds)<2 or any(type(s)is not int or s<0 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('at least two distinct nonnegative seeds required')
    split_check(batches)
    if not batches['test'].mask.all(dim=1).any():raise ValueError('no complete common test cohort')
    for _,columns in VARIANTS.values():split_check({name:subset(batch,columns)[0] for name,batch in batches.items()})
    runs=[compare(batches,epochs=epochs,seed=seed,common_test_cohort=True) for seed in seeds]
    summary={}
    for variant in VARIANTS:
        values=[run['variants'][variant]['test']['f1'] for run in runs]
        if not all(math.isfinite(x) for x in values):raise ValueError('finite scores required')
        summary[variant]={'test_f1_mean':statistics.mean(values),
            'test_f1_initialization_sd':statistics.stdev(values),'initializations':len(values)}
    return {'schema_version':'D04InitializationStudy/1.0','scope':scope,'seeds':list(seeds),
        'epochs_per_variant':epochs,'trained_models':len(seeds)*len(VARIANTS),'runs':runs,
        'summary':summary,'deployment_authorized':False,'physical_efficacy_established':False,
        'limits':'Initialization variability on fixed groups only; no independent episode replication, significance test or physical safety inference. Validation-only thresholds. Complete-case test cohort excludes missing modalities. Frozen random visual backbone has no pretrained semantics.'}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    import torch
    from .multimodal_model import synthetic_batches
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-demo',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=study(synthetic_batches(),scope='synthetic_engineering_only')
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Completed 21 synthetic training runs; no physical efficacy conclusion.')
