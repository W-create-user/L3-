"""Offline D research reproduction on fixed verified B sensor archives.

Output directory, checkpoints and derived inputs stay private. Only the aggregate
summary is eligible for publication under the D aggregate authorization.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from review_r03_supplement import review
from l3_system.monitoring.sensor_fusion_data import build,RECIPE
from l3_system.monitoring.fusion_comparison import compare,VARIANTS
from l3_system.monitoring.fusion_bundle import save_bundle,load_trusted_bundle
from l3_system.monitoring.binding import make_binding
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package-root','evidence-root','schema','action-dictionary','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('new private output directory required')
    episodes,source_receipt=review(args.package_root,args.evidence_root)
    batches,data_receipt=build(episodes)
    schema=json.loads(args.schema.read_text());actions=json.loads(args.action_dictionary.read_text())
    if schema['backend']!='AtropineEnvGym' or schema['dimension']!=39 or schema['normalize'] is not False:raise ValueError('verified source schema required')
    observations=[dict(index=f['index'],name=f['name'],unit=f['unit'],coordinate='B unnormalized model observation; physical calibration not established') for f in schema['fields']]
    observations[0].update(name='D_excluded_clock_zero',unit='excluded',coordinate='D zero padding; never source clock')
    actions=[dict(index=f['index'],name=f['name'],unit=f['unit'],coordinate=f['request_scale']) for f in actions if f['backend']=='AtropineEnvGym']
    config=hashlib.sha256(json.dumps({'source':source_receipt,'recipe':RECIPE,'window':4},sort_keys=True).encode()).hexdigest()
    binding=make_binding(dict(environment='Atropine',observation=observations,action=actions),config)
    args.output.mkdir(parents=True);torch.set_num_threads(1);runs=[]
    for seed in (17,29,43):
        def save(name,model,vectorizer,nb,validation):
            directory=args.output/f'seed-{seed}'/name
            receipt=save_bundle(directory,model,vectorizer,nb,binding=binding,label_scope='sensor_fault')
            restored,rv,rnb,card=load_trusted_bundle(directory,expected_card_sha256=receipt['card_sha256'],expected_binding=binding)
            texts=[text if mask else '' for text,mask in zip(validation.texts,validation.mask[:,2])]
            x=vectorizer.transform(texts).toarray();y=rv.transform(texts).toarray()
            np.testing.assert_array_equal(x,y)
            with torch.no_grad():
                first=model(validation.series,validation.images,torch.tensor(x).float(),validation.mask)[0]
                second=restored(validation.series,validation.images,torch.tensor(y).float(),validation.mask)[0]
            if not torch.equal(first,second):raise ValueError('checkpoint validation logits changed')
            if nb is not None:np.testing.assert_array_equal(nb.predict(x),rnb.predict(y))
            return {**receipt,'weights_sha256':card['files']['weights.pt'],'validation_roundtrip_exact':True}
        runs.append(compare(batches,epochs=1,seed=seed,common_test_cohort=True,artifact_writer=save))
        print(f'Finished initialization {seed}',flush=True)
    _,after=review(args.package_root,args.evidence_root)
    if source_receipt!=after:raise ValueError('original input package changed')
    report={'schema_version':'D04DerivedSensorStudy/1.0','source_receipt':source_receipt,'derived_data_receipt':data_receipt,
        'runs':runs,'seeds':[17,29,43],'trained_models':21,'original_inputs_unchanged':True,
        'scope':'previously inspected B simulation sensor-fault data; D-derived image/text and training augmentation; exploratory only',
        'physical_efficacy_established':False,'independent_research_acceptance_passed':False,
        'limits':'Images/text derive from the same numeric window, not independent sensors. Test was inspected previously. Train augmentation matches B injection recipe. No threshold or model selected using this test; no production or physical safety claim.',
        'code_sha256':{p:hashlib.sha256((Path(__file__).resolve().parents[3]/p).read_bytes()).hexdigest() for p in ('l3_system/monitoring/sensor_fusion_data.py','l3_system/monitoring/fusion_comparison.py','l3_system/monitoring/multimodal_model.py','deliverables/D01/scripts/run_derived_sensor_fusion.py')}}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Completed 21 exploratory models; original inputs unchanged; checkpoints remain private.')

if __name__=='__main__':main()
