"""Reproduce final D sensor-bias model using training/validation files only."""
import hashlib,json
from pathlib import Path
from .residual_detector import ResidualDetector

def train(dataset,output,*,binding):
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    episodes=[]
    for split in ('train','validation'):
        for path in sorted(dataset.glob(split+'*/model_episode.json')):
            e=json.loads(path.read_text());number=int(e['episode_id'].split('_')[-1])
            e['fit_role']='train' if split=='train' else 'calibration' if number in (17001,17002,17003,17007,17008,17009) else 'threshold'
            episodes.append(e)
    model=ResidualDetector();card=model.fit(episodes,binding=binding);sha=model.save(output/'model.pkl')
    record={'card':card,'model_sha256':sha,'fresh_test_opened':False,
            'recipe_source_sha256':hashlib.sha256(Path(__file__).with_name('residual_detector.py').read_bytes()).hexdigest(),
            'input_manifest_sha256':hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
            'role_policy':'8 train; validation clean 17001..3 and fault 17007..9 calibrate; remaining validation selects threshold'}
    (output/'freeze.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','binding-record','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();binding=json.loads(a.binding_record.read_text())['card']['binding']
    result=train(a.dataset,a.output,binding=binding)
    print(json.dumps({'model_sha256':result['model_sha256'],'test_loaded':False}))
