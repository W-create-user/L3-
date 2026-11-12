"""Bridge accepted D01 records and independently reviewed label sidecars to IF."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from intake_dataset import inspect_dataset,read_json,digest
from l3_system.monitoring.binding import make_binding
from l3_system.monitoring.anomaly_detector import AnomalyDetector


def prepare(manifest,root,config_path,dictionary_path,labels):
    report=inspect_dataset(manifest,root,config_path,dictionary_path)
    if not report['passed']:raise ValueError('D01 intake failed: '+'; '.join(report['errors']))
    if (labels.get('schema_version')!='D03Labels/1.0' or labels.get('scope') not in ('sensor_fault','process_anomaly')
            or labels.get('alignment')!='observation_before' or not labels.get('split_reviewed_by')):
        raise ValueError('independent label scope/alignment and split review required')
    selected=[e for e in manifest['episodes'] if e['split']!='excluded']
    if set(labels.get('episodes',{}))!={e['episode_id'] for e in selected}:raise ValueError('exact label coverage required')
    dataset=[]
    for episode in selected:
        sidecar=labels['episodes'][episode['episode_id']]
        for key in ('reviewed_by','source_ref','normal_basis'):
            if not isinstance(sidecar.get(key),str) or not sidecar[key].strip():raise ValueError('label review, source and normal basis required')
        rows=[]
        for artifact in episode['artifacts']['timeseries']:
            path=(Path(root)/artifact['path']).resolve()
            if digest(path)!=artifact['sha256']:raise ValueError('source changed after intake')
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        truth=sidecar.get('labels')
        if not isinstance(truth,list) or len(truth)!=len(rows) or any(type(v) is not bool for v in truth):
            raise ValueError('one explicit independent boolean label per observation_before required')
        if episode['label_summary']=='normal' and any(truth):raise ValueError('labels contradict normal manifest summary')
        if episode['label_summary']=='injected_anomaly' and not any(truth):raise ValueError('labels contradict anomaly summary')
        if episode['split']=='train' and not episode['iforest_train']:continue
        dataset.append(dict(episode_id=episode['episode_id'],lineage_group=episode['lineage_group'],split=episode['split'],
                            observations=[r['observation_before'] for r in rows],labels=truth,
                            label_source=sidecar['source_ref'],label_scope=labels['scope']))
    binding=make_binding(read_json(dictionary_path),manifest['collection_config_sha256'])
    receipt={'dataset_id':manifest['dataset_id'],'dictionary_sha256':manifest['variable_dictionary_sha256'],
             'labels_reviewed_by':{eid:v['reviewed_by'] for eid,v in labels['episodes'].items()},
             'split_reviewed_by':labels['split_reviewed_by'],
             'limitations':'Review fields are declarations, not independent proof of physical truth or permission.'}
    return dataset,binding,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('manifest','root','config','dictionary','labels','output'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--window-size',type=int,default=4);args=parser.parse_args()
    labels=read_json(args.labels)
    episodes,binding,receipt=prepare(read_json(args.manifest),args.root,args.config,args.dictionary,labels)
    model=AnomalyDetector(window_size=args.window_size)
    model.fit(episodes,label_scope=labels['scope'],binding=binding,receipt=receipt)
    model.save(args.output)
    print(json.dumps({'trained':True,'label_scope':labels['scope'],'physical_safety_limit':False}))


if __name__=='__main__':main()
