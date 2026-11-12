"""Train grouped normal-only Isolation Forest, or explicitly synthetic smoke demo."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.monitoring.anomaly_detector import AnomalyDetector


def synthetic_episodes():
    rng=np.random.default_rng(1201);episodes=[]
    for split,count in [('train',6),('validation',4),('test',4)]:
        for index in range(count):
            values=rng.normal(0,0.05,(40,39));labels=[False]*40
            if split!='train' and index%2:
                values[20:30,:3]+=2
                labels[20:30]=[True]*10
            eid=f'SYNTHETIC-{split}-{index}'
            episodes.append(dict(episode_id=eid,lineage_group=eid,split=split,observations=values.tolist(),
                                 labels=labels,label_source='synthetic generator; not process data',label_scope='synthetic_fixture'))
    return episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--episodes',type=Path);group.add_argument('--synthetic-demo',action='store_true')
    parser.add_argument('--label-scope',choices=['sensor_fault','process_anomaly'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-size',type=int,default=4)
    args=parser.parse_args()
    if args.episodes and not args.label_scope:parser.error('real input requires explicit --label-scope')
    if args.synthetic_demo:
        episodes=synthetic_episodes();scope='synthetic_fixture'
    else:
        def invalid(value):raise ValueError(value)
        episodes=json.loads(args.episodes.read_text(),parse_constant=invalid);scope=args.label_scope
    model=AnomalyDetector(window_size=args.window_size)
    card=model.fit(episodes,label_scope=scope);model.save(args.output)
    print(json.dumps({'trained':True,'label_scope':scope,'threshold_source':card['threshold_source'],
                      'physical_safety_limit':False,'model_path':str(args.output/'model.pkl')},ensure_ascii=False))


if __name__=='__main__':main()
