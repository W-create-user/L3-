"""Read-only replay of trusted D artifacts; no new training or simulator steps.

Source episodes are already-inspected development/regression data, never a new
blind evaluation. Labels never enter inference tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

RESIDUAL_SHA = '181c5604f8f54277ef98a029abb169f6382b7a19a189b0269996088198f20bb1'
STUDY_SHA = '9a1c28215ac2da567441fa4f8d22ee23bedc8da047db26fe3ff86cf21dfb562f'

def timed(fn, count=20):
    samples = []
    for _ in range(count):
        started = time.perf_counter(); result = fn(); samples.append(time.perf_counter()-started)
    return result, {'count':count, 'mean_s':float(np.mean(samples)),
                    'p95_s':float(np.percentile(samples,95)), 'max_s':max(samples),
                    'scope':'same already-inspected window repeated, not independent episodes'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual-model', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from l3_system.monitoring.residual_detector import ResidualDetector
    from l3_system.monitoring.factored_fusion import load_study_variant, score_prepared
    from l3_system.monitoring.sensor_fusion_data import build
    from l3_system.monitoring.multimodal_model import Batch
    from l3_system.monitoring.binding import fingerprint
    torch.set_num_threads(1)
    started = time.perf_counter()
    residual = ResidualDetector.load_trusted(args.residual_model, RESIDUAL_SHA)
    residual_load = time.perf_counter()-started
    frozen = json.loads((Path(__file__).resolve().parents[3]/'configs/research/d_monitoring_frozen_20261123.json').read_text())
    if residual.binding != frozen['model_binding']:
        raise ValueError('residual model binding changed')
    episodes = []
    for split in ('train','validation','test'):
        for path in sorted(args.dataset.glob(split+'*/model_episode.json')):
            e = json.loads(path.read_text())
            e['observations'] = [row[1:] for row in e['observations']]
            episodes.append(e)
    if not episodes: raise ValueError('reviewed source episodes unavailable')
    # Restore raw39 for the residual model; fusion intentionally zeros clock.
    first_path = sorted(args.dataset.glob('train*/model_episode.json'))[0]
    raw = json.loads(first_path.read_text()); window = np.asarray(raw['observations'][:4])
    residual_score, residual_times = timed(lambda: residual.score(window))
    started = time.perf_counter()
    model, vectorizer, _, receipt = load_study_variant(args.study, seed=17,
        variant='full_attention', expected_study_sha256=STUDY_SHA)
    fusion_load = time.perf_counter()-started
    dataset_sha = hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
    if receipt['binding']['collection_config_sha256'] != dataset_sha:
        raise ValueError('fusion source dataset differs')
    batches, data_receipt = build(episodes)
    b = batches['train']
    batch = Batch(b.series[:1], b.images[:1], b.texts[:1], b.mask[:1], b.labels[:1], b.groups[:1])
    fusion_score, fusion_times = timed(lambda: score_prepared(model,vectorizer,batch,receipt))
    report = {'schema_version':'DLearnedReplaySmoke/1.0', 'source_kind':'already_inspected_regression',
        'simulator_step_calls':0,'new_training_performed':False,'deployment_authorized':False,
        'python':sys.version,'platform':platform.platform(),'cpu_inference_threads':1,
        'residual':{'model_sha256':RESIDUAL_SHA,'binding':residual.binding,'load_seconds':residual_load,
                    'source_episode_id':raw['episode_id'],'window_indices':[0,1,2,3],
                    'synthetic_window_label':any(raw['labels'][:4]),
                    'prediction_matches_synthetic_label':residual_score['anomalous']==any(raw['labels'][:4]),
                    'score':residual_score,'repeated_window_timing':residual_times},
        'fusion':{'study_sha256':STUDY_SHA,'seed':17,'variant':'full_attention',
                  'binding_sha256':fingerprint(receipt['binding']),'load_seconds':fusion_load,
                  'score':fusion_score,'repeated_window_timing':fusion_times,
                  'modality_source':'D derived causal numeric image/text, not independent field modalities'},
        'times_are_host_measurements_not_deadline_guarantees':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'residual_score':residual_score,'fusion_score':fusion_score,
        'residual_timing':residual_times,'fusion_timing':fusion_times}))

if __name__ == '__main__': main()
