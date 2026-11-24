"""D04 fixed-budget pretrained visual baseline; derived modalities stay explicit."""
import argparse,hashlib,json,sys
from pathlib import Path
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from l3_system.monitoring.sensor_fusion_data import build
from l3_system.monitoring.multimodal_model import Batch,FusionBaseline
from l3_system.monitoring.fusion_comparison import compare
from l3_system.monitoring.fusion_bundle import save_bundle
from l3_system.monitoring.binding import make_binding
from sklearn.feature_extraction.text import TfidfVectorizer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['dataset','weights','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    episodes=[]
    for split in ['train','validation','test']:
        for path in sorted(a.dataset.glob(split+'*/model_episode.json')):
            e=json.loads(path.read_text());e['observations']=[row[1:] for row in e['observations']];episodes.append(e)
    batches,receipt=build(episodes)
    # Fixed deterministic budget, label-independent row stride. All source groups
    # remain represented; this is not an epoch count chosen from test results.
    for split,b in list(batches.items()):
        idx=torch.arange(0,len(b.labels),8)
        batches[split]=Batch(b.series[idx],b.images[idx],[b.texts[i] for i in idx],b.mask[idx],b.labels[idx],[b.groups[i] for i in idx])
    registry=json.loads(Path('configs/goals/atropine_model_registry.json').read_text())
    # Exact model coordinate declarations, not assertions of physical calibration.
    obs=[{'index':i,'name':('D_excluded_clock_zero' if i==0 else 'Atropine_model_observation_'+str(i)),
          'unit':'excluded' if i==0 else 'model coordinate; physical calibration not established',
          'coordinate':'D zero padding' if i==0 else 'normalize_false'} for i in range(39)]
    act=[{'index':i,'name':'Atropine_model_action_'+str(i),'unit':'model coordinate','coordinate':'uss_subtracted'} for i in range(4)]
    binding=make_binding({'environment':'Atropine','observation':obs,'action':act},sha(a.dataset/'manifest.json'))
    config={'seeds':[17,29,43],'epochs':20,'stride':8,'source_receipt':receipt,'binding':binding,
            'pretrained_weights_sha256':sha(a.weights),'weights_source':'torchvision ResNet18 IMAGENET1K_V1',
            'test_previously_inspected':True,'scope':'fixed-budget derived-modality engineering study, not independent research acceptance'}
    (a.output/'frozen_plan.json').write_text(json.dumps(config,indent=2)+'\n')
    visual=torch.load(a.weights,map_location='cpu',weights_only=True)
    shared=a.output/'shared_visual.pt';torch.save(visual,shared);runs=[]
    for seed in config['seeds']:
        def save(name,model,vectorizer,nb,validation):
            folder=a.output/f'seed-{seed}'/name
            save_bundle(folder,model,vectorizer,nb,binding=binding,label_scope='sensor_fault')
            # Factor identical frozen visual trunk into one shared asset. Keep
            # trained heads, LSTM, text and normalization as tensor-only deltas.
            head={k:v for k,v in model.state_dict().items() if not k.startswith('vision.') or k.startswith('vision.fc.')}
            torch.save(head,folder/'heads.pt')
            original=json.loads((folder/'card.json').read_text())
            card={'schema_version':'D04FactoredBundle/1.0','mode':model.mode,'width':model.text.out_features,
                  'binding':binding,'shared_visual_sha256':sha(shared),'heads_sha256':sha(folder/'heads.pt'),
                  'text_sha256':sha(folder/'text.json'),'original_versions':original['versions'],
                  'pretrained_visual_weights':True,'deployment_authorized':False}
            (folder/'factor_card.json').write_text(json.dumps(card,indent=2)+'\n')
            restored=FusionBaseline(len(vectorizer.vocabulary_),model.mode,width=model.text.out_features,visual_state=visual)
            missing,unexpected=restored.load_state_dict(torch.load(folder/'heads.pt',map_location='cpu',weights_only=True),strict=False)
            expected={k for k in restored.state_dict() if k.startswith('vision.') and not k.startswith('vision.fc.')}
            assert set(missing).issubset(expected) and not unexpected
            assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
            restored.eval()
            text=json.loads((folder/'text.json').read_text());params=text['parameters'];params['ngram_range']=tuple(params['ngram_range'])
            rv=TfidfVectorizer(vocabulary=text['vocabulary'],**params);rv.idf_=np.array(text['idf'])
            texts=[t if m else '' for t,m in zip(validation.texts,validation.mask[:,2])]
            x=torch.tensor(vectorizer.transform(texts).toarray()).float();rx=torch.tensor(rv.transform(texts).toarray()).float()
            with torch.no_grad():
                first=model(validation.series,validation.images,x,validation.mask)[0]
                second=restored(validation.series,validation.images,rx,validation.mask)[0]
            assert torch.equal(first,second),'factored restoration changed logits'
            # These two newly generated temporary full-bundle files are replaced
            # by the checked, self-describing factored representation.
            (folder/'weights.pt').unlink();(folder/'card.json').unlink()
            return {'factor_card_sha256':sha(folder/'factor_card.json'),'validation_roundtrip_exact':True,
                    'shared_visual_sha256':sha(shared),'heads_sha256':sha(folder/'heads.pt')}
        result=compare(batches,epochs=20,seed=seed,common_test_cohort=True,artifact_writer=save,visual_state=visual)
        runs.append(result);(a.output/'progress.json').write_text(json.dumps({'runs':runs},indent=2)+'\n')
        print('seed',seed,'complete',flush=True)
    report={'schema_version':'D04PretrainedStudy/1.0','config':config,'runs':runs,'models_trained':21,
            'sample_counts':{k:len(b.labels) for k,b in batches.items()},'original_data_sha256':sha(a.dataset/'manifest.json'),
            'pretrained_visual_weights':True,'visual_input':'numeric-derived heatmaps at 32x32, not camera images',
            'original_inputs_modified':False,'independent_research_acceptance':False,
            'limits':'Pretrained ImageNet features are not industrial visual validation. Fixed 20-epoch budget is not convergence evidence. Previously inspected test is an engineering comparison only.'}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Completed 21 pretrained-visual engineering models and exact artifact roundtrips.')
if __name__=='__main__':main()
