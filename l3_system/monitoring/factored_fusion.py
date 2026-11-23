"""Restore D's shared-pretrained-trunk artifacts from a trusted study receipt.

Inputs remain prepared causal batches. This loader never authorizes deployment
or turns sigmoid logits into a calibrated probability.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .multimodal_model import FusionBaseline
from .fusion_bundle import _binding,_versions

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_study_variant(root,*,seed,variant,expected_study_sha256):
    root=Path(root).resolve()
    if sha(root/'summary.json')!=expected_study_sha256:raise ValueError('trusted study fingerprint differs')
    report=json.loads((root/'summary.json').read_text())
    run=next((r for r in report['runs'] if r['seed']==seed),None)
    if run is None or variant not in run['variants']:raise ValueError('unknown study model')
    result=run['variants'][variant];receipt=result['private_artifact_receipt']
    directory=(root/f'seed-{seed}'/variant).resolve()
    if not directory.is_relative_to(root):raise ValueError('bundle path escaped study')
    card_path=directory/'factor_card.json'
    if sha(card_path)!=receipt['factor_card_sha256']:raise ValueError('factor card changed')
    card=json.loads(card_path.read_text());_binding(card['binding'])
    if (card['schema_version']!='D04FactoredBundle/1.0' or card['original_versions']!=_versions()
            or card['binding']!=report['config']['binding'] or card['deployment_authorized'] is not False):
        raise ValueError('bundle semantics/version changed')
    shared=root/'shared_visual.pt'
    for path,want in [(shared,card['shared_visual_sha256']),(directory/'heads.pt',card['heads_sha256']),
                      (directory/'text.json',card['text_sha256'])]:
        if sha(path)!=want:raise ValueError('factored bundle bytes changed')
    text=json.loads((directory/'text.json').read_text());parameters=text['parameters']
    parameters['ngram_range']=tuple(parameters['ngram_range'])
    vocabulary=text['vocabulary']
    if not 1<=len(vocabulary)<=256 or sorted(vocabulary.values())!=list(range(len(vocabulary))):
        raise ValueError('invalid vocabulary')
    rv=TfidfVectorizer(vocabulary=vocabulary,**parameters);rv.idf_=np.asarray(text['idf'],dtype=float)
    model=FusionBaseline(len(vocabulary),card['mode'],width=card['width'],
        visual_state=torch.load(shared,map_location='cpu',weights_only=True))
    heads=torch.load(directory/'heads.pt',map_location='cpu',weights_only=True)
    if any(not torch.isfinite(t).all() for t in heads.values()):raise ValueError('nonfinite trained tensor')
    missing,unexpected=model.load_state_dict(heads,strict=False)
    if unexpected or any(not k.startswith('vision.') or k.startswith('vision.fc.') for k in missing):
        raise ValueError('trained tensors missing or incompatible')
    model.eval();nb=None
    if text['nb'] is not None:
        nb=MultinomialNB()
        for key,value in text['nb'].items():setattr(nb,key,np.asarray(value))
        nb.n_features_in_=len(vocabulary)
    return model,rv,nb,{'threshold':result['threshold'],'columns':result['modality_columns'],
        'binding':card['binding'],'deployment_authorized':False,'sigmoid_probability_calibrated':False}

def score_prepared(model,vectorizer,batch,receipt):
    from .fusion_comparison import subset
    batch.validate();reduced,counts=subset(batch,receipt['columns'])
    texts=[t if m else '' for t,m in zip(reduced.texts,reduced.mask[:,2])]
    with torch.no_grad():
        values=torch.sigmoid(model(reduced.series,reduced.images,
            torch.tensor(vectorizer.transform(texts).toarray()).float(),reduced.mask)[0]).numpy()
    return {'uncalibrated_sigmoid_scores':values.tolist(),'anomalous':(values>=receipt['threshold']).tolist(),
            'sample_counts':counts,'execution_authorized':False}
