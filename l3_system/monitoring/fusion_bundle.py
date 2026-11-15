"""Trusted offline experimental D04 bundles; never authorizes deployment.

Caller must obtain card hash and binding from its own trusted training record,
not from an untrusted bundle. Hashes are integrity checks, not signatures.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import sklearn
import torch
import torchvision
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .binding import fingerprint
from .multimodal_model import FusionBaseline


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(value):
    if (not isinstance(value,dict) or value.get('schema_version')!='AnomalyBinding/1.0'
            or value.get('environment')!='Atropine'
            or not re.fullmatch('[0-9a-f]{64}',str(value.get('collection_config_sha256','')))
            or value['collection_config_sha256']=='0'*64):raise ValueError('explicit Atropine semantic binding required')
    for name,n in (('observation',39),('action',4)):
        entries=value.get(name)
        if not isinstance(entries,list) or len(entries)!=n:raise ValueError('bound dimensions differ')
        for i,entry in enumerate(entries):
            if (not isinstance(entry,dict) or set(entry)!={'index','name','unit','coordinate'}
                    or type(entry['index'])is not int or entry['index']!=i
                    or any(not isinstance(entry[k],str) or not entry[k].strip() for k in ('name','unit','coordinate'))):
                raise ValueError('ordered complete variable semantics required')


def _json(path,value):path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def _versions():return {'torch':torch.__version__,'torchvision':torchvision.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__}


def save_bundle(directory,model,vectorizer,nb,*,binding,label_scope):
    _binding(binding)
    if label_scope not in ('synthetic_fixture','sensor_fault','process_anomaly'):raise ValueError('explicit label scope required')
    if not isinstance(model,FusionBaseline) or not isinstance(vectorizer,TfidfVectorizer):raise ValueError('known D04 baseline required')
    if nb is not None and not isinstance(nb,MultinomialNB):raise ValueError('known text baseline required')
    n=len(vectorizer.vocabulary_)
    if n!=model.text.in_features or n<1 or n>256:raise ValueError('model/vocabulary dimensions differ')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    # Only tensors, never the Python model or sklearn object, go into checkpoint.
    torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},directory/'weights.pt')
    text={'vocabulary':{key:int(value) for key,value in vectorizer.vocabulary_.items()},'idf':vectorizer.idf_.tolist(),
          'parameters':{'lowercase':vectorizer.lowercase,'token_pattern':vectorizer.token_pattern,
                        'norm':vectorizer.norm,'use_idf':vectorizer.use_idf,'smooth_idf':vectorizer.smooth_idf,
                        'sublinear_tf':vectorizer.sublinear_tf,'binary':vectorizer.binary,
                        'analyzer':vectorizer.analyzer,'ngram_range':list(vectorizer.ngram_range)},'nb':None}
    if nb is not None:
        text['nb']={key:getattr(nb,key).tolist() for key in ('classes_','class_count_','class_log_prior_','feature_count_','feature_log_prob_')}
    _json(directory/'text.json',text)
    card={'schema_version':'D04Bundle/0.1','binding':binding,'binding_sha256':fingerprint(binding),
          'label_scope':label_scope,'mode':model.mode,'width':model.text.out_features,'versions':_versions(),
          'files':{name:_sha(directory/name) for name in ('weights.pt','text.json')},
          'scope':'trusted_offline_experimental_only','deployment_authorized':False,
          'pretrained_visual_weights':False,'probability_calibrated':False}
    _json(directory/'card.json',card)
    return {'card_sha256':_sha(directory/'card.json'),'binding_sha256':fingerprint(binding),'deployment_authorized':False}


def load_trusted_bundle(directory,*,expected_card_sha256,expected_binding):
    _binding(expected_binding);directory=Path(directory).resolve()
    for name,limit in (('card.json',100000),('text.json',1000000),('weights.pt',64000000)):
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or path.stat().st_size>limit:raise ValueError('bundle path or size invalid')
    if _sha(directory/'card.json')!=expected_card_sha256:raise ValueError('trusted card fingerprint mismatch')
    def invalid(value):raise ValueError('nonfinite bundle JSON')
    card=json.loads((directory/'card.json').read_text(),parse_constant=invalid)
    if (card.get('schema_version')!='D04Bundle/0.1' or card.get('binding')!=expected_binding
            or card.get('binding_sha256')!=fingerprint(expected_binding) or card.get('versions')!=_versions()
            or card.get('deployment_authorized') is not False or card.get('mode') not in ('concat','attention')
            or type(card.get('width'))is not int or not 1<=card['width']<=256):
        raise ValueError('bundle version, semantics or experimental scope mismatch')
    if set(card['files'])!={'weights.pt','text.json'}:raise ValueError('unexpected bundle files')
    for name,want in card['files'].items():
        if _sha(directory/name)!=want:raise ValueError('bundle file hash mismatch')
    text=json.loads((directory/'text.json').read_text(),parse_constant=invalid)
    vocabulary=text['vocabulary'];n=len(vocabulary)
    if (not 1<=n<=256 or any(not isinstance(k,str) or type(v)is not int for k,v in vocabulary.items())
            or sorted(vocabulary.values())!=list(range(n))):raise ValueError('invalid vocabulary')
    idf=np.asarray(text['idf'],dtype=float)
    if idf.shape!=(n,) or not np.isfinite(idf).all() or not (idf>0).all():raise ValueError('invalid IDF')
    parameters=text['parameters'];parameters['ngram_range']=tuple(parameters['ngram_range'])
    vectorizer=TfidfVectorizer(vocabulary=vocabulary,**parameters);vectorizer.idf_=idf
    nb=None
    if text['nb'] is not None:
        nb=MultinomialNB()
        shapes={'classes_':(2,),'class_count_':(2,),'class_log_prior_':(2,),
                'feature_count_':(2,n),'feature_log_prob_':(2,n)}
        for key,shape in shapes.items():
            value=np.asarray(text['nb'][key],dtype=float)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid text baseline arrays')
            setattr(nb,key,value)
        if nb.classes_.tolist()!=[0.,1.]:raise ValueError('binary baseline classes required')
        nb.n_features_in_=n
    model=FusionBaseline(n,mode=card['mode'],width=card['width'])
    tensors=torch.load(directory/'weights.pt',map_location='cpu',weights_only=True)
    if any(not torch.isfinite(value).all() for value in tensors.values()):raise ValueError('nonfinite checkpoint')
    model.load_state_dict(tensors,strict=True);model.eval()
    return model,vectorizer,nb,card
