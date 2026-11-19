"""D-only causal derived modalities from reviewed B sensor archives.

Not camera/log evidence or physical-health labels. Original episodes are copied.
"""
import hashlib
import json
import numpy as np
import torch
from torch.nn.functional import interpolate
from .multimodal_model import Batch,split_check

RECIPE='D-sensor-derived-modalities/1.0'


def render(window,mean,std):
    values=np.asarray(window,dtype=np.float32)
    if values.ndim!=2 or values.shape[1]!=38 or not np.isfinite(values).all():raise ValueError('finite non-clock 38-channel window required')
    z=np.clip((values-mean)/std,-4,4)/8+.5
    pixels=torch.from_numpy(z.T.copy())[None,None]
    image=interpolate(pixels,size=(32,32),mode='nearest')[0].repeat(3,1,1)
    # Values only: no label, split, seed, group, injected schedule or future state.
    text=' '.join(f'channel{i+1} value {float(v):.9g} change {float(d):.9g}'
                  for i,(v,d) in enumerate(zip(values[-1],values[-1]-values[0])))
    return image,text


def build(episodes,window=4):
    if type(window)is not int or not 2<=window<=30:raise ValueError('window in 2..30 required')
    seen={};ids=set();records=[]
    for e in episodes:
        if e.get('label_scope')!='sensor_fault' or e.get('split') not in ('train','validation','test'):raise ValueError('reviewed sensor-only splits required')
        eid=e['episode_id'];group=e['lineage_group'];split=e['split']
        if eid in ids or group in seen and seen[group]!=split:raise ValueError('duplicate episode or cross-split lineage')
        x=np.array(e['observations'],dtype=np.float32,copy=True);y=list(e['labels'])
        if x.ndim!=2 or x.shape[1]!=38 or len(x)<window or not np.isfinite(x).all() or len(y)!=len(x) or any(type(v)is not bool for v in y):raise ValueError('complete reviewed 38-channel episodes required')
        if split=='train' and any(y):raise ValueError('source train must be uncorrupted')
        ids.add(eid);seen[group]=split;records.append((eid,group,split,x,y))
    train=[x for _,_,s,x,_ in records if s=='train']
    if not train:raise ValueError('normal source train required')
    values=np.concatenate(train);mean=values.mean(axis=0);std=values.std(axis=0).clip(1e-6)
    rows={s:[] for s in ('train','validation','test')}
    for eid,group,split,x,y in records:
        views=[(x,y)]
        if split=='train':
            # D augmentation only, follows B sensor recipe; never edits B originals.
            altered=x.copy();altered[9:20,5]*=1.2
            labels=[9<=i<20 for i in range(len(x))];views.append((altered,labels))
        for data,labels in views:
            for end in range(window-1,len(data)):
                current=data[end-window+1:end+1];image,text=render(current,mean,std)
                series=torch.zeros(window,39);series[:,1:]=torch.from_numpy(current.copy())
                rows[split].append([series,image,text,any(labels[end-window+1:end+1]),group,[True,True,True]])
    # Identical derived assets across splits cannot imply independent modalities.
    owners={};keys={}
    for split,items in rows.items():
        for i,row in enumerate(items):
            for col,data in ((1,row[1].numpy().tobytes()),(2,row[2].encode())):
                key=(col,hashlib.sha256(data).hexdigest());keys[split,i,col]=key;owners.setdefault(key,set()).add(split)
    removed={s:{'image':0,'text':0} for s in rows}
    for split,items in rows.items():
        for i,row in enumerate(items):
            for col,name in ((1,'image'),(2,'text')):
                if len(owners[keys[split,i,col]])>1:
                    row[5][col]=False;removed[split][name]+=1
                    if col==1:row[1]=torch.zeros_like(row[1])
                    else:row[2]=''
    batches={}
    for split,items in rows.items():
        if not items:raise ValueError('all three splits required')
        batches[split]=Batch(torch.stack([r[0] for r in items]),torch.stack([r[1] for r in items]),
            [r[2] for r in items],torch.tensor([r[5] for r in items],dtype=torch.bool),
            torch.tensor([r[3] for r in items],dtype=torch.float32),[r[4] for r in items])
    split_check(batches)
    receipt={'recipe':RECIPE,'source_scope':'B artificial sensor bias; physical labels unavailable',
        'train_augmentation':'one clean and one D-only obs[6]*1.2 view at original sample indices 9..19, same lineage/split',
        'clock_input':'source observation[0] excluded; model column0 fixed zero',
        'display_scaling':'original uncorrupted train mean/std only; z clipped +/-4 solely for visualization',
        'visual_and_text_source':'causal numeric window only; derived modalities, not independent camera/log data',
        'cross_split_identical_assets_masked':removed,'windows':{s:len(b.labels) for s,b in batches.items()},
        'groups':{s:len(set(b.groups)) for s,b in batches.items()},
        'scaler_sha256':hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest(),
        'previously_inspected_test':True,'acceptance_scope':'exploratory only; new independent holdout required'}
    return batches,receipt
