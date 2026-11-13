"""Optional D04 CPU baseline. Inputs must already pass causal/lineage review.

This module does not authorize simulator actions or infer physical safety limits.
Torch/torchvision are optional dependencies; no downloaded pretrained weights.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score


@dataclass
class Batch:
    series: torch.Tensor
    images: torch.Tensor
    texts: list
    mask: torch.Tensor  # time, image, text
    labels: torch.Tensor
    groups: list

    def validate(self):
        n=len(self.texts)
        if (self.series.ndim!=3 or self.series.shape[0]!=n or self.series.shape[2]!=39
                or self.series.shape[1]<1 or self.images.shape!=(n,3,32,32)
                or self.mask.shape!=(n,3) or self.mask.dtype!=torch.bool
                or self.labels.shape!=(n,) or len(self.groups)!=n or n==0):
            raise ValueError('expected nonempty Atropine batches, images RGB 32x32 and explicit modality masks')
        if not all(isinstance(s,str) for s in self.texts) or not all(isinstance(g,str) and g for g in self.groups):
            raise ValueError('text strings and lineage groups required')
        if not torch.isfinite(self.labels).all() or not ((self.labels==0)|(self.labels==1)).all():
            raise ValueError('explicit binary labels required')
        for column,data in ((0,self.series),(1,self.images)):
            if not torch.isfinite(data[self.mask[:,column]]).all():raise ValueError('available inputs must be finite')
        if any(self.mask[i,2] and not text.strip() for i,text in enumerate(self.texts)):
            raise ValueError('available text must be nonempty')
        if not self.mask.any(dim=1).all():raise ValueError('all modalities missing: abstain')


class FusionBaseline(nn.Module):
    def __init__(self,text_dimension,mode='attention',width=16):
        super().__init__()
        if mode not in ('concat','attention') or text_dimension<1:raise ValueError('invalid fusion configuration')
        self.mode=mode
        self.register_buffer('series_mean',torch.zeros(39))
        self.register_buffer('series_std',torch.ones(39))
        self.vision=resnet18(weights=None)
        for parameter in self.vision.parameters():parameter.requires_grad_(False)
        self.vision.fc=nn.Linear(self.vision.fc.in_features,width)
        self.temporal=nn.LSTM(39,width,batch_first=True)
        self.text=nn.Linear(text_dimension,width)
        self.attention=nn.Linear(width,1)
        self.classifier=nn.Linear(width*3 if mode=='concat' else width,1)

    def train(self,mode=True):
        super().train(mode)
        # Frozen random backbone includes BatchNorm running statistics.
        self.vision.eval()
        return self

    def forward(self,series,images,text,mask):
        if mask.dtype!=torch.bool or mask.shape!=(len(series),3) or not mask.any(dim=1).all():
            raise ValueError('explicit masks required; all-missing samples must abstain')
        # Erase missing values before encoding: NaN padding cannot contaminate outputs.
        series=torch.where(mask[:,0,None,None],series,torch.zeros_like(series))
        images=torch.where(mask[:,1,None,None,None],images,torch.zeros_like(images))
        text=torch.where(mask[:,2,None],text,torch.zeros_like(text))
        series=(series-self.series_mean)/self.series_std
        temporal=self.temporal(series)[0][:,-1,:]
        embeddings=torch.stack((temporal,self.vision(images),self.text(text)),dim=1)
        embeddings=embeddings.masked_fill(~mask[:,:,None],0)
        if self.mode=='attention':
            weights=torch.softmax(self.attention(embeddings).squeeze(-1).masked_fill(~mask,-torch.inf),dim=1)
            fused=(embeddings*weights[:,:,None]).sum(dim=1)
        else:
            weights=mask.to(embeddings.dtype)  # availability indicators, not attention
            fused=embeddings.flatten(1)
        return self.classifier(fused).squeeze(-1),weights


def split_check(batches):
    seen={}
    for split,batch in batches.items():
        batch.validate()
        for group in batch.groups:
            if group in seen and seen[group]!=split:raise ValueError('lineage crosses splits')
            seen[group]=split
    if set(batches)!= {'train','validation','test'}:raise ValueError('three independent splits required')
    for split in ('train','validation'):
        if set(batches[split].labels.tolist())!={0.,1.}:raise ValueError('training and threshold selection need both classes')


def metrics(labels,predictions):
    tn,fp,fn,tp=confusion_matrix(labels,predictions,labels=[0,1]).ravel()
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'f1':float(f1_score(labels,predictions,zero_division=0))}


def train_baseline(batches,mode='attention',epochs=1,seed=17):
    """Fit transforms only on train; select binary decision threshold on validation.

    This API expects reviewed labels. A tensor or mask is not a provenance receipt.
    Preprocessing/scalers for production inputs require a separate bound artifact.
    """
    split_check(batches)
    if type(epochs)is not int or epochs<1:raise ValueError('positive epoch count required')
    torch.manual_seed(seed)
    train=batches['train']
    vocabulary=[t for t,m in zip(train.texts,train.mask[:,2]) if m]
    # Fixed neutral vocabulary when this variant excludes text entirely.
    # It is never derived from held-out text or labels.
    vocabulary=vocabulary or ['unavailablemodality']
    vectorizer=TfidfVectorizer(max_features=256)
    vectorizer.fit(vocabulary)
    text={k:torch.from_numpy(vectorizer.transform([t if m else '' for t,m in zip(b.texts,b.mask[:,2])]).toarray()).float()
          for k,b in batches.items()}
    available=train.mask[:,2].numpy()
    nb=(MultinomialNB().fit(text['train'].numpy()[available],train.labels.numpy()[available])
        if set(train.labels.numpy()[available].tolist())=={0.,1.} else None)
    model=FusionBaseline(len(vectorizer.vocabulary_),mode)
    available_series=train.series[train.mask[:,0]]
    if len(available_series):
        model.series_mean.copy_(available_series.mean(dim=(0,1)))
        model.series_std.copy_(available_series.std(dim=(0,1),unbiased=False).clamp_min(1e-6))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    losses=[]
    for _ in range(epochs):
        model.train();optimizer.zero_grad()
        logits,_=model(train.series,train.images,text['train'],train.mask)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,train.labels.float())
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        scores={k:torch.sigmoid(model(b.series,b.images,text[k],b.mask)[0]).numpy().astype('float64') for k,b in batches.items()}
        y=batches['validation'].labels.numpy()
        candidates=np.unique(np.r_[0.,scores['validation'],np.nextafter(float(max(scores['validation'])),math.inf)])
        threshold=max(candidates,key=lambda t:(f1_score(y,scores['validation']>=t,zero_division=0),float(t)))
        test=batches['test'];report={'mode':mode,'epochs':epochs,'losses':losses,'threshold':float(threshold),
            'threshold_source':'validation_only','preprocessing_source':'training_only','pretrained_visual_weights':False,'visual_backbone_frozen':True,
            'test':metrics(test.labels.numpy(),scores['test']>=threshold),'ablations':{}}
        for column,name in enumerate(('time_only','image_only','text_only')):
            mask=torch.zeros_like(test.mask);mask[:,column]=test.mask[:,column]
            valid=mask.any(dim=1)
            count=int(valid.sum())
            result={'eligible_samples':count,'abstained_samples':len(mask)-count,'threshold_source':'full_model_validation_frozen'}
            if count:
                logits,_=model(test.series[valid],test.images[valid],text['test'][valid],mask[valid])
                result.update(metrics(test.labels[valid].numpy(),torch.sigmoid(logits).numpy()>=threshold))
            report['ablations'][name]=result
        valid=test.mask[:,2].numpy()
        report['text_naive_bayes']={'eligible_samples':int(valid.sum()),'abstained_samples':int((~valid).sum())}
        if nb is not None and valid.any():
            report['text_naive_bayes'].update(metrics(test.labels.numpy()[valid],nb.predict(text['test'].numpy()[valid])))
        elif nb is None:
            report['text_naive_bayes'].update(eligible_samples=0,abstained_samples=len(valid),
                unavailable_reason='training text absent or lacks both classes')
    return model,vectorizer,nb,report


def synthetic_batches(seed=17):
    """Artificial signals for engineering only; no process-truth assertions."""
    rng=np.random.default_rng(seed);out={}
    for split,n in (('train',8),('validation',6),('test',6)):
        labels=np.arange(n)%2
        series=rng.normal(size=(n,4,39)).astype('float32')+labels[:,None,None]*0.2
        images=rng.random((n,3,32,32)).astype('float32')
        mask=torch.ones(n,3,dtype=torch.bool);mask[0,1]=False;mask[1,2]=False
        out[split]=Batch(torch.tensor(series,dtype=torch.float32),torch.tensor(images),
            ['synthetic stable sample' if not y else 'synthetic shifted sample' for y in labels],mask,
            torch.tensor(labels,dtype=torch.float32),[f'{split}-{i}' for i in range(n)])
    return out


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic-demo',action='store_true',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    reports=[train_baseline(synthetic_batches(),mode)[3] for mode in ('concat','attention')]
    result={'scope':'synthetic_engineering_only','process_efficacy_established':False,'models':reports,
            'torch':torch.__version__,'note':'No real data, model weights or physical thresholds exported.'}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))
