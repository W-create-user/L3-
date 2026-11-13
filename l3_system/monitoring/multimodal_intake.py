"""Experimental bridge: revalidate causal manifest, independent labels and assets.

Declarations are not proof of physical truth or authorization. No simulator calls.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import warnings
import numpy as np
import torch
from PIL import Image
from .binding import fingerprint
from .multimodal_alignment import align
from .multimodal_model import Batch, train_baseline


def _read_asset(root,asset,limit):
    path=(root/asset['path']).resolve()
    if not path.is_relative_to(root):raise ValueError('asset escapes root')
    # Read once; the bytes hashed are exactly those decoded below.
    with path.open('rb') as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('asset exceeds decoding byte budget')
    if hashlib.sha256(data).hexdigest()!=asset['sha256']:raise ValueError('asset changed after alignment')
    return data


def _image(data):
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            if (image.format not in ('PNG','JPEG') or image.mode!='RGB' or getattr(image,'is_animated',False)
                    or image.width*image.height>4_000_000):
                raise ValueError('single-frame RGB PNG/JPEG within pixel budget required')
            image.load()
            array=np.asarray(image.resize((32,32),Image.Resampling.BILINEAR),dtype='float32')/255.
    return torch.from_numpy(array.copy()).permute(2,0,1)


def prepare_batches(manifest,labels,root,*,window_size,allow_experimental=False):
    if allow_experimental is not True:raise ValueError('experimental training intake requires opt-in')
    if type(window_size)is not int or window_size<2:raise ValueError('window_size must be integer >=2')
    root=Path(root).resolve()
    aligned=align(manifest,root)  # Never trust a caller-edited aligned output.
    fields={'schema_version','alignment_sha256','label_scope','source_kind','source_ref','reviewed_by','episodes'}
    if not isinstance(labels,dict) or set(labels)!=fields or labels['schema_version']!='D04Labels/0.1':
        raise ValueError('separate D04 label sidecar required')
    if labels['alignment_sha256']!=fingerprint(manifest):raise ValueError('labels belong to another manifest')
    scopes={'synthetic_fixture':'synthetic_fixture','process_anomaly':'independent_annotation'}
    if (not isinstance(labels['label_scope'],str) or labels['label_scope'] not in scopes
            or scopes[labels['label_scope']]!=labels['source_kind']):raise ValueError('independent label provenance required')
    if any(not isinstance(labels[k],str) or not labels[k].strip() for k in ('source_ref','reviewed_by')):
        raise ValueError('source and reviewer references required')
    if not isinstance(labels['episodes'],list):raise ValueError('episode labels required')
    by_id={e['episode_id']:e for e in manifest['episodes'] if e['split']!='excluded'}
    targets={}
    for entry in labels['episodes']:
        if not isinstance(entry,dict) or set(entry)!={'episode_id','lineage_group','split','record_labels'}:
            raise ValueError('invalid episode label entry')
        eid=entry['episode_id']
        if eid not in by_id or eid in targets:raise ValueError('unknown, excluded or duplicate label episode')
        source=by_id[eid];values=entry['record_labels']
        if (entry['lineage_group']!=source['lineage_group'] or entry['split']!=source['split']
                or not isinstance(values,list) or len(values)!=len(source['series'])
                or any(type(v)is not bool for v in values)):
            raise ValueError('exact groups, splits and independent boolean record labels required')
        targets[eid]=values
    if set(targets)!=set(by_id):raise ValueError('complete labels for every included episode required')
    groups={split:[] for split in ('train','validation','test')};skipped=0;ids={split:[] for split in groups}
    for sample in aligned['samples']:
        split=sample['split'];end=sample['record_index'];eid=sample['episode_id']
        if split=='excluded':continue
        if end<window_size-1:skipped+=1;continue
        source=by_id[eid]['series']
        series=torch.tensor([r['observation'] for r in source[end-window_size+1:end+1]],dtype=torch.float32)
        image=torch.zeros(3,32,32);text=''
        if sample['visual'] is not None:image=_image(_read_asset(root,sample['visual'],8_000_000))
        if sample['text'] is not None:
            text=_read_asset(root,sample['text'],64_000).decode('utf-8',errors='strict')
            if not text.strip():raise ValueError('available text must be nonempty')
        groups[split].append((series,image,text,sample['modality_mask'],targets[eid][end],sample['lineage_group']))
        ids[split].append({'episode_id':eid,'record_index':end})
    batches={}
    for split,rows in groups.items():
        if not rows:raise ValueError('every split needs a full causal window')
        series,images,texts,masks,values,lineages=zip(*rows)
        batch=Batch(torch.stack(series),torch.stack(images),list(texts),torch.tensor(masks,dtype=torch.bool),
                    torch.tensor(values,dtype=torch.float32),list(lineages))
        batch.validate();batches[split]=batch
    receipt={'schema_version':'D04TrainingIntake/0.1','alignment_sha256':fingerprint(manifest),
        'labels_sha256':fingerprint(labels),'binding_sha256':aligned['binding_sha256'],
        'label_scope':labels['label_scope'],'label_semantics':'raw label at causal window end; no event dilation',
        'window_size_records':window_size,'warmup_records_excluded':skipped,'sample_ids':ids,
        'review_claims_verified':False,'scope':'experimental_only','deployment_authorized':False}
    return batches,receipt


def train_from_manifest(manifest,labels,root,*,window_size,mode='attention',epochs=1,allow_experimental=False):
    batches,receipt=prepare_batches(manifest,labels,root,window_size=window_size,allow_experimental=allow_experimental)
    model,vectorizer,nb,report=train_baseline(batches,mode,epochs)
    report['intake_receipt']=receipt
    report['evaluation_unit']='causal window end; correlated windows are not independent episodes'
    report['process_efficacy_established']=False
    return model,vectorizer,nb,report
