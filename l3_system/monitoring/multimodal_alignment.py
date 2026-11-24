"""Causal record-index alignment only; no encoder, fusion model or physical labels."""
import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from l3_system.agents.safety_rules import observation_integrity_error
from .binding import fingerprint


def _text(value):return isinstance(value,str) and bool(value.strip())


def align(manifest,root):
    root=Path(root).resolve()
    fields={'schema_version','environment','observation_size','binding','binding_sha256','max_lag_records','episodes'}
    if not isinstance(manifest,dict) or set(manifest)!=fields:raise ValueError('invalid alignment manifest fields')
    if manifest['schema_version']!='D04Alignment/0.1' or manifest['environment']!='Atropine' or manifest['observation_size']!=39:
        raise ValueError('v0.1 supports only explicit Atropine 39-variable records')
    binding=manifest['binding']
    if (not isinstance(binding,dict) or binding.get('schema_version')!='AnomalyBinding/1.0'
            or binding.get('environment')!='Atropine'
            or not isinstance(binding.get('observation'),list) or len(binding['observation'])!=39
            or not isinstance(binding.get('action'),list) or len(binding['action'])!=4
            or fingerprint(binding)!=manifest['binding_sha256']):
        raise ValueError('Atropine semantic binding and exact fingerprint required')
    config_hash=binding.get('collection_config_sha256')
    if not isinstance(config_hash,str) or not re.fullmatch('[0-9a-f]{64}',config_hash) or config_hash=='0'*64:
        raise ValueError('real collection configuration fingerprint required')
    for group in ('observation','action'):
        if any(not isinstance(v,dict) or set(v)!={'index','name','unit','coordinate'}
               or type(v['index']) is not int or any(not _text(v[k]) for k in ('name','unit','coordinate'))
               for v in binding[group]):raise ValueError('complete variable binding entries required')
        if [v.get('index') for v in binding[group]]!=list(range(len(binding[group]))):
            raise ValueError('binding variables must retain declared index order')
    lag=manifest['max_lag_records']
    if not isinstance(lag,dict) or set(lag)!={'visual','text'} or any(type(v) is not int or v<0 for v in lag.values()):
        raise ValueError('explicit nonnegative visual/text record lag budgets required')
    if not isinstance(manifest['episodes'],list) or not manifest['episodes']:raise ValueError('nonempty episode list required')
    ids=set();groups={};hash_splits={};path_splits={};samples=[];receipts=[]
    for episode in manifest['episodes']:
        if not isinstance(episode,dict) or set(episode)!={'episode_id','lineage_group','split','series','visual','text'}:
            raise ValueError('invalid episode fields; labels belong in separate sidecars')
        eid=episode['episode_id'];group=episode['lineage_group'];split=episode['split']
        if not _text(eid) or eid in ids or not _text(group) or split not in ('train','validation','test','excluded'):
            raise ValueError('unique episode, lineage and known split required')
        ids.add(eid)
        if group in groups and groups[group]!=split:raise ValueError('source lineage crosses splits')
        groups[group]=split
        series=episode['series']
        if not isinstance(series,list) or not series:raise ValueError('time series required')
        for i,row in enumerate(series):
            if (not isinstance(row,dict) or set(row)!={'record_index','observation'}
                    or type(row['record_index']) is not int or row['record_index']!=i
                    or observation_integrity_error(row['observation'],(39,))):
                raise ValueError('ordered finite 39-variable observations without label fields required')
        # Different IDs or number spellings cannot disguise an identical source series.
        canonical=[[float(v) for v in row['observation']] for row in series]
        series_hash=hashlib.sha256(json.dumps(canonical,allow_nan=False).encode()).hexdigest()
        if series_hash in hash_splits and hash_splits[series_hash]!=split:raise ValueError('same series crosses splits')
        hash_splits[series_hash]=split
        for modality in ('visual','text'):
            assets=episode[modality]
            if not isinstance(assets,list):raise ValueError('missing modality must be an explicit empty list')
            timestamps=set()
            for asset in assets:
                expected={'path','sha256','record_index','available_at_record','source_start_record','source_end_record',
                          'source_kind','derived_from','generator_version','leakage_reviewed_by','input_role'}
                if not isinstance(asset,dict) or set(asset)!=expected:raise ValueError('invalid asset fields')
                numbers=[asset[k] for k in ('record_index','available_at_record','source_start_record','source_end_record')]
                if any(type(v) is not int or v<0 for v in numbers):raise ValueError('nonnegative record indices required')
                record,available,start,end=numbers
                if not start<=end<=record<len(series) or available<record:
                    raise ValueError('asset contains future source records or invalid availability')
                key=(record,available)
                if key in timestamps:raise ValueError('ambiguous same-time assets; multiview needs a new contract')
                timestamps.add(key)
                if asset['source_kind'] not in ('native_simulation','authorized_external','derived','synthetic'):
                    raise ValueError('explicit asset origin required')
                if not isinstance(asset['derived_from'],list) or any(ref!=eid for ref in asset['derived_from']):
                    raise ValueError('cross-episode derivation not allowed')
                if asset['source_kind'] in ('derived','synthetic') and asset['derived_from']!=[eid]:
                    raise ValueError('derived modalities must identify current source episode')
                if not _text(asset['generator_version']) or not _text(asset['leakage_reviewed_by']) or asset['input_role']!='causal_input':
                    raise ValueError('generator and content-leakage review declaration required; label assets are not inputs')
                if not _text(asset['path']):raise ValueError('asset path required')
                path=(root/asset['path']).resolve()
                if not path.is_relative_to(root):raise ValueError('asset escapes dataset root')
                actual=hashlib.sha256(path.read_bytes()).hexdigest()
                if actual!=asset['sha256']:raise ValueError('asset hash mismatch')
                for registry,identity in ((hash_splits,actual),(path_splits,str(path))):
                    if identity in registry and registry[identity]!=split:raise ValueError('asset content/path crosses splits')
                    registry[identity]=split
                receipts.append(dict(episode_id=eid,modality=modality,path=asset['path'],sha256=actual,
                                     source_kind=asset['source_kind'],derived_from=list(asset['derived_from'])))
        for row in series:
            record=row['record_index'];chosen={}
            for modality in ('visual','text'):
                eligible=[a for a in episode[modality] if a['record_index']<=record and a['available_at_record']<=record
                          and record-a['record_index']<=lag[modality]]
                selected=max(eligible,key=lambda a:(a['record_index'],a['available_at_record'])) if eligible else None
                chosen[modality]=({k:selected[k] for k in ('path','sha256','record_index','source_kind')}
                                  if selected else None)
            samples.append(dict(episode_id=eid,lineage_group=group,split=split,record_index=record,
                                observation=list(row['observation']),visual=chosen['visual'],text=chosen['text'],
                                modality_mask=[True,chosen['visual'] is not None,chosen['text'] is not None]))
    return {'schema_version':'D04AlignedIndex/0.1','modality_order':['timeseries','visual','text'],
            'binding':deepcopy(binding),'binding_sha256':manifest['binding_sha256'],
            'samples':samples,'asset_sources':receipts,
            'summary':{'episode_count':len(ids),'sample_count':len(samples),
                       'visual_present_samples':sum(s['modality_mask'][1] for s in samples),
                       'text_present_samples':sum(s['modality_mask'][2] for s in samples),
                       'trained_fusion_model':False,'accepted_for_training':False,
                       'limits':'Record mapping and content review are declarations. Hashes do not prove authorization, physical time sync or absence of label leakage inside assets.'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def invalid(value):raise ValueError(f'nonfinite JSON: {value}')
    try:
        manifest=json.loads(args.manifest.read_text(encoding='utf-8'),parse_constant=invalid)
        result=align(manifest,args.root)
        # Output can contain numerical observations: local use only unless separately authorized.
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except (OSError,UnicodeError,ValueError) as exc:parser.exit(1,f'Alignment failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':main()
