"""Consistency checks for the non-executable Atropine research draft v0.1.

Passing validates draft structure, not a physical threshold or permission to run.
"""
import argparse
import json
from pathlib import Path


def read(path):
    def invalid(value):raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def validate(catalog,protocol):
    errors=[]
    def require(value,message):
        if not value:errors.append(message)
    if not isinstance(catalog,dict) or not isinstance(protocol,dict):
        return ['catalog and protocol must be objects']
    require(catalog.get('schema_version')=='D02CandidateCatalog/0.1','unsupported catalog schema')
    require(protocol.get('schema_version')=='D02CollectionProtocol/0.1','unsupported collection schema')
    require(catalog.get('status')=='draft' and protocol.get('status')=='draft','v0.1 must remain a draft')
    require(catalog.get('runtime_export_allowed') is False,'candidate catalog is not a runtime rule configuration')
    require(protocol.get('execution_authorized') is False,'collection cannot be authorized by this draft')
    require(protocol.get('R03_enabled') is False,'R-03 must remain disabled')
    source=catalog.get('source',{})
    require(source==protocol.get('source'),'catalog and protocol must pin the same source/configuration')
    require(source.get('environment')=='AtropineEnvGym' and protocol.get('environment')=='AtropineEnvGym',
            'Atropine protocol cannot be reused for another backend')
    require(source.get('observation_shape')==[39] and source.get('action_shape')==[4],'Atropine shape contract required')
    require(source.get('normalize') is False and source.get('uss_subtracted') is True,'Atropine coordinate mode mismatch')
    candidates=catalog.get('candidates',[])
    if not isinstance(candidates,list):return errors+['candidates must be a list']
    require(len(candidates)==8,'first batch must contain the declared eight proposals')
    ids=set()
    thresholds={'range':{'lower','upper'},'rate':{'max_rate_per_second'},'freshness':{'max_age_seconds'}}
    expected={'R-03-EFACTOR':('range','observation',6),'R-03-Q-ORGANIC':('range','observation',37),
              'R-03-Q-AQUEOUS':('range','observation',38),'R-03-FRESHNESS':('freshness','context',None),
              **{f'R-03-RATE-A{i}':('rate','action',i) for i in range(4)}}
    for item in candidates:
        if not isinstance(item,dict):errors.append('candidate must be an object');continue
        rid=item.get('rule_id')
        if not isinstance(rid,str):errors.append('rule_id must be a string');continue
        require(rid not in ids,'duplicate candidate rule_id');ids.add(rid)
        require(rid in expected,'unknown first-batch rule_id')
        require(item.get('enabled') is False,'unconfirmed candidate must be disabled')
        kind=item.get('kind');threshold=item.get('threshold')
        require(kind in thresholds,'unsupported proposed rule kind')
        if kind in thresholds:
            require(isinstance(threshold,dict) and set(threshold)==thresholds[kind]
                    and all(v is None for v in threshold.values()),'v0.1 cannot fill unverified thresholds')
        if rid in expected:
            require((kind,item.get('source'),item.get('index'))==expected[rid],'candidate variable/kind mismatch')
        if kind=='rate':require(item.get('coordinate')=='delta_from_USS','candidate must retain B delta coordinates')
        require(isinstance(item.get('evidence'),list) and bool(item['evidence']),'candidate needs traceable source evidence')
        require(bool(item.get('acceptance_owner')) and bool(item.get('required_inputs')),'candidate needs owners and pending inputs')
    require(ids==set(expected),'candidate coverage differs from declared first batch')
    families=protocol.get('scenario_families',[])
    if not isinstance(families,list):return errors+['scenario families must be a list']
    by_id={f.get('id'):f for f in families if isinstance(f,dict) and isinstance(f.get('id'),str)}
    require(len(by_id)==len(families) and set(by_id)=={'INTERFACE','INPUT_RESPONSE','SENSOR','PROCESS'},
            'interface, input response, sensor and process families must remain separate')
    for family in by_id.values():
        for key in ('amplitude','amplitude_unit','start_step','end_step','process_anomaly_label'):
            require(key in family and family[key] is None,'unreviewed injection parameters/labels must remain null')
    require(by_id.get('PROCESS',{}).get('mode')=='blocked_pending_physical_injection_design'
            and by_id.get('PROCESS',{}).get('injection_site') is None,'physical injection point not yet established')
    require(by_id.get('SENSOR',{}).get('mode')=='offline_replay_plan','sensor pilot is offline replay, not closed-loop evidence')
    labels=protocol.get('label_contract',{})
    require(labels.get('initial_process_anomaly_label') is None,'no process anomaly truth is available by default')
    require(labels.get('preflight_and_model_verdict_cannot_define_ground_truth') is True,'model verdict must not define truth')
    split=protocol.get('split_policy',{})
    for flag in ('group_before_windowing','derived_variants_follow_source_group','normal_training_only_for_iforest'):
        require(split.get(flag) is True,'grouping and normal-only IF training safeguards required')
    require(split.get('seed_alone_is_independence_evidence') is False and split.get('test_set_retuning_allowed') is False,
            'seed identity does not prove independence; test retuning forbidden')
    budget=protocol.get('pilot_budget_proposal',{})
    require(budget.get('independence_claim') is False and budget.get('normal_label_auto_assign') is False,
            'pilot does not establish independence or normal labels')
    phases=budget.get('phases_proposed',[])
    if isinstance(phases,list) and all(isinstance(p,dict) and type(p.get('records')) is int and p['records']>0 for p in phases):
        require(sum(p['records'] for p in phases)==budget.get('cycles_per_base_episode'),'phase counts must equal proposed episode budget')
    else:errors.append('positive phase counts required')
    require(protocol.get('clock_policy',{}).get('freshness_limit_seconds') is None,'sampling period cannot define freshness limit')
    require(protocol.get('stop_policy',{}).get('keep_R01_R02_and_runtime_guards') is True,'existing hard guards must remain enabled')
    require(protocol.get('scientific_claims_allowed') is False,'draft does not support performance claims')
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    try:errors=validate(read(args.catalog),read(args.protocol))
    except (OSError,ValueError) as exc:errors=[f'cannot read research draft: {type(exc).__name__}']
    print(json.dumps({'draft_consistent':not errors,'errors':errors,'collection_authorized':False,
                      'R03_enabled':False,'physical_thresholds_confirmed':False},ensure_ascii=False,indent=2))
    raise SystemExit(1 if errors else 0)


if __name__=='__main__':main()
