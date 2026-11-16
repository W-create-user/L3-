"""Rebuild A's model-coordinate registry from pinned B source dictionaries.

Uses byte-identical B dictionary snapshots; no numeric thresholds invented.
"""
import hashlib
import json
from pathlib import Path
from l3_system.planning.goals import validate_registry, SCHEMA_VERSION

B_COMMIT = 'e0e12ec1aa577a66578887f436237dc14d25ee9d'
PREFIX = 'handoffs/B_to_D_20261112/'
ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {'observation_schema_verified.json': 'f70731fac3f3ca855b830e41962b158feb98358378b1d89e3cdb8be0efa3ed67', 'action_dictionary.json': '0d680df0821d7dda1d67637f4dfe5ceef7888a7ab21fe904a3a417d40f53c14c'}


def build():
    sources = {}
    def read(name):
        path = PREFIX + name
        raw = (ROOT/'configs'/'goals'/'sources'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError('B source snapshot hash mismatch')
        sources[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    observations = read('observation_schema_verified.json')
    actions = read('action_dictionary.json')
    if observations['backend'] != 'AtropineEnvGym' or observations['normalize'] is not False:
        raise ValueError('unsupported backend/coordinates')
    variables, excluded = {}, []
    for item in observations['fields']:
        if item['unit_status'] != 'source_documented_not_field_calibrated':
            excluded.append(dict(name=item['name'], reason=item['unit_status']))
            continue
        variables[item['name']] = dict(source='observation', index=item['index'], unit=item['unit'],
            coordinate='model_observation_normalize_false',
            bounds=dict(lower=item['model_box_low'], upper=item['model_box_high']))
    action_notes = {}
    for item in actions:
        if item['backend'] != 'AtropineEnvGym':
            continue
        if item['normalize'] is not False or item['request_scale'] != 'delta_from_USS':
            raise ValueError('unsupported action coordinates')
        variables[item['name']] = dict(source='action', index=item['index'], unit=item['unit'],
            coordinate=item['request_scale'], bounds=dict(lower=item['low'], upper=item['high']))
        action_notes[item['name']] = {key: item[key] for key in ('unit_status', 'physical_value_formula', 'plant_L_per_min_formula')}
    registry = validate_registry(dict(schema_version=SCHEMA_VERSION, environment_id='AtropineEnvGym',
        registry_version='B-e0e12ec-atropine-1',
        evidence_ref=f'https://github.com/W-create-user/L3-/tree/{B_COMMIT}/{PREFIX}',
        observation_size=observations['dimension'], action_size=4, variables=variables))
    notes = dict(source_commit=B_COMMIT, source_sha256=sources, excluded_observations=excluded,
                 action_unit_notes=action_notes, r03_enabled=False,
                 limitations=['Model Box is not a process-safety threshold.',
                     'Action units retain B source-inference qualifications.',
                     'No approved production target or mAb per-variable registry supplied.'])
    return registry, notes


def main():
    registry, notes = build()
    for name, value in [('atropine_model_registry.json', registry), ('atropine_registry_provenance.json', notes)]:
        (ROOT/'configs'/'goals'/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
