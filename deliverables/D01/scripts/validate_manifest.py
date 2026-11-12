"""Validate the proposed D01 collection contract; never advances a simulator."""
import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def validate(data, root):
    root = Path(root).resolve()
    schema = json.loads((Path(__file__).resolve().parents[1] / 'data_manifest.schema.json').read_text())
    errors = [f'schema {list(e.path)}: {e.message}' for e in
              Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(data)]
    if errors:
        return errors
    if data['status'] != 'ready_for_review':
        errors.append('manifest is a draft, not ready_for_review')
    episodes, groups, hashes, paths = set(), {}, {}, {}
    splits = set()
    train_normal = False
    for episode in data['episodes']:
        eid, split = episode['episode_id'], episode['split']
        splits.add(split)
        if eid in episodes:
            errors.append(f'duplicate episode_id: {eid}')
        episodes.add(eid)
        group = episode['lineage_group']
        if group in groups and groups[group] != split:
            errors.append(f'lineage group crosses splits: {group}')
        groups[group] = split
        if split == 'train' and episode['label_summary'] == 'normal' and episode['iforest_train']:
            train_normal = True
        if episode['iforest_train'] and (split != 'train' or episode['label_summary'] != 'normal'):
            errors.append(f'Isolation Forest train must contain normal episodes only in train split: {eid}')
        if not episode['artifacts']['timeseries']:
            errors.append(f'missing timeseries: {eid}')
        if episode['event_end_step'] < episode['event_start_step']:
            errors.append(f'reversed event interval: {eid}')
        if episode['event_end_step'] >= episode['step_count']:
            errors.append(f'event interval outside episode: {eid}')
        if split != 'excluded' and episode['label_summary'] == 'unknown':
            errors.append(f'unknown label cannot enter modelling splits: {eid}')
        for modality, artifacts in episode['artifacts'].items():
            for item in artifacts:
                p = (root / item['path']).resolve()
                if not p.is_relative_to(root):
                    errors.append(f'path outside dataset root: {item["path"]}')
                    continue
                if item['source_kind'] in ('derived', 'synthetic') and not item['derived_from']:
                    errors.append(f'derived/synthetic artifact needs source references: {eid}/{modality}')
                if any(ref != eid for ref in item['derived_from']):
                    errors.append(f'cross-episode derivation not permitted by v0.1: {eid}/{modality}')
                if item['alignment_max_error_steps'] > data['alignment_tolerance_steps']:
                    errors.append(f'alignment exceeds tolerance: {eid}/{modality}')
                for key, registry, label in [(str(p), paths, 'path'), (item['sha256'], hashes, 'hash')]:
                    if key in registry and registry[key] != split:
                        errors.append(f'artifact {label} crosses splits: {item["path"]}')
                    registry[key] = split
                if not p.is_file():
                    errors.append(f'missing artifact: {item["path"]}')
                elif hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
                    errors.append(f'artifact hash mismatch: {item["path"]}')
    if not {'train', 'validation', 'test'}.issubset(splits):
        errors.append('all three modelling splits required')
    if not train_normal:
        errors.append('normal training episodes required')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text(encoding='utf-8'),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        errors = validate(data, args.root)
    except (OSError, ValueError) as exc:
        errors = [f'cannot read input: {exc}']
    print(json.dumps({'passed': not errors, 'errors': errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
