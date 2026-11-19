import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validator', BASE/'scripts/validate_manifest.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = json.loads((BASE/'examples/data_manifest.draft.json').read_text())
        self.data.update(status='ready_for_review', collection_config_sha256='1'*64,
                         variable_dictionary_sha256='2'*64,
                         permission_evidence='synthetic unit-test fixtures only')
        for i, split in enumerate(['train', 'validation', 'test']):
            file = self.root/f'{split}.jsonl'
            file.write_text(json.dumps({'synthetic_fixture': True, 'index': i})+'\n')
            self.data['episodes'].append(dict(
                episode_id=f'fixture-{i}', lineage_group=f'root-{i}', split=split,
                environment='Atropine', environment_config_sha256='3'*64, seed=i,
                initial_state_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                scenario_id='synthetic-fixture', step_count=10, label_summary='normal',
                iforest_train=(split == 'train'),
                label_source='unit-test fixture declaration', event_start_step=0,
                event_end_step=0, artifacts=dict(timeseries=[dict(
                    path=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    source_kind='synthetic', derived_from=[f'fixture-{i}'],
                    generator_version='test-fixture/1', alignment_max_error_steps=0,
                    label_free_input_confirmed=True)], visual=[], text=[])))

    def check_bad(self, substring):
        errors = validator.validate(self.data, self.root)
        self.assertTrue(any(substring in e for e in errors), errors)

    def test_valid_contract(self):
        self.assertEqual(validator.validate(self.data, self.root), [])

    def test_draft_rejected(self):
        self.data['status'] = 'draft'; self.check_bad('draft')

    def test_empty_splits_rejected(self):
        self.data['episodes'] = []; self.check_bad('all three')

    def test_duplicate_episode(self):
        self.data['episodes'][1]['episode_id'] = 'fixture-0'; self.check_bad('duplicate episode')

    def test_lineage_leakage(self):
        self.data['episodes'][1]['lineage_group'] = 'root-0'; self.check_bad('lineage group crosses')

    def test_identical_bytes_leakage(self):
        a,b = [e['artifacts']['timeseries'][0] for e in self.data['episodes'][:2]]
        (self.root/b['path']).write_bytes((self.root/a['path']).read_bytes())
        b['sha256'] = a['sha256']; self.check_bad('hash crosses splits')

    def test_artifact_corruption(self):
        (self.root/'train.jsonl').write_text('changed'); self.check_bad('hash mismatch')

    def test_missing_artifact(self):
        (self.root/'test.jsonl').unlink(); self.check_bad('missing artifact')

    def test_path_escape(self):
        self.data['episodes'][0]['artifacts']['timeseries'][0]['path'] = '../outside'
        self.check_bad('outside dataset root')

    def test_train_contamination(self):
        self.data['episodes'][0]['label_summary'] = 'injected_anomaly'; self.check_bad('normal episodes only')

    def test_iforest_cannot_use_validation(self):
        self.data['episodes'][1]['iforest_train'] = True; self.check_bad('normal episodes only')

    def test_supervised_train_allows_labeled_anomaly(self):
        episode = copy.deepcopy(self.data['episodes'][0])
        episode.update(episode_id='extra', lineage_group='extra', label_summary='injected_anomaly', iforest_train=False)
        episode['artifacts']['timeseries'][0]['derived_from'] = ['extra']
        self.data['episodes'].append(episode)
        self.assertEqual(validator.validate(self.data, self.root), [])

    def test_zero_placeholder_hash(self):
        self.data['variable_dictionary_sha256'] = '0'*64; self.check_bad('schema')

    def test_unknown_label(self):
        self.data['episodes'][1]['label_summary'] = 'unknown'; self.check_bad('unknown label')

    def test_alignment(self):
        self.data['episodes'][0]['artifacts']['timeseries'][0]['alignment_max_error_steps'] = 1
        self.check_bad('alignment exceeds')

    def test_label_as_input(self):
        self.data['episodes'][0]['artifacts']['timeseries'][0]['label_free_input_confirmed'] = False
        self.check_bad('schema')

    def test_cross_episode_derivation(self):
        self.data['episodes'][1]['artifacts']['timeseries'][0]['derived_from'] = ['fixture-0']
        self.check_bad('cross-episode derivation')

    def test_no_source_reference(self):
        self.data['episodes'][0]['artifacts']['timeseries'][0]['derived_from'] = []
        self.check_bad('source references')

    def test_bad_event_interval(self):
        self.data['episodes'][0].update(event_start_step=5,event_end_step=3)
        self.check_bad('reversed event')

    def test_event_outside_episode(self):
        self.data['episodes'][0]['event_end_step'] = 10; self.check_bad('outside episode')

    def test_missing_field(self):
        del self.data['variable_dictionary_sha256']; self.check_bad('schema')

    def test_missing_timeseries(self):
        self.data['episodes'][0]['artifacts']['timeseries'] = []; self.check_bad('missing timeseries')


if __name__ == '__main__':
    unittest.main()
