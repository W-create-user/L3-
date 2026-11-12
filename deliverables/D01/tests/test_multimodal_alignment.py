from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import subprocess
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from l3_system.monitoring.multimodal_alignment import align
from l3_system.monitoring.binding import make_binding,fingerprint


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'view.bin').write_bytes(b'synthetic visual fixture, not an actual image encoder input')
        (self.root/'text.txt').write_text('synthetic causal event fixture')
        def asset(path):return dict(path=path,sha256=hashlib.sha256((self.root/path).read_bytes()).hexdigest(),
            record_index=1,available_at_record=1,source_start_record=0,source_end_record=1,source_kind='derived',
            derived_from=['fixture'],generator_version='test-only',leakage_reviewed_by='synthetic-test',input_role='causal_input')
        self.episode={'episode_id':'fixture','lineage_group':'fixture-root','split':'excluded',
                      'series':[{'record_index':i,'observation':[float(i)]*39} for i in range(4)],
                      'visual':[asset('view.bin')],'text':[asset('text.txt')]}
        self.manifest={'schema_version':'D04Alignment/0.1','environment':'Atropine','observation_size':39,
                       'max_lag_records':{'visual':0,'text':0},'episodes':[self.episode]}
        def var(i):return dict(index=i,name=f'fixture-{i}',unit='fixture_unit',coordinate='fixture_absolute')
        binding=make_binding({'environment':'Atropine','observation':[var(i) for i in range(39)],
                              'action':[var(i) for i in range(4)]},'1'*64)
        self.manifest.update(binding=binding,binding_sha256=fingerprint(binding))

    def result(self):return align(self.manifest,self.root)
    def bad(self):
        with self.assertRaises(ValueError):self.result()

    def test_exact_alignment_and_mask(self):
        result=self.result();self.assertEqual(result['samples'][0]['modality_mask'],[True,False,False])
        self.assertEqual(result['samples'][1]['modality_mask'],[True,True,True])
        self.assertEqual(result['samples'][2]['modality_mask'],[True,False,False])
        self.assertFalse(result['summary']['trained_fusion_model'])

    def test_late_available_image_not_used_early(self):
        self.episode['visual'][0]['available_at_record']=2
        self.manifest['max_lag_records']['visual']=1
        result=self.result()['samples']
        self.assertFalse(result[1]['modality_mask'][1]);self.assertTrue(result[2]['modality_mask'][1])

    def test_future_asset_not_used(self):
        self.assertIsNone(self.result()['samples'][0]['visual'])

    def test_generated_future_content_refused(self):
        self.episode['visual'][0]['source_end_record']=2;self.bad()

    def test_missing_modalities_are_supported(self):
        self.episode['visual']=[];self.episode['text']=[]
        self.assertTrue(all(s['modality_mask']==[True,False,False] for s in self.result()['samples']))

    def test_same_source_group_cross_split_refused(self):
        other=deepcopy(self.episode);other.update(episode_id='other',split='test')
        self.episode['split']='train';self.manifest['episodes'].append(other);self.bad()

    def test_same_series_cross_split_refused(self):
        other=deepcopy(self.episode);other.update(episode_id='other',lineage_group='other-root',split='test')
        self.episode['split']='train';self.manifest['episodes'].append(other);self.bad()

    def test_asset_reuse_cross_split_refused(self):
        other=deepcopy(self.episode);other.update(episode_id='other',lineage_group='other-root',split='test')
        for row in other['series']:row['observation']=[v+10 for v in row['observation']]
        for modality in ('visual','text'):other[modality][0]['derived_from']=['other']
        self.episode['split']='train';self.manifest['episodes'].append(other);self.bad()

    def test_cross_episode_derivation_refused(self):
        self.episode['visual'][0]['derived_from']=['other'];self.bad()

    def test_hash_mismatch_refused(self):
        (self.root/'text.txt').write_text('changed');self.bad()

    def test_path_escape_refused(self):
        self.episode['visual'][0]['path']='../outside';self.bad()

    def test_label_asset_not_input(self):
        self.episode['text'][0]['input_role']='ground_truth';self.bad()

    def test_label_fields_not_observation_features(self):
        self.episode['series'][0]['label']=True;self.bad()

    def test_ambiguous_multiview_refused(self):
        self.episode['visual'].append(deepcopy(self.episode['visual'][0]));self.bad()

    def test_backend_and_dimensions_not_mixed(self):
        self.manifest['environment']='mAb';self.bad()

    def test_no_mutation_of_inputs(self):
        original=deepcopy(self.manifest);self.result();self.assertEqual(self.manifest,original)

    def test_content_review_declaration_required(self):
        self.episode['visual'][0]['leakage_reviewed_by']='';self.bad()

    def test_changed_binding_not_silently_accepted(self):
        original=deepcopy(self.manifest)
        self.manifest['binding']['observation'][0]['unit']='wrong';self.bad()
        self.manifest=original;self.manifest['binding']['observation']=None
        self.manifest['binding_sha256']=fingerprint(self.manifest['binding']);self.bad()

    def test_returned_values_do_not_alias_source(self):
        result=self.result();result['samples'][0]['observation'][0]=99
        self.assertEqual(self.episode['series'][0]['observation'][0],0)

    def test_cli_local_output(self):
        path=self.root/'input.json';path.write_text(json.dumps(self.manifest))
        result=subprocess.run([sys.executable,'-m','l3_system.monitoring.multimodal_alignment','--manifest',str(path),
                               '--root',str(self.root),'--output',str(self.root/'aligned.json')],
                              cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['sample_count'],4)


if __name__=='__main__':unittest.main()
