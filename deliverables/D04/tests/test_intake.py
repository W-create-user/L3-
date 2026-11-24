from copy import deepcopy
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from l3_system.monitoring.binding import make_binding,fingerprint
from l3_system.monitoring.multimodal_intake import prepare_batches,train_from_manifest

class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        episodes=[];entries=[]
        for j,split in enumerate(('train','validation','test')):
            for y in (False,True):
                eid=f'{split}-{int(y)}';image=f'{eid}.png';text=f'{eid}.txt'
                Image.new('RGB',(40,40),(j*50+int(y)*20,10,30)).save(self.root/image)
                (self.root/text).write_text(f'causal fixture {eid}')
                def asset(path):return dict(path=path,sha256=hashlib.sha256((self.root/path).read_bytes()).hexdigest(),
                    record_index=0,available_at_record=1,source_start_record=0,source_end_record=0,
                    source_kind='synthetic',derived_from=[eid],generator_version='test',leakage_reviewed_by='test',input_role='causal_input')
                episodes.append(dict(episode_id=eid,lineage_group=eid,split=split,
                    series=[dict(record_index=i,observation=[float(j*10+int(y)+i/10)]*39) for i in range(3)],
                    visual=[asset(image)],text=[asset(text)]))
                entries.append(dict(episode_id=eid,lineage_group=eid,split=split,record_labels=[y]*3))
        def var(i):return dict(index=i,name=f'v{i}',unit='fixture',coordinate='fixture')
        binding=make_binding({'environment':'Atropine','observation':[var(i) for i in range(39)],'action':[var(i) for i in range(4)]},'1'*64)
        self.manifest=dict(schema_version='D04Alignment/0.1',environment='Atropine',observation_size=39,binding=binding,
            binding_sha256=fingerprint(binding),max_lag_records={'visual':2,'text':2},episodes=episodes)
        self.labels=dict(schema_version='D04Labels/0.1',alignment_sha256=fingerprint(self.manifest),
            label_scope='synthetic_fixture',source_kind='synthetic_fixture',source_ref='test',reviewed_by='test',episodes=entries)

    def run_intake(self,**kw):return prepare_batches(self.manifest,self.labels,self.root,window_size=2,allow_experimental=True,**kw)

    def test_causal_windows_and_raw_labels(self):
        batches,receipt=self.run_intake();batch=batches['train']
        self.assertEqual(batch.series.shape,(4,2,39));self.assertEqual(batch.labels.tolist(),[0,0,1,1])
        self.assertAlmostEqual(float(batch.series[0,-1,0]),.1,places=5)
        self.assertEqual(receipt['warmup_records_excluded'],6)
        self.assertFalse(receipt['deployment_authorized'])
        self.assertTrue(batch.mask.all())

    def test_missing_modalities_keep_masks(self):
        for episode in self.manifest['episodes']:episode['visual']=[];episode['text']=[]
        self.labels['alignment_sha256']=fingerprint(self.manifest)
        batches,_=self.run_intake()
        self.assertEqual(batches['test'].mask.tolist(),[[True,False,False]]*4)

    def test_missing_or_unknown_labels_rejected(self):
        self.labels['episodes'][0]['record_labels'][0]=None
        with self.assertRaises(ValueError):self.run_intake()
        self.labels['episodes'][0]['record_labels'][0]=False;self.labels['episodes'].pop()
        with self.assertRaises(ValueError):self.run_intake()

    def test_label_binding_and_split_rejected(self):
        self.labels['episodes'][0]['split']='test'
        with self.assertRaises(ValueError):self.run_intake()
        self.labels['episodes'][0]['split']='train';self.labels['alignment_sha256']='0'*64
        with self.assertRaises(ValueError):self.run_intake()

    def test_asset_tamper_rejected(self):
        (self.root/'train-0.txt').write_text('changed')
        with self.assertRaises(ValueError):self.run_intake()

    def test_no_future_rows_in_windows(self):
        self.manifest['episodes'][0]['series'][2]['observation']=[999.]*39
        self.labels['alignment_sha256']=fingerprint(self.manifest)
        batches,_=self.run_intake()
        self.assertLess(float(batches['train'].series[0].max()),1)

    def test_decode_invalid_image_rejected(self):
        path=self.root/'train-0.png';path.write_bytes(b'not an image')
        self.manifest['episodes'][0]['visual'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        self.labels['alignment_sha256']=fingerprint(self.manifest)
        with self.assertRaises(OSError):self.run_intake()

    def test_asset_changed_between_alignment_and_decode_rejected(self):
        from l3_system.monitoring.multimodal_alignment import align
        def race(manifest,root):
            result=align(manifest,root)
            (self.root/'train-0.txt').write_text('changed after verification')
            return result
        with patch('l3_system.monitoring.multimodal_intake.align',side_effect=race):
            with self.assertRaisesRegex(ValueError,'changed after alignment'):self.run_intake()

    def test_unknown_provenance_cannot_pass_with_null_kind(self):
        self.labels['label_scope']='unknown';self.labels['source_kind']=None
        with self.assertRaises(ValueError):self.run_intake()

    def test_opt_in_required(self):
        with self.assertRaises(ValueError):prepare_batches(self.manifest,self.labels,self.root,window_size=2)

    def test_end_to_end_training(self):
        _,_,_,report=train_from_manifest(self.manifest,self.labels,self.root,window_size=2,allow_experimental=True)
        self.assertFalse(report['process_efficacy_established'])
        self.assertEqual(report['intake_receipt']['label_scope'],'synthetic_fixture')
        self.assertEqual(report['threshold_source'],'validation_only')

if __name__=='__main__':unittest.main()
