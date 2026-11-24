from copy import deepcopy
import unittest
import numpy as np
import torch
from l3_system.monitoring.sensor_fusion_data import build,render

class DerivedDataTests(unittest.TestCase):
    def episodes(self):
        rng=np.random.default_rng(11);out=[]
        for split in ('train','validation','test'):
            x=rng.normal(size=(30,38)).tolist()
            out.append(dict(episode_id=split,lineage_group=split,split=split,observations=x,
                labels=[split!='train' and 9<=i<20 for i in range(30)],label_scope='sensor_fault'))
        return out
    def test_sources_unchanged_and_augmentation_stays_in_train(self):
        source=self.episodes();copy=deepcopy(source);b,r=build(source)
        self.assertEqual(source,copy);self.assertEqual(r['windows'],{'train':54,'validation':27,'test':27})
        self.assertEqual(set(b['train'].labels.tolist()),{0.,1.})
        self.assertTrue((b['test'].series[:,:,0]==0).all())
        self.assertEqual(set(b['train'].groups),{'train'})
    def test_heldout_changes_do_not_change_train(self):
        source=self.episodes();first,_=build(source);source[-1]['observations'][20][4]+=9
        second,_=build(source);self.assertTrue(torch.equal(first['train'].images,second['train'].images))
    def test_render_has_no_label_or_future_argument(self):
        source=self.episodes();a=np.array(source[-1]['observations']);window=a[:4].copy()
        first=render(window,np.zeros(38),np.ones(38));a[20]+=20
        second=render(a[:4],np.zeros(38),np.ones(38));self.assertTrue(torch.equal(first[0],second[0]));self.assertEqual(first[1],second[1])
    def test_cross_split_lineage_rejected(self):
        source=self.episodes();source[-1]['lineage_group']='train'
        with self.assertRaises(ValueError):build(source)
    def test_label_changes_do_not_change_derived_inputs(self):
        source=self.episodes();first,_=build(source)
        source[-1]['labels']=[not v for v in source[-1]['labels']]
        second,_=build(source)
        self.assertTrue(torch.equal(first['test'].images,second['test'].images))
        self.assertEqual(first['test'].texts,second['test'].texts)
    def test_identical_cross_split_assets_are_masked(self):
        source=self.episodes();source[1]['observations']=deepcopy(source[0]['observations'])
        batches,receipt=build(source)
        self.assertFalse(batches['validation'].mask[:,1:].any())
        self.assertEqual(receipt['cross_split_identical_assets_masked']['validation']['image'],27)
