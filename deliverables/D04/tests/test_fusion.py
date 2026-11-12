import unittest
import torch
from l3_system.monitoring.multimodal_model import FusionBaseline,synthetic_batches,split_check,train_baseline

class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_missing_padding_is_ignored(self):
        for mode in ('concat','attention'):
            torch.manual_seed(2);model=FusionBaseline(5,mode).eval()
            x=torch.randn(2,4,39);image=torch.randn(2,3,32,32);text=torch.randn(2,5)
            mask=torch.tensor([[True,False,False],[True,True,True]])
            with torch.no_grad():
                first,weights=model(x,image,text,mask)
                image[0]=float('nan');text[0]=float('nan')
                second,_=model(x,image,text,mask)
            self.assertTrue(torch.equal(first,second))
            self.assertTrue(torch.equal(weights[0,1:],torch.zeros(2)))
            if mode=='attention':self.assertTrue(torch.allclose(weights.sum(1),torch.ones(2)))

    def test_all_missing_abstains(self):
        model=FusionBaseline(5)
        with self.assertRaises(ValueError):model(torch.zeros(2,4,39),torch.zeros(2,3,32,32),torch.zeros(2,5),torch.zeros(2,3,dtype=torch.bool))

    def test_lineage_leak_rejected(self):
        batches=synthetic_batches();batches['test'].groups[0]=batches['train'].groups[0]
        with self.assertRaisesRegex(ValueError,'lineage'):split_check(batches)

    def test_missing_text_does_not_fit_vocabulary(self):
        batches=synthetic_batches();batches['train'].texts[1]='secretword'
        _,vectorizer,_,report=train_baseline(batches)
        self.assertNotIn('secretword',vectorizer.vocabulary_)
        self.assertEqual(report['threshold_source'],'validation_only')
        self.assertEqual(report['text_naive_bayes']['abstained_samples'],1)

    def test_test_split_cannot_change_fit_or_threshold(self):
        first=synthetic_batches();second=synthetic_batches()
        second['test'].series+=1000
        second['test'].texts=['unseentestword']*6
        second['test'].labels=1-second['test'].labels
        m1,v1,_,r1=train_baseline(first)
        m2,v2,_,r2=train_baseline(second)
        self.assertNotIn('unseentestword',v2.vocabulary_)
        self.assertEqual(v1.vocabulary_,v2.vocabulary_)
        self.assertEqual(r1['threshold'],r2['threshold'])
        self.assertEqual(r1['losses'],r2['losses'])
        for key,value in m1.state_dict().items():self.assertTrue(torch.equal(value,m2.state_dict()[key]))
        self.assertTrue(torch.allclose(m1.series_mean,first['train'].series.mean(dim=(0,1))))

    def test_nonfinite_available_and_invalid_labels_rejected(self):
        batches=synthetic_batches();batches['test'].series[0,0,0]=float('nan')
        with self.assertRaises(ValueError):split_check(batches)
        batches=synthetic_batches();batches['validation'].labels[:]=0
        with self.assertRaises(ValueError):split_check(batches)

    def test_both_models_actually_train(self):
        for mode in ('concat','attention'):
            model,_,_,report=train_baseline(synthetic_batches(),mode)
            self.assertTrue(report['losses'][0]>0)
            self.assertEqual(sum(report['test'][k] for k in ('tn','fp','fn','tp')),6)
            self.assertEqual(report['ablations']['image_only']['abstained_samples'],1)
            self.assertFalse(report['pretrained_visual_weights'])

if __name__=='__main__':unittest.main()
