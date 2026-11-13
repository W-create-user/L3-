import unittest
import torch
from l3_system.monitoring.fusion_comparison import compare,subset,VARIANTS
from l3_system.monitoring.multimodal_model import synthetic_batches,train_baseline,FusionBaseline

class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_seven_variants_actually_train(self):
        result=compare(synthetic_batches())
        self.assertEqual(set(result['variants']),set(VARIANTS))
        for value in result['variants'].values():
            self.assertEqual(value['threshold_source'],'validation_only')
            self.assertEqual(sum(value['test'][key] for key in ('tp','fp','tn','fn')),value['sample_counts']['test']['eligible_samples'])
            self.assertGreater(value['losses'][0],0)
        self.assertFalse(result['paired_comparison_established'])

    def test_missing_variant_samples_counted_not_filled(self):
        batches=synthetic_batches();original=batches['test'].mask.clone()
        batch,counts=subset(batches['test'],(1,))
        self.assertEqual(counts['abstained_samples'],1);self.assertEqual(len(batch.labels),5)
        self.assertTrue(torch.equal(original,batches['test'].mask))
        self.assertFalse(batch.mask[:,0].any());self.assertFalse(batch.mask[:,2].any())

    def test_frozen_backbone_and_statistics_unchanged(self):
        torch.manual_seed(17);initial=FusionBaseline(4)
        # Vocabulary size affects only later layers, not the visual initialization.
        trained,_,_,report=train_baseline(synthetic_batches())
        self.assertTrue(report['visual_backbone_frozen'])
        for key,value in initial.vision.state_dict().items():
            if not key.startswith('fc.'):self.assertTrue(torch.equal(value,trained.vision.state_dict()[key]),key)
        self.assertFalse(trained.vision.conv1.weight.requires_grad)
        self.assertTrue(trained.vision.fc.weight.requires_grad)

if __name__=='__main__':unittest.main()
