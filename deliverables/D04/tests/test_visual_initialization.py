from pathlib import Path
import tempfile,unittest
import torch
from torchvision.models import resnet18
from l3_system.monitoring.multimodal_model import train_baseline,synthetic_batches
from l3_system.monitoring.fusion_bundle import save_bundle,load_trusted_bundle
from l3_system.monitoring.binding import make_binding

class ExplicitVisualInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(18)
        # Random fixture tensors test the explicit-state path without network.
        # Actual ImageNet provenance is verified by the separate study receipt.
        cls.visual=resnet18(weights=None).state_dict();cls.batches=synthetic_batches()
        cls.model,cls.vectorizer,cls.nb,cls.report=train_baseline(cls.batches,visual_state=cls.visual)
    def test_explicit_trunk_is_loaded_frozen_and_normalized(self):
        self.assertTrue(torch.equal(self.model.vision.conv1.weight,self.visual['conv1.weight']))
        self.assertFalse(self.model.vision.conv1.weight.requires_grad)
        self.assertTrue(self.model.vision.fc.weight.requires_grad)
        self.assertTrue(self.report['pretrained_visual_weights'])
        self.assertFalse(self.model.vision.training)
    def test_explicit_visual_mode_survives_trusted_bundle_restoration(self):
        def var(i):return dict(index=i,name=f'fixture-{i}',unit='fixture',coordinate='fixture')
        binding=make_binding({'environment':'Atropine','observation':[var(i) for i in range(39)],'action':[var(i) for i in range(4)]},'2'*64)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle';receipt=save_bundle(path,self.model,self.vectorizer,self.nb,binding=binding,label_scope='synthetic_fixture')
            restored,rv,nb,card=load_trusted_bundle(path,expected_card_sha256=receipt['card_sha256'],expected_binding=binding)
            self.assertTrue(card['pretrained_visual_weights']);self.assertTrue(restored.visual_pretrained)
            b=self.batches['test'];x=torch.tensor(rv.transform(b.texts).toarray()).float()
            with torch.no_grad():
                original=self.model(b.series,b.images,x,b.mask)[0];again=restored(b.series,b.images,x,b.mask)[0]
            self.assertTrue(torch.equal(original,again))

if __name__=='__main__':unittest.main()
