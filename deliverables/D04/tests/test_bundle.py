from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from l3_system.monitoring.multimodal_model import train_baseline,synthetic_batches
from l3_system.monitoring.fusion_bundle import save_bundle,load_trusted_bundle
from l3_system.monitoring.binding import make_binding

class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.batches=synthetic_batches();cls.model,cls.vectorizer,cls.nb,_=train_baseline(cls.batches)
        def var(i):return dict(index=i,name=f'fixture-{i}',unit='fixture',coordinate='fixture')
        cls.binding=make_binding({'environment':'Atropine','observation':[var(i) for i in range(39)],'action':[var(i) for i in range(4)]},'1'*64)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'model'
        self.receipt=save_bundle(self.root,self.model,self.vectorizer,self.nb,binding=self.binding,label_scope='synthetic_fixture')
    def load(self,**kw):
        return load_trusted_bundle(self.root,expected_card_sha256=self.receipt['card_sha256'],expected_binding=kw.get('binding',self.binding))
    def test_roundtrip_predictions_identical(self):
        model,vectorizer,nb,card=self.load();batch=self.batches['test']
        original=self.vectorizer.transform(batch.texts).toarray();restored=vectorizer.transform(batch.texts).toarray()
        np.testing.assert_allclose(original,restored,rtol=0,atol=0)
        np.testing.assert_array_equal(self.nb.predict(original),nb.predict(restored))
        with torch.no_grad():
            first=self.model(batch.series,batch.images,torch.tensor(original).float(),batch.mask)[0]
            second=model(batch.series,batch.images,torch.tensor(restored).float(),batch.mask)[0]
        self.assertTrue(torch.equal(first,second));self.assertFalse(card['deployment_authorized'])
    def test_tampering_rejected_before_checkpoint_load(self):
        (self.root/'text.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash'):self.load()
    def test_wrong_trusted_receipt_rejected(self):
        self.receipt['card_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'fingerprint'):self.load()
    def test_semantic_mismatch_rejected(self):
        binding=deepcopy(self.binding);binding['observation'][0]['unit']='other'
        with self.assertRaises(ValueError):self.load(binding=binding)
    def test_dependency_mismatch_rejected_even_with_new_card_hash(self):
        import json,hashlib
        path=self.root/'card.json';card=json.loads(path.read_text());card['versions']['numpy']='different'
        path.write_text(json.dumps(card))
        self.receipt['card_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'version'):self.load()

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):save_bundle(self.root,self.model,self.vectorizer,self.nb,binding=self.binding,label_scope='synthetic_fixture')

if __name__=='__main__':unittest.main()
