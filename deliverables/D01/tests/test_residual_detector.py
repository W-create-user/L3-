import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from l3_system.monitoring.residual_detector import ResidualDetector

def episodes():
    rng=np.random.default_rng(416);rows=[]
    for i,role in enumerate(('train','train','calibration','calibration','threshold','threshold')):
        x=rng.normal(0,.01,(32,39));x[:,5]=10+rng.normal(0,.01,32);x[:,6]=x[:,5]*(1+rng.normal(0,.001,32))
        labels=[False]*32
        if i in (3,5):
            x[10:20,6]*=1.2;labels[10:20]=[True]*10
        rows.append(dict(episode_id=str(i),lineage_group=str(i),split='train' if role=='train' else 'validation',
                         fit_role=role,label_scope='sensor_fault',observations=x.tolist(),labels=labels))
    return rows

class ResidualDetectorTests(unittest.TestCase):
    def test_fit_refuses_test_and_cross_role_lineage(self):
        rows=episodes();rows[-1]['split']='test'
        with self.assertRaises(ValueError):ResidualDetector().fit(rows,binding={})
        rows=episodes();rows[-1]['lineage_group']=rows[2]['lineage_group']
        with self.assertRaises(ValueError):ResidualDetector().fit(rows,binding={})
    def test_clock_and_unused_channels_cannot_change_scores(self):
        rows=episodes();m=ResidualDetector();m.fit(rows,binding={})
        x=np.array(rows[0]['observations'][:4]);a=m.score(x)
        x[:,0]=1e9;x[:,10:]=123456.
        self.assertEqual(a,m.score(x))
    def test_probability_and_trusted_artifact_roundtrip(self):
        rows=episodes();m=ResidualDetector();m.fit(rows,binding={})
        result=m.evaluate(rows[-2:]);self.assertGreaterEqual(result['brier_score'],0)
        self.assertLessEqual(result['brier_score'],1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pkl';sha=m.save(path);restored=ResidualDetector.load_trusted(path,sha)
            self.assertEqual(m.score(rows[0]['observations'][:4]),restored.score(rows[0]['observations'][:4]))
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValueError):ResidualDetector.load_trusted(path,sha)
    def test_test_values_cannot_influence_fit(self):
        rows=episodes();m=ResidualDetector();first=m.fit(rows,binding={})
        # Fit consumes only the explicitly supplied development rows; evaluation
        # of another source must not change the frozen learned parameters.
        threshold=m.threshold;center=m.center
        held=copy.deepcopy(rows[-1]);held['observations'][12][6]*=3
        m.evaluate([held]);self.assertEqual(m.threshold,threshold);self.assertEqual(m.center,center)

if __name__=='__main__':unittest.main()
