import unittest
from types import SimpleNamespace
import numpy as np
from experiments.verify_mab_daily import validate_step


class MabStepEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(observation_space=SimpleNamespace(shape=(2,)),error_reward=-100.,step_count=1)

    def test_solver_error_reward_is_rejected_even_when_counter_advances(self):
        with self.assertRaisesRegex(RuntimeError,'error reward'):
            validate_step(self.env,np.zeros(2),-100.,0)

    def test_invalid_observation_does_not_count_as_success(self):
        for obs in (np.zeros(3),np.array([0.,float('nan')])):
            with self.subTest(obs=obs),self.assertRaises(RuntimeError):
                validate_step(self.env,obs,0.,0)

    def test_nonfinite_reward_and_missing_advance_are_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_step(self.env,np.zeros(2),float('nan'),0)
        self.env.step_count=0
        with self.assertRaisesRegex(RuntimeError,'advance'):
            validate_step(self.env,np.zeros(2),0.,0)

    def test_finite_successful_result_is_accepted(self):
        validate_step(self.env,np.zeros(2),0.,0)
