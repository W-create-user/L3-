"""Synthetic ratio/ceiling fixtures; no physical threshold approval."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem


def config():
    return {'schema_version':'ProcessRules/1.0','enabled':True,'environment':'SYNTHETIC_RATIO_FIXTURE','version':'fixture-1',
        'observation_shape':[39],'action_shape':[4],
        'variables':{key:{'source':'observation','index':index,'unit':'fixture-ratio','coordinate':'absolute'}
                     for key,index in [('current',6),('reference',5)]},
        'rules':[{'rule_id':'R-03-RELATIVE-FIXTURE','kind':'relative_upper','enabled':True,'priority':1,
                  'variable':'current','reference_variable':'reference','relative_margin':.25,'observable_upper':100.,
                  'basis':{'kind':'team_experiment','reference':'synthetic test only','reviewed_by':'fixture-not-human-approval','reviewed_on':'2026-11-16'},
                  'suggestion':'restore trusted reference and obtain a fresh review'}]}


def context():
    return {'environment':'SYNTHETIC_RATIO_FIXTURE','episode_id':'fixture',
            'variables':{key:{'unit':'fixture-ratio','coordinate':'absolute'} for key in ('current','reference')}}

class RelativeRuleTests(unittest.TestCase):
    def setUp(self):
        self.c=config();self.ctx=context();self.obs=[0.]*39;self.obs[5]=8.;self.obs[6]=10.
    def evaluate(self):return ProcessRuleEngine(self.c,allow_experimental=True).evaluate(self.obs,[0.]*4,self.ctx)
    def test_equality_and_breach(self):
        self.assertEqual(self.evaluate(),[]);self.obs[6]=10.01
        result=self.evaluate()[0];self.assertEqual(result['rule_id'],'R-03-RELATIVE-FIXTURE')
        self.assertEqual(result['risk_level'],3);self.assertTrue(result['basis']);self.assertTrue(result['suggestion'])
    def test_no_implicit_experimental_enable(self):
        with self.assertRaises(ValueError):ProcessRuleEngine(self.c)
        self.c['enabled']=False
        self.assertEqual(ProcessRuleEngine(self.c).evaluate(None,None,None),[])
    def test_reference_guard_and_observability(self):
        for reference in (0.,-1.,80.,1e308):
            self.obs[5]=reference
            self.assertTrue(self.evaluate())
    def test_reference_metadata_is_required(self):
        del self.ctx['variables']['reference'];self.assertTrue(self.evaluate())
        self.ctx=context();self.ctx['variables']['reference']['coordinate']='delta';self.assertTrue(self.evaluate())
    def test_configuration_units_reference_and_numbers(self):
        for field,value in (('reference_variable','absent'),('reference_variable','current'),('relative_margin',True),('relative_margin',10**1000),('observable_upper',float('nan'))):
            c=config();c['rules'][0][field]=value
            with self.assertRaises(ValueError):ProcessRuleEngine(c,allow_experimental=True)
        self.c['variables']['reference']['unit']='other'
        with self.assertRaises(ValueError):ProcessRuleEngine(self.c,allow_experimental=True)
    def test_state_review_applies_and_hard_rejection_wins(self):
        adapter=FakeAdapter();self.obs[6]=11.;adapter._obs[:]=self.obs
        mon=MonitoringAgent(adapter.action_space,adapter.observation_space,
            process_rules=ProcessRuleEngine(self.c,allow_experimental=True),process_context_provider=lambda state,action:self.ctx)
        self.assertFalse(mon.check_observation(self.obs)['approved'])
        self.assertEqual(mon.check_action(self.obs,[2.]*4)['rule_id'],'R-02')
    def test_full_chain_breach_never_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=FakeAdapter();adapter._obs[:]=self.obs;adapter._obs[6]=11.
            mon=MonitoringAgent(adapter.action_space,adapter.observation_space,
                process_rules=ProcessRuleEngine(self.c,allow_experimental=True),process_context_provider=lambda state,action:self.ctx)
            system=MultiAgentSystem(executor=ExecutionAgent(adapter,Path(tmp)/'B.jsonl'),monitor=mon)
            system.run_cycle()
            self.assertEqual(adapter.step_calls,0);self.assertEqual(system.state,'takeover_requested')
            with self.assertRaises(RuntimeError):system.resume(operator='fixture',reason='ack alone')
    def test_configuration_snapshot_isolated(self):
        engine=ProcessRuleEngine(self.c,allow_experimental=True);self.c['rules'][0]['relative_margin']=100
        self.obs[6]=11.;self.assertTrue(engine.evaluate(self.obs,None,self.ctx))

if __name__=='__main__':unittest.main()
