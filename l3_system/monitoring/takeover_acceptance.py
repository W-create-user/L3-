"""D06 reproducible HTTP and session acceptance, with simulated operator actions.

Uses existing A/B/C implementations unchanged. Fixtures isolate failure paths;
--real additionally uses actual SMPL Atropine. No human-factor claims are made.
"""
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.ui.app import create_app


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_case(output, name, *, real=False, terminal=False, probe=None, seed=1101):
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=False)
    events = []
    def stamp(event, **fields):
        row = {'event': event, 'timestamp': datetime.now(timezone.utc).isoformat(),
               'monotonic_ns': time.perf_counter_ns(), **fields}
        events.append(row)
        return row
    with ExitStack() as stack, (folder/'stdout.txt').open('w') as log, redirect_stdout(log):
        if real:
            from l3_system.execution import EnvironmentAdapter
            adapter = stack.enter_context(EnvironmentAdapter(max_steps=1 if terminal else 10))
        else:
            from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
            adapter = FakeAdapter()
        executor = ExecutionAgent(adapter, log_path=folder/'B_events.jsonl')
        executor.reset(seed=seed)
        initial_hash = hashlib.sha256(json.dumps(executor.get_observation(), allow_nan=False).encode()).hexdigest()
        monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
        system = MultiAgentSystem(executor=executor, monitor=monitor, event_path=folder/'session_events.jsonl')
        app = create_app(system, environment='AtropineEnvGym' if real else 'CONTRACT_FIXTURE')
        client = app.test_client()
        def form():
            page = client.get('/')
            require(page.status_code == 200, 'dashboard unavailable')
            html = page.get_data(as_text=True)
            token = re.search(r'name="token" value="([^"]+)"', html).group(1)
            revision = re.search(r'name="revision" value="([^"]+)"', html).group(1)
            return {'token': token, 'revision': revision, 'operator': 'D06-simulated-operator',
                    'reason': 'D06 reproducible session review'}
        def post(route, data=None):
            before = adapter.step_calls
            response = client.post(route, data=form() if data is None else data)
            stamp('http_response', route=route, status=response.status_code,
                  step_delta=adapter.step_calls-before, state=system.state)
            return response
        before = adapter.step_calls
        if probe == 'normal':
            require(post('/cycle').status_code == 303, 'normal cycle failed')
            require(adapter.step_calls == before+1, 'normal cycle must step once')
        else:
            stamp('risk_probe_started', probe=probe, terminal=terminal)
            if terminal:
                # Upstream checks its pre-increment step counter for timeout.
                # Observe the actual terminal flag rather than assuming one step.
                for _ in range(3):
                    system.run_cycle()
                    if adapter.done:
                        break
                require(adapter.done, 'terminal scenario did not reach episode end')
            else:
                system.run_cycle(action_generator=lambda o, s, a: s.high+1)
                require(adapter.step_calls == before, 'rejected candidate was executed')
            if terminal:
                # Next operation on an ended episode must latch the existing session.
                system.run_cycle()
            require(system.state == 'takeover_requested', 'takeover not requested')
            request_event = stamp('takeover_observed')
            count = adapter.step_calls
            try:
                system.run_cycle()
            except RuntimeError:
                pass
            else:
                raise AssertionError('unconfirmed cycle was allowed')
            require(adapter.step_calls == count, 'unconfirmed cycle stepped')
            require(post('/resume', {}).status_code == 403, 'missing token accepted')
            if probe == 'missing_confirmation':
                invalid = form(); invalid['operator'] = ''
                require(post('/resume', invalid).status_code == 400, 'empty operator accepted')
                require(system.state == 'takeover_requested', 'invalid confirmation resumed')
            elif probe == 'lock':
                system.scheduler.active_agent_tasks['D06-fixture'] = 'unresolved'
                require(post('/resume').status_code == 409, 'unreleased resource accepted')
                system.scheduler.active_agent_tasks.clear()
            elif probe == 'pending':
                pending = executor.prepare_action({'task_id':'D06-pending','task_type':'control_cycle'},
                                                  action=adapter.action_space.low.tolist())
                require(post('/resume').status_code == 409, 'pending request accepted')
                executor.cancel_action_request(pending['request_id'], 'D06 fixture cleanup')
            elif probe == 'missing_observation':
                adapter._obs[0] = float('nan')
                require(post('/resume').status_code == 409, 'NaN observation accepted')
                adapter._obs[0] = 0.
            elif probe == 'stale_request':
                pending = executor.prepare_action({'task_id':'D06-stale','task_type':'control_cycle'},
                                                  action=adapter.action_space.low.tolist())
                decision = monitor.check_request(pending, executor.get_observation())
                adapter.state_version += 1
                try:
                    executor.execute_action(pending, decision)
                except ValueError:
                    pass
                else:
                    raise AssertionError('stale approval accepted')
                executor.cancel_action_request(pending['request_id'], 'D06 stale fixture cleanup')
            elif probe == 'unknown_result':
                adapter._ready = False
                require(post('/resume').status_code == 409, 'unknown/unready context resumed')
                require(system.state == 'takeover_requested', 'unready context lost latch')
            if terminal:
                require(post('/resume').status_code == 409, 'ended episode resumed')
            elif probe != 'unknown_result':
                data = form()
                confirmed = stamp('simulated_confirmation_submitted')
                require(post('/resume', data).status_code == 303, 'legal confirmation failed')
                restored = stamp('resume_observed')
                require(adapter.step_calls == count, 'confirmation executed an action')
                require(post('/resume', data).status_code == 409, 'duplicate form accepted')
                require(post('/resume').status_code == 409, 'duplicate confirmation accepted')
                # Same rejected policy after confirmation must still be denied by D.
                system.run_cycle(action_generator=lambda o, s, a: s.high+1)
                require(system.state == 'takeover_requested', 'new action bypassed D gate')
                require(adapter.step_calls == count, 'new rejected action stepped')
                require(post('/resume').status_code == 303, 'second review failed')
                require(post('/cycle').status_code == 303, 'reviewed normal cycle failed')
                require(adapter.step_calls == count+1, 'resumed normal cycle must step once')
                stamp('recovery_timing',
                      request_to_simulated_confirmation_s=(confirmed['monotonic_ns']-request_event['monotonic_ns'])/1e9,
                      confirmation_to_resume_s=(restored['monotonic_ns']-confirmed['monotonic_ns'])/1e9,
                      human_thinking_time_s=None)
        require(not system.scheduler.active_agent_tasks and not executor._pending, 'resources not clear')
        monitor.export_logs(folder/'D_reviews.jsonl')
        transitions = system.transitions
        report = {'case':name, 'passed':True, 'backend':'real SMPL Atropine' if real else 'contract fixture',
                  'seed':seed, 'initial_observation_sha256':initial_hash,
                  'total_step_calls':adapter.step_calls, 'state':system.state,
                  'operator':'simulated HTTP test client', 'browser_render_tested':False,
                  'human_factor_study':False, 'process_safety_certified':False,
                  'events':events, 'transitions':transitions}
        (folder/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return report


def run(output, *, real=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    cases = [run_case(output, 'fixture_'+p, probe=p) for p in
             ('normal','missing_confirmation','lock','pending','missing_observation','stale_request','unknown_result')]
    if real:
        cases += [run_case(output, 'real_normal', real=True, probe='normal'),
                  run_case(output, 'real_recoverable', real=True, probe='missing_confirmation'),
                  run_case(output, 'real_terminal', real=True, terminal=True)]
    root = Path(__file__).resolve().parents[2]
    files = ['runtime.py','ui/app.py','ui/templates/dashboard.html','execution/execution_agent.py',
             'agents/monitoring_agent.py','agents/scheduling_agent.py','monitoring/takeover_acceptance.py']
    report = {'schema_version':'D06Acceptance/1.0', 'passed':all(r['passed'] for r in cases),
              'cases':cases, 'code_sha256':{n:hashlib.sha256((root/'l3_system'/n).read_bytes()).hexdigest() for n in files},
              'scope':'local simulation HTTP/session acceptance; no browser rendering or human participant study',
              'D05_risk_policy_integrated':False, 'new_physical_thresholds':False}
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    result = run(args.output, real=args.real)
    print(json.dumps({'passed':result['passed'], 'cases':len(result['cases'])}))
