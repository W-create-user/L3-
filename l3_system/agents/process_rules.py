"""Versioned, opt-in simulation fences. Configuration is not physical certification.

All numeric limits must come from reviewed, environment-specific evidence.
This module never loads or invents a default physical threshold.
"""
from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path

import numpy as np


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


class ProcessRuleEngine:
    """Immutable-by-copy configuration; evaluate does not execute or mutate a plant."""

    def __init__(self, config, *, allow_experimental=False):
        if type(allow_experimental) is not bool:
            raise ValueError('allow_experimental must be boolean')
        if not isinstance(config, dict):
            raise ValueError('process rule configuration must be an object')
        self._config = deepcopy(config)
        c = self._config
        expected = {'schema_version', 'enabled', 'environment', 'version',
                    'observation_shape', 'action_shape', 'variables', 'rules'}
        if set(c) != expected or c['schema_version'] != 'ProcessRules/1.0':
            raise ValueError('invalid process rule configuration fields/schema')
        if type(c['enabled']) is not bool or not all(_text(c[k]) for k in ('environment', 'version')):
            raise ValueError('enabled, environment or version is invalid')
        self.enabled = c['enabled']
        self.environment = c['environment']
        self.version = c['version']
        for key in ('observation_shape', 'action_shape'):
            shape = c[key]
            if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError('shapes must contain positive integer dimensions')
        self.observation_shape = tuple(c['observation_shape'])
        self.action_shape = tuple(c['action_shape'])
        if not isinstance(c['variables'], dict) or not isinstance(c['rules'], list):
            raise ValueError('variables and rules must be an object and list')
        for name, variable in c['variables'].items():
            if not _text(name) or not isinstance(variable, dict) or set(variable) != {'source', 'index', 'unit', 'coordinate'}:
                raise ValueError('invalid variable dictionary entry')
            if variable['source'] not in ('observation', 'action'):
                raise ValueError('variable source must be observation or action')
            shape = self.observation_shape if variable['source'] == 'observation' else self.action_shape
            if type(variable['index']) is not int or not 0 <= variable['index'] < math.prod(shape):
                raise ValueError('variable index outside declared shape')
            if not _text(variable['unit']) or not _text(variable['coordinate']):
                raise ValueError('variable unit and coordinate required')
        occupied = [(v['source'], v['index']) for v in c['variables'].values()]
        if len(set(occupied)) != len(occupied):
            raise ValueError('variable dictionary cannot alias a source index')
        ids, ranges, equalities = set(), {}, {}
        base = {'rule_id', 'kind', 'enabled', 'priority', 'basis', 'suggestion'}
        specific = {'range': {'variable', 'lower', 'upper'},
                    'rate': {'variable', 'max_rate_per_second'},
                    'precondition': {'variable', 'equals'},
                    'freshness': {'max_age_seconds'}}
        for rule in c['rules']:
            if not isinstance(rule, dict) or rule.get('kind') not in specific:
                raise ValueError('unknown process rule kind')
            if set(rule) != base | specific[rule['kind']]:
                raise ValueError('invalid process rule fields')
            if not _text(rule['rule_id']) or not rule['rule_id'].startswith('R-03-') or rule['rule_id'] in ids:
                raise ValueError('unique R-03- rule_id required')
            ids.add(rule['rule_id'])
            if type(rule['enabled']) is not bool or type(rule['priority']) is not int or rule['priority'] < 0:
                raise ValueError('invalid rule enabled/priority')
            if not _text(rule['suggestion']):
                raise ValueError('recovery suggestion required')
            basis = rule['basis']
            if (not isinstance(basis, dict) or set(basis) != {'kind', 'reference', 'reviewed_by', 'reviewed_on'}
                    or basis['kind'] not in ('model_definition', 'authorized_rulebook', 'team_experiment')
                    or any(not _text(basis[k]) for k in ('reference', 'reviewed_by', 'reviewed_on'))):
                raise ValueError('versioned reviewed rule basis required')
            try:
                date.fromisoformat(basis['reviewed_on'])
            except ValueError as exc:
                raise ValueError('basis reviewed_on must be an ISO date') from exc
            if self.enabled and rule['enabled'] and basis['kind'] == 'team_experiment' and not allow_experimental:
                raise ValueError('team experiment rules require explicit allow_experimental=True')
            kind = rule['kind']
            if kind != 'freshness':
                if rule['variable'] not in c['variables']:
                    raise ValueError('unknown rule variable')
                if kind == 'rate' and c['variables'][rule['variable']]['source'] != 'action':
                    raise ValueError('rate rule must target an action variable')
            if kind == 'range':
                if not all(_number(rule[k]) for k in ('lower', 'upper')) or rule['lower'] > rule['upper']:
                    raise ValueError('invalid finite range')
                if rule['enabled']:
                    old = ranges.get(rule['variable'], (-math.inf, math.inf))
                    limits = max(old[0], rule['lower']), min(old[1], rule['upper'])
                    if limits[0] > limits[1]:
                        raise ValueError('conflicting enabled ranges')
                    ranges[rule['variable']] = limits
            elif kind == 'precondition' and not _number(rule['equals']):
                raise ValueError('precondition equals must be finite numeric')
            elif kind in ('rate', 'freshness'):
                key = 'max_rate_per_second' if kind == 'rate' else 'max_age_seconds'
                if not _number(rule[key]) or rule[key] < 0:
                    raise ValueError('nonnegative finite rate/age required')
            if kind == 'precondition' and rule['enabled']:
                name = rule['variable']
                if name in equalities and equalities[name] != rule['equals']:
                    raise ValueError('conflicting enabled preconditions')
                equalities[name] = rule['equals']
        for name, value in equalities.items():
            if name in ranges and not ranges[name][0] <= value <= ranges[name][1]:
                raise ValueError('precondition conflicts with enabled range')
        self._rules = sorted([r for r in c['rules'] if r['enabled']], key=lambda r: (r['priority'], r['rule_id']))
        if self.enabled and not self._rules:
            raise ValueError('enabled process rules require at least one enabled rule')

    @classmethod
    def from_json(cls, path, **kwargs):
        def reject_constant(value):
            raise ValueError(f'nonfinite JSON constant: {value}')
        return cls(json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant), **kwargs)

    def _violation(self, rule_id, reason, *, rule=None):
        return {'rule_id': rule_id, 'reason': reason, 'risk_level': 3,
                'basis': deepcopy(rule['basis']) if rule else {'kind': 'data_contract', 'reference': 'ProcessRules/1.0'},
                'suggestion': rule['suggestion'] if rule else 'Restore verified context and request a fresh review',
                'rule_version': self.version, 'environment': self.environment}

    def evaluate(self, observation, action, context):
        """Return all violations in priority order. action=None performs state-only review.

        Context clocks are seconds in one producer-defined monotonic clock/episode.
        The caller must provide trusted provenance; labels alone cannot verify units.
        """
        if not self.enabled:
            return []
        invalid = lambda reason: [self._violation('R-03-CONTEXT', reason)]
        if not isinstance(context, dict) or context.get('environment') != self.environment:
            return invalid('Missing or mismatched process environment context')
        if not _text(context.get('episode_id')):
            return invalid('Missing process episode identity')
        arrays = {}
        for source, raw, shape in [('observation', observation, self.observation_shape),
                                   ('action', action, self.action_shape)]:
            if source == 'action' and action is None:
                continue
            try:
                values = np.asarray(raw)
                if values.shape != shape or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
                    return invalid(f'Invalid {source} numeric values/shape')
            except (TypeError, ValueError):
                return invalid(f'Invalid {source} data')
            arrays[source] = values.reshape(-1)
        # Only metadata needed by active rules is mandatory; a disabled rule adds no requirements.
        required = {r['variable'] for r in self._rules if r['kind'] != 'freshness'
                    and (action is not None or r['kind'] != 'precondition')
                    and (action is not None or self._config['variables'][r['variable']]['source'] != 'action')}
        for name in required:
            var = self._config['variables'][name]
            supplied = context.get('variables', {}).get(name) if isinstance(context.get('variables'), dict) else None
            if not isinstance(supplied, dict) or any(supplied.get(k) != var[k] for k in ('unit', 'coordinate')):
                return invalid(f'Missing/mismatched unit or coordinate for {name}')
        violations = []
        for rule in self._rules:
            kind = rule['kind']
            if action is None and kind == 'precondition':
                continue  # Preconditions gate a candidate, not the already advanced state.
            if kind == 'freshness':
                now, observed = context.get('reviewed_at'), context.get('observed_at')
                if not _number(now) or not _number(observed) or now < observed:
                    return invalid('Freshness requires ordered finite timestamps in seconds')
                violated = now - observed > rule['max_age_seconds']
            else:
                var = self._config['variables'][rule['variable']]
                if var['source'] not in arrays:
                    continue
                value = float(arrays[var['source']][var['index']])
                if kind == 'range':
                    violated = not rule['lower'] <= value <= rule['upper']
                elif kind == 'precondition':
                    violated = value != rule['equals']
                else:
                    if context.get('previous_action_episode_id') != context['episode_id']:
                        return invalid('Previous action must belong to the current episode')
                    previous = context.get('previous_action')
                    now, before = context.get('action_at'), context.get('previous_action_at')
                    if not _number(now) or not _number(before) or now <= before:
                        return invalid('Action rate requires positive elapsed seconds')
                    try:
                        previous = np.asarray(previous)
                        if previous.shape != self.action_shape or previous.dtype.kind not in 'fiu' or not np.isfinite(previous).all():
                            return invalid('Action rate requires last executed action in the same coordinates')
                    except (TypeError, ValueError):
                        return invalid('Invalid previous action')
                    violated = abs(value - float(previous.reshape(-1)[var['index']])) / (now - before) > rule['max_rate_per_second']
            if violated:
                violations.append(self._violation(rule['rule_id'], f'Process fence violated: {kind}', rule=rule))
        return violations
