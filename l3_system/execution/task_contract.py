"""Accept A's original Task or its three-field B transport representation."""
import copy


META_FIELDS = ('priority', 'preconditions', 'dependencies', 'constraints',
               'subtasks', 'completion_criteria', 'status', 'source')


def normalize_task(task):
    if not isinstance(task, dict):
        raise TypeError('Task must be a dictionary')
    task_id = task.get('task_id')
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError('Task.task_id must be a nonempty string; it is preserved unchanged')
    parameters = task.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('Task.parameters must be a dictionary')
    parameters = copy.deepcopy(parameters)
    task_types = [value for value in (task.get('task_type'), task.get('goal_type'),
                                     parameters.get('task_type')) if value is not None]
    if task_types and any(value != task_types[0] for value in task_types):
        raise ValueError('Conflicting task_type/goal_type values')
    if task_types:
        if not isinstance(task_types[0], str) or not task_types[0].strip():
            raise ValueError('task_type must be a nonempty string')
        parameters['task_type'] = task_types[0]
    for key in META_FIELDS:
        if key in task:
            if key in parameters and parameters[key] != task[key]:
                raise ValueError(f'Conflicting top-level and parameters.{key}')
            parameters[key] = copy.deepcopy(task[key])
    for key in ('preconditions', 'dependencies', 'completion_criteria'):
        if key in parameters and (not isinstance(parameters[key], list)
                                  or not all(isinstance(item, str) and item for item in parameters[key])):
            raise ValueError(f'{key} must be a list of nonempty strings')
    if 'constraints' in parameters and not isinstance(parameters['constraints'], dict):
        raise ValueError('constraints must be a dictionary')
    if 'subtasks' in parameters:
        subtasks = parameters['subtasks']
        if not isinstance(subtasks, list) or not all(isinstance(item, dict) for item in subtasks):
            raise ValueError('subtasks must be a list of dictionaries')
        ids = [item.get('subtask_id') for item in subtasks]
        if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError('subtasks require unique nonempty subtask_id values')
    task_name = task.get('task_name', task.get('description'))
    if task_name is not None and (not isinstance(task_name, str) or not task_name.strip()):
        raise ValueError('task_name/description must be a nonempty string when supplied')
    return {'task_id': task_id, 'task_name': task_name, 'parameters': parameters}


def as_plan(task):
    """Build a private plan view; never move/remove fields in A's original dict."""
    normalized = normalize_task(task)
    result = copy.deepcopy(normalized['parameters'])
    result.update(task_id=normalized['task_id'], task_name=normalized['task_name'])
    result['goal_type'] = result.get('task_type')
    result.setdefault('subtasks', [])
    return result
