"""Typed task recipes: YAML defaults followed by explicit command-line overrides."""
import argparse
import json
import math
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise ValueError(f'Duplicate recipe field: {key}')
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def recipe_defaults(parser, path, task):
    data = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(data, dict) or data.get('task') != task:
        raise ValueError(f'Recipe task must be {task}')
    actions = {action.dest: action for action in parser._actions
               if action.dest not in ('help', 'recipe', 'print_config')}
    unknown = set(data) - set(actions) - {'task'}
    if unknown:
        raise ValueError(f'Unknown recipe fields: {sorted(unknown)}')
    defaults = {}
    for key, value in data.items():
        if key == 'task':
            continue
        action = actions[key]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if type(value) is not bool:
                raise ValueError(f'{key}: expected boolean')
            defaults[key] = value
            continue
        multiple = action.nargs in ('+', '*') or isinstance(action.nargs, int)
        if multiple:
            if not isinstance(value, list) or action.nargs == '+' and not value:
                raise ValueError(f'{key}: expected nonempty list')
            if isinstance(action.nargs, int) and len(value) != action.nargs:
                raise ValueError(f'{key}: incorrect list length')
            values = value
        else:
            values = [value]
        converted = []
        for item in values:
            target = action.type or str
            if target is int and type(item) is not int:
                raise ValueError(f'{key}: expected integer')
            if target is float and (isinstance(item, bool) or not isinstance(item, (int, float))):
                raise ValueError(f'{key}: expected number')
            if target in (str, Path) and not isinstance(item, str):
                raise ValueError(f'{key}: expected string')
            item = target(item)
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f'{key}: expected finite number')
            if action.choices is not None and item not in action.choices:
                raise ValueError(f'{key}: unsupported value {item}')
            converted.append(item)
        defaults[key] = converted if multiple else converted[0]
    return defaults


def parse_training_args(parser, *, task, default_recipe, argv=None):
    parser.add_argument('--recipe', type=Path, default=default_recipe)
    parser.add_argument('--print-config', action='store_true', help='Print resolved recipe/CLI settings without training')
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--recipe', type=Path, default=default_recipe)
    selected, _ = selector.parse_known_args(argv)
    try:
        defaults = recipe_defaults(parser, selected.recipe, task)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f'Invalid training recipe: {exc}')
    parser.set_defaults(**defaults)
    for action in parser._actions:
        if action.dest in defaults:
            action.required = False
    args = parser.parse_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), default=str, sort_keys=True))
        raise SystemExit(0)
    return args
