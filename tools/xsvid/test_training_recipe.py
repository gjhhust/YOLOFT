import argparse
import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from tools.xsvid.training_recipe import parse_training_args, recipe_defaults

ROOT = Path(__file__).resolve().parents[2]
TASKS = {'vid': ('tools/xsvid/train_vid.py', 'vid.yaml'),
         'mot': ('train_omni_embed.py', 'unified_mot.yaml'),
         'sot': ('train_sot_transt.py', 'unified_sot.yaml')}
EXPECTED = {
    'vid': {'epochs': 15, 'batch': 24, 'imgsz': 1024, 'fraction': 1.0, 'device': '0'},
    'mot': {'epochs': 8, 'samples': 6000, 'bv': 8, 'lr': .001, 'intervals': [1, 2, 3],
            'emb_stride': 4, 'embed_dim': 256, 'pre_mstf': 0, 'emb_src_hi': 20,
            'feat_mode': 'simple', 'tau': .07, 'seed': 0, 'tag': 'unified_mot'},
    'sot': {'epochs': 40, 'samples': 12000, 'bv': 24, 'lr': .0001, 'lr_bb': 0.,
            'unfreeze_bb': 1, 'feat_mode': 'fuse918', 'z': 128, 'x': 256,
            'no_pretrain': False, 'tag': 'unified_sot'},
}


def entry_parser(task):
    tree = ast.parse((ROOT / TASKS[task][0]).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    nodes = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('ap', 'parser') for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            if isinstance(function, ast.Attribute) and function.attr == 'add_argument':
                nodes.append(node)
    environment = {'argparse': argparse, 'Path': Path, 'ROOT': ROOT, 'Z': 128, 'X': 256}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-entry-parser>', 'exec'), environment)
    return environment.get('parser', environment.get('ap'))


class TrainingRecipeTests(unittest.TestCase):
    def defaults(self, task):
        return ROOT / 'config/recipes' / TASKS[task][1]

    def inputs(self, task):
        if task == 'vid':
            return ['--data', 'data.yaml', '--weights', 'vid.pt', '--project', 'out']
        return ['--data-root', 'data', '--det-ckpt', 'vid.pt', '--out-dir', 'out']

    def test_defaults_equal_mature_commands(self):
        for task, expected in EXPECTED.items():
            with self.subTest(task=task):
                args = parse_training_args(entry_parser(task), task=task,
                    default_recipe=self.defaults(task), argv=self.inputs(task))
                for field, value in expected.items():
                    self.assertEqual(getattr(args, field), value, field)

    def test_cli_overrides_each_recipe(self):
        for task in TASKS:
            with self.subTest(task=task):
                args = parse_training_args(entry_parser(task), task=task,
                    default_recipe=self.defaults(task), argv=self.inputs(task) + ['--epochs', '1'])
                self.assertEqual(args.epochs, 1)

    def test_list_override(self):
        args = parse_training_args(entry_parser('mot'), task='mot', default_recipe=self.defaults('mot'),
                                   argv=self.inputs('mot') + ['--intervals', '2', '3'])
        self.assertEqual(args.intervals, [2, 3])

    def test_unknown_field_fails_closed(self):
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                recipe = Path(directory) / 'bad.yaml'
                recipe.write_text(f'task: {task}\nunknown_switch: 1\n')
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_training_args(entry_parser(task), task=task, default_recipe=recipe, argv=self.inputs(task))
                self.assertEqual(error.exception.code, 2)

    def test_wrong_task_duplicate_wrong_type_and_choice(self):
        invalid = ['task: sot\n', 'task: mot\nepochs: 1\nepochs: 2\n',
                   'task: mot\nepochs: true\n', 'task: mot\nintervals: 1\n',
                   'task: mot\nfeat_mode: unknown\n', 'task: mot\nlr: .nan\n']
        for text in invalid:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                recipe = Path(directory) / 'bad.yaml'
                recipe.write_text(text)
                with self.assertRaises(ValueError):
                    recipe_defaults(entry_parser('mot'), recipe, 'mot')

    def test_boolean_override(self):
        args = parse_training_args(entry_parser('sot'), task='sot', default_recipe=self.defaults('sot'),
                                   argv=self.inputs('sot') + ['--no-pretrain'])
        self.assertTrue(args.no_pretrain)


if __name__ == '__main__':
    unittest.main()
