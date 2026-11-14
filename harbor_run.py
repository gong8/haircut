#!/usr/bin/env python3
"""Stage only the simulator files and run the local Harbor/OpenCode task."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
FREE_MODEL = 'gemini-3.1-flash-lite'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=FREE_MODEL)
    parser.add_argument('--key-file', type=Path, default=HERE / '.gemini-key')
    parser.add_argument('--steps', type=int, default=280, help='maximum OpenCode model turns')
    args = parser.parse_args()
    if args.model != FREE_MODEL and args.key_file == HERE / '.gemini-key':
        parser.error('Provide the funded project’s key file explicitly when changing models')
    if not args.key_file.is_file() or not (key := args.key_file.read_text().strip()):
        parser.error('Google API key file is missing or empty')

    config = {
        'permission': {'*': 'deny', 'haircut_*': 'allow'},
        'agent': {'build': {'steps': args.steps}},
        'provider': {'google': {'options': {'apiKey': '{env:GOOGLE_GENERATIVE_AI_API_KEY}'},
                                'models': {args.model: {'name': args.model, 'attachment': True,
                                                       'modalities': {'input': ['text', 'image'], 'output': ['text']}}}}},
    }
    with tempfile.TemporaryDirectory(prefix='haircut-harbor-') as tmp:
        task = Path(tmp) / 'haircut'
        shutil.copytree(HERE / 'harbor-task', task)
        simulator = task / 'environment' / 'simulator'
        simulator.mkdir()
        for name in ('haircut.py', 'vision.py', 'viewer.html', 'render-cameras.mjs', 'package.json', 'package-lock.json'):
            shutil.copy2(HERE / name, simulator / name)
        shutil.copytree(HERE / 'assets', simulator / 'assets')
        env = {**os.environ, 'GOOGLE_GENERATIVE_AI_API_KEY': key, 'GEMINI_API_KEY': key}
        subprocess.run(['uvx', '-q', '--from', 'harbor==0.26.0', 'harbor', 'run',
                        '-p', str(task), '-a', 'opencode', '-m', f'google/{args.model}',
                        '--ak', 'version=1.17.8', '--ak', 'opencode_config=' + json.dumps(config),
                        '--api-key-name', 'GOOGLE_GENERATIVE_AI_API_KEY',
                        '-o', str(HERE / 'runs' / 'harbor'), '-n', '1'], env=env, check=True)


if __name__ == '__main__':
    main()
