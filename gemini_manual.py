#!/usr/bin/env python3
"""Run Gemini Decisions directly on the RGB Haircut controls for one episode."""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import haircut
from gemini_episode import ENDPOINT
from snapshot_eval import GEMINI_MODEL, HERE, post
from vision import CAMERAS, Cameras


# A fixed menu of manual controls. A choice changes one joint or one orbit angle,
# or presses cut. All controls are offered in every pose.
CONTROLS = {}
for joint, name in enumerate(('yaw', 'shoulder', 'elbow', 'wrist')):
    for size, radians in (('fine', 0.025), ('coarse', 0.2)):
        for sign, multiplier in (('minus', -1), ('plus', 1)):
            delta = radians * multiplier
            CONTROLS[f'{name}_{sign}_{size}'] = (f'Turn {name} by {delta:+g} radians.',
                                                       ('move', joint, delta))
for axis, name in enumerate(('azimuth', 'elevation')):
    for sign, delta in (('minus', -0.3), ('plus', 0.3)):
        CONTROLS[f'orbit_{name}_{sign}'] = (f'Turn orbit camera {name} by {delta:+g} radians.',
                                                    ('look', axis, delta))
CONTROLS['cut'] = ('Cut at the current blade position.', ('cut', None, None))


def command_for(choice, camera_angles):
    action, axis, delta = CONTROLS[choice][1]
    if action == 'cut':
        return 'cut'
    values = [0.0] * 4 if action == 'move' else list(camera_angles)
    values[axis] += delta
    return f'{action} ' + ' '.join(f'{value:.4f}' for value in values)


def run(seed, folder, key):
    env = haircut.Haircut(seed)
    question = {
        'type': 'choice',
        'instructions': (
            f'Trim the hair to a length of {env.target:g} units without nicking the scalp. Choose one manual control. '
            'Blade XYZ is in head coordinates. Y is up and positive Z points out of the face. '
            'The orbit camera moves around the head. The blade camera follows the arm and looks at the blade tip. '
            'Yaw turns the arm around the vertical axis. The shoulder, elbow and wrist bend the links in that plane. '
            'Cut trims the hair under the blade to its current height above the scalp.'
        ),
        'criteria': {name: description for name, (description, _) in CONTROLS.items()},
    }
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'question.json').write_text(json.dumps(question, indent=2) + '\n')
    started = time.monotonic()
    turns = []

    def save():
        times = [turn['api_seconds'] for turn in turns]
        summary = {
            'seed': seed, 'model': GEMINI_MODEL, 'endpoint': ENDPOINT, 'done': env.over,
            'target': env.target, 'steps_left': env.left, 'decisions': len(turns),
            'actions': dict(Counter(turn['action'].split()[0] for turn in turns)),
            'cut_cells': sum(a != b for a, b in zip(env.log[0]['len'], env.log[-1]['len'])),
            'input_tokens': sum(turn['usage']['total_input_tokens'] for turn in turns),
            'output_tokens': sum(turn['usage']['total_output_tokens'] for turn in turns),
            'median_api_seconds': round(statistics.median(times), 3) if times else None,
            'elapsed_seconds': round(time.monotonic() - started, 1), **env.metrics(),
        }
        (folder / f'run-{seed}.json').write_text(json.dumps(summary, indent=2) + '\n')
        env.save_replay(folder / f'run-{seed}.html', vision=True)

    save()
    with Cameras(env) as cameras, (folder / 'transcript.jsonl').open('w') as log:
        while not env.over:
            call_start = time.monotonic()
            blade_xyz = list(env.tool)
            content = [{'type': 'text', 'text': json.dumps({'blade_xyz': blade_xyz})}]
            pictures = cameras.images()
            for name, label in CAMERAS:
                content.extend([{'type': 'text', 'text': label}, {'type': 'image', 'mime_type': 'image/jpeg',
                                 'data': pictures[name].partition(',')[2], 'resolution': 'medium'}])
            reply, elapsed, retries = post(ENDPOINT, {'x-goog-api-key': key}, {
                'model': GEMINI_MODEL, 'input': content, 'questions': {'control': question},
            })
            answer = reply['answers']['control']
            choice = answer['choice']
            if choice not in CONTROLS:
                raise RuntimeError('Gemini returned an unknown control')
            action = command_for(choice, env.camera)
            env.step(action)
            turn = {'step': len(env.log) - 1, 'blade_xyz': blade_xyz, 'answer': answer, 'action': action,
                    'api_seconds': round(elapsed, 4), 'retries': retries, 'usage': reply['usage']}
            turns.append(turn)
            log.write(json.dumps(turn) + '\n')
            log.flush()
            save()
            if turn['step'] % 12 == 0 or turn['step'] == 1 or env.over:
                print(f'Step {turn["step"]:3d} | {choice:22} | nicks {env.nicks:2d} | reward {env.score():.3f}', flush=True)
            if not env.over:
                # The free project allows 50K input tokens per minute. Pace from
                # the measured payload and leave headroom for temporary variation.
                interval = max(2.6, 60 * turn['usage']['total_input_tokens'] / 40000)
                time.sleep(max(0, interval - (time.monotonic() - call_start)))
    print((folder / f'run-{seed}.json').read_text(), end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, nargs='?', default=6)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    key_file = HERE / '.gemini-key'
    if not key_file.is_file() or not (key := key_file.read_text().strip()):
        parser.error('Save the free-tier project key in .gemini-key first')
    folder = args.out or HERE / 'runs' / f'seed-{args.seed}' / 'gemini-manual'
    run(args.seed, folder, key)


if __name__ == '__main__':
    main()
