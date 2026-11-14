#!/usr/bin/env python3
"""Play an RGB Haircut episode using the Gemini Decisions API for cut selection."""

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import haircut
from snapshot_eval import GEMINI_MODEL, HERE, post
from vision import CAMERAS, Cameras


ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/decisions'
DECISION_PROMPT = (
    'Choose the next cell and cutting height to make progress on this simulated haircut. '
    'The orbit image shows the overall head. The blade image shows the area near the clipper. '
    'The candidate directions use head coordinates, with Y up, positive Z toward the face and the head at the origin. '
    'Prefer visibly long, unfinished hair. Prefer the higher cut where the scalp is hard to see, '
    'and favor a nearby cell when options look similar. Never aim to contact the scalp.'
)


def move_toward(env, target):
    """Propose a bounded move using the public nominal scalp and the current joint angles."""
    goal = haircut.solve(target, env.joints)
    if goal is None:
        return None
    desired = list(map(float, haircut._turn(env.joints, goal).split()[1:]))
    distance = math.dist(env.tool, target)
    for fraction in (1, 0.5, 0.25):
        delta = [round(x * fraction, 4) for x in desired]
        end = [q + x for q, x in zip(env.joints, delta)]
        if math.dist(haircut.arm_points(end)[-1], target) >= distance - 0.001:
            continue
        if all(haircut._safe([q + x * i / haircut.SWEEP for q, x in zip(env.joints, delta)], haircut.nominal)
               for i in range(1, haircut.SWEEP + 1)):
            return 'move ' + ' '.join(f'{x:.4f}' for x in delta)
    return None


def cut_cell(env, cell, height):
    direction = haircut.CELLS[cell]
    radius = haircut.nominal(direction)
    approach = haircut.scale(direction, radius + height + 0.18)
    finish = haircut.scale(direction, radius + height)
    for i in range(11):
        if env.over:
            return 'budget'
        if env.cell == cell and math.dist(env.tool, finish) < 0.045:
            env.step('cut')
            return 'cut'

        current = haircut.unit(env.tool)
        near = haircut.dot(current, direction) > 0.993
        target = finish if near else approach
        action = move_toward(env, target)
        if action is None:
            outward = min(max(haircut.norm(env.tool) + 0.18, 1.35), 2.1)
            action = move_toward(env, haircut.scale(current, outward))
        if action is None:
            return 'blocked'
        env.step(action)
        if env.log[-1]['nick']:
            return 'nick'
    return 'step_limit'


def options(env, remaining):
    direction = haircut.unit(env.tool)
    ordered = sorted(remaining, key=lambda cell: (-haircut.dot(direction, haircut.CELLS[cell]), cell))
    choices = {}
    for cell in ordered[:3]:
        for name, height in [('target', max(0.09, env.target + 0.02)), ('high', max(0.12, env.target + 0.055))]:
            choices[f'cell_{cell:02d}_{name}'] = (cell, height)
    return choices


def choose(env, camera, key, choices, remaining, previous):
    state = {'telemetry': env.obs(vision=True), 'pending_cells': len(remaining), 'previous_attempt': previous}
    content = [{'type': 'text', 'text': json.dumps(state)}]
    pictures = camera.images()
    for name, label in CAMERAS:
        content.extend([{'type': 'text', 'text': label}, {'type': 'image', 'mime_type': 'image/jpeg',
                         'data': pictures[name].partition(',')[2], 'resolution': 'medium'}])
    criteria = {}
    for name, (cell, height) in choices.items():
        direction = [round(x, 3) for x in haircut.CELLS[cell]]
        distance = round(math.degrees(math.acos(max(-1, min(1, haircut.dot(haircut.unit(env.tool), direction))))))
        criteria[name] = (f'Cut cell {cell}, direction {direction}, {distance} degrees from the blade, '
                          f'at {height:.3f} above the estimated scalp.')
    reply, elapsed, retries = post(ENDPOINT, {'x-goog-api-key': key}, {
        'model': GEMINI_MODEL, 'input': content,
        'questions': {'next_cut': {'type': 'choice', 'instructions': DECISION_PROMPT, 'criteria': criteria}},
    })
    answer = reply['answers']['next_cut']
    selected = answer['choice']
    if selected not in choices:
        raise RuntimeError('Gemini returned an unknown candidate')
    return selected, {'answer': answer, 'options': criteria, 'api_seconds': round(elapsed, 4),
                      'retries': retries, 'usage': reply['usage']}


def run(seed, folder, key):
    folder.mkdir(parents=True, exist_ok=False)
    env = haircut.Haircut(seed)
    remaining = set(range(len(haircut.CELLS)))
    results, previous = [], None
    started = time.monotonic()

    def save():
        latencies = [r['api_seconds'] for r in results]
        summary = {'seed': seed, 'model': GEMINI_MODEL, 'endpoint': ENDPOINT, 'done': env.over,
                   'target': env.target, 'steps_left': env.left, 'decisions': len(results),
                   'outcomes': dict(Counter(r['outcome'] for r in results)),
                   'input_tokens': sum(r['usage']['total_input_tokens'] for r in results),
                   'output_tokens': sum(r['usage']['total_output_tokens'] for r in results),
                   'median_api_seconds': round(statistics.median(latencies), 3) if latencies else None,
                   'elapsed_seconds': round(time.monotonic() - started, 1), **env.metrics()}
        env.save_replay(folder / f'run-{seed}.html', vision=True)
        (folder / f'run-{seed}.json').write_text(json.dumps(summary, indent=2) + '\n')

    save()
    with Cameras(env) as camera, (folder / 'transcript.jsonl').open('w') as log:
        while remaining and not env.over:
            call_start = time.monotonic()
            candidates = options(env, remaining)
            selected, result = choose(env, camera, key, candidates, remaining, previous)
            cell, height = candidates[selected]
            before = len(env.log)
            outcome = cut_cell(env, cell, height)
            remaining.remove(cell)
            previous = {'cell': cell, 'outcome': outcome}
            result.update(selected=selected, cell=cell, height=height, outcome=outcome,
                          first_step=before, last_step=len(env.log) - 1)
            results.append(result)
            log.write(json.dumps(result) + '\n')
            log.flush()
            save()
            print(f'{len(results):2d}/42 cell {cell:2d} {outcome:10} step {len(env.log)-1:3d} '
                  f'nicks {env.nicks} reward {env.score():.3f} API {result["api_seconds"]:.2f}s', flush=True)
            time.sleep(max(0, 2.6 - (time.monotonic() - call_start)))
        if not env.over:
            env.step('done')
        save()
    print((folder / f'run-{seed}.json').read_text(), end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, nargs='?', default=6)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    # Use the saved AI Studio project key so an unrelated environment key cannot take precedence.
    key_file = HERE / '.gemini-key'
    if not key_file.is_file() or not (key := key_file.read_text().strip()):
        parser.error('Save the free-tier project key in .gemini-key first')
    folder = args.out or HERE / 'runs' / f'seed-{args.seed}' / 'gemini-decisions'
    run(args.seed, folder, key)


if __name__ == '__main__':
    main()
