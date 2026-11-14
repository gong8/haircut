# Haircut

A minimal RL environment: drive a robot arm around a head, cut the hair to a target length, never touch the skin.
The agent turns the arm's joints and aims an orbit camera. A second camera follows the blade. Both send RGB images.
The simulator runs in `haircut.py`. `viewer.html` draws the scene in three.js for both the browser and the agent.

```
python3 haircut.py               # evaluate the reference and the baselines (30 seeds, under a minute)
python3 haircut.py mcp [SEED]    # agent interface: an MCP server on stdin/stdout
python3 haircut.py play [SEED]   # plain JSON with legacy depth readings, for scripts and debugging
python3 haircut.py serve [SEED]  # drive the arm yourself in the browser
python3 haircut.py demo [SEED]   # run the reference and open a 3D replay in the browser
```

## Rules

**Head.** Centred on the origin, about 1 unit in radius; y is up and z points out of the face. Its skin radius along
each direction is hidden. Every head is roughly the same ellipsoid with a jaw and a face (`nominal(d)` in
`haircut.py`), but each has its own proportions (semi-axes off by up to 0.04) and a few smooth bumps of up to 0.05.
Hair grows in 42 cells, given as fixed unit directions.

**Arm.** Four joints. The shoulder is fixed at (0, 2.8, -1.6), above and behind the head. Joint 1 (yaw) turns the arm
about the vertical line through the shoulder. Joints 2 to 4 (shoulder, elbow, wrist) bend three links of length 1.8,
1.8 and 0.6 in the vertical plane that yaw selects. The blade is the tip of the last link. The exact kinematics are in
the text `play` prints and in `arm_points`.

**Cameras.** The orbit camera looks at the whole head and can move around it. The blade camera follows the wrist
and shifts outward as needed to clear the head while keeping the blade tip centered.

**Actions.** One per step, 252 steps per episode (6 per cell):

- `move D1 D2 D3 D4`: turn each joint by up to 0.3 rad. Joints land within about 2% of the commanded turn. The whole
  sweep is checked: if the blade goes under the skin, or the arm (from the shoulder to 0.3 short of the blade) comes
  within 0.06 of it, that step is a nick.
- `look AZ EL`: set the orbit camera's absolute azimuth and elevation in radians. Azimuth 0 is in front of the
  face. Elevation is kept between -0.2 and 1.5 rad.
- `cut`: trim the cell under the blade to the blade's height above the skin. Irreversible.
- `done`: end the episode.

The blade is over the cell nearest in direction, if that is within 0.25 rad. Otherwise it is over bare skin and `cut`
does nothing.

**Observation.** Each step gives two 512 × 384 RGB images, plus the joint angles, blade position, orbit angles,
current cell, target hair length, remaining steps and nicks. In legacy depth mode, `seen` gives the tip radius of
visible hair cells and `gap` is a noisy estimate of the blade's height above the scalp.

**Reward.** Given once, at the end: `(sum of cell scores - 3 * nicks) / 42`. With `e = length - target`, a cell scores

- `max(0, 1 - e / 0.1)` if it is too long, which a later cut can still fix;
- `max(-1, 1 - 3 * |e| / 0.1)` if it is too short, which nothing can fix.

The target length is 0.0, 0.1 or 0.2, fixed per episode and shown in the observation. Untouched hair scores 0. A bad
action returns an error, costs no step and gives no new reading. The reward is deliberately not paid step by step:
the change in score after a cut would tell the agent how far the blade was from the hidden skin.

## Connecting an agent

```
npm ci
npx playwright install chromium
python3 haircut.py mcp SEED
```

This is a standard-input/output MCP server (newline-delimited JSON-RPC). Use a client and model that accept MCP image
content. The server renders RGB locally with Playwright and Chromium. It exposes three tools:

- `get_spec`: the rules, the hair cell directions, and the arm's geometry and starting joints.
- `observe`: the current telemetry and two labeled JPEG images, without advancing the episode.
- `step`: one action, as text (`move D1 D2 D3 D4`, `look AZ EL`, `cut`, `done`). Returns the next observation and,
  once the episode is over, the reward.

One server process is one fixed, seeded episode. There is no reset tool, so an agent cannot retry an unlucky episode.
After every step the server rewrites `run-SEED.json` (reward, nicks, steps, length errors) and `run-SEED.html` (a 3D
replay) in its working directory, so a run that just stops, or is killed, is still scored on its final state.
Use `python3 haircut.py mcp SEED --depth` for the original text-only benchmark.

## Seeing it

`serve` opens a live page: the browser sends actions and the Python process owns the environment, so a human plays by
the agent's rules and budget. The two insets show the orbit and blade cameras.

| key | action |
|---|---|
| arrows | jog the blade along the scalp (relative to your view) |
| W / S | lift / lower the blade |
| click the head | go there at the current height |
| 1-8 | turn one joint: 1/2 yaw, 3/4 shoulder, 5/6 elbow, 7/8 wrist |
| J / L, I / K | aim the orbit camera |
| shift | fine steps |
| space | cut |
| enter | done |
| N | new head |
| V | cycle through the spectator, orbit, and blade views |
| C | colour hair by score (green on target, red too short) |
| T | show the target length as a shell above the scalp |
| drag | orbit your own view |

The jog keys and click-to-go are a teach-pendant convenience: the server works out which joints to turn (`solve`),
and what is sent to the environment is still an ordinary `move`. An agent on `play` gets no such help.

`demo` and `play` write `replay.html`, a self-contained replay with a step slider.

The viewer preserves the scan's face and temples, fits the crown and back to the simulated head, and draws about
18,100 fibers from about 600 guides.
Long hair falls towards the shoulders. The guides bend under the moving blade, and clippings carry their momentum
onto the cape or floor. The viewer maps each cell's measured length to a visual length suited to the groom.
The page loads three.js 0.187.0 from jsDelivr.

The portrait adapts *Infinite, 3D Head Scan* by Lee Perry-Smith / Infinite-Realities. The geometry and color map
come from the [three.js example](https://github.com/mrdoob/three.js/tree/r187/examples/models/gltf/LeePerrySmith)
under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The bundled GLB combines both and flips the color
map to match glTF texture coordinates. The GLB also stores a radial profile sampled from the aligned scan so the
hair, target shell, and clippings can follow the visible scalp and face.

## What is hard

- **How close to go.** Going closer gives a better reading and a closer cut, but one nick costs three perfect cells.
- **Driving an arm, not a blade.** Actions are joint turns. Getting the blade somewhere means solving the arm's
  kinematics, and the links must stay clear of the head on the way, including during the sweep of each move.
- **Seeing.** Use the orbit image to see the arm and plan a move and the blade image to inspect local clearance. Aiming the camera
  costs a step, so where to look and when is part of the plan.
- **The budget.** 6 steps per cell is too few to settle slowly everywhere, so what was learned about the skin in one
  cell has to carry over to the next.

## Results

`python3 haircut.py`, seeds 0 to 29. It also writes `results.json` with every episode; the file has no timestamps, so
a rerun should reproduce it exactly.

| policy | reward | sd | nicks per episode | mean length error | cells on target (of 42) | cells too short |
|---|---|---|---|---|---|---|
| do nothing | 0.000 | 0.000 | 0.00 | 0.502 | 0.0 | 0.0 |
| stop early (reference, 1/4 of the budget) | 0.167 | 0.047 | 0.07 | 0.371 | 6.7 | 0.2 |
| naive (reference with no safety margin) | -0.081 | 0.976 | 10.67 | 0.037 | 29.6 | 1.4 |
| timid (reference with a 6 sd margin) | 0.225 | 0.242 | 0.83 | 0.123 | 8.2 | 1.1 |
| reference | 0.571 | 0.097 | 0.17 | 0.059 | 21.8 | 0.5 |

"On target" means within 0.03 of the target length; "too short" means more than 0.03 under it. Read the reward next to
these: the naive policy gets the most cells on target and still scores below doing nothing, because it nicks the skin
about eleven times an episode.

The reference uses the original depth observations, so its results are a depth-assisted baseline for RGB runs.
It goes cell to cell and learns the head on the way: its model of the skin is
the average head plus the error measured at the nearest cell already visited. It solves the arm's kinematics for a
spot just above the next cell, checks that the way there is clear, aims the camera to see the blade in profile, fuses
the gap readings, sinks to 2 sd above its estimate (or to just above the target, if that is higher) and cuts.

## Running models on it

```
python3 experiment.py [SEED] [MODEL ...]                  # Claude models through the MCP server (needs the claude CLI)
python3 nemotron.py [SEED] [--via nvidia|verda] [--assist] [--depth]
```

Both play one seeded episode with only the environment's tools and save everything under `runs/seed-SEED/NAME/`:
`run-SEED.json` (score), `run-SEED.html` (3D replay) and `transcript.jsonl`. Each prints a table next to the reference.

`nemotron.py` defaults to Nemotron 3.5 Super VL on NVIDIA Build; `--via verda` uses Verda's Inference API. The key is
read from `NVIDIA_API_KEY` or `VERDA_INFERENCE_API_KEY`, or from a file `.nvidia-key` or `.verda-key` next to the
script. `--assist` adds two helper tools (an inverse-kinematics helper and a camera helper, neither of which sees the
hidden skin) and a playbook. RGB runs use the setup above and save under `nemotron-rgb` or `nemotron-rgb-assist`.
Use `--depth` for the original numeric observations.

For a short NVIDIA camera and tool test, run `python3 nemotron.py 6 --assist --turns 5 --reasoning-effort low`.
The reasoning setting gets its own run directory, and `transcript.jsonl` records camera and API time per reply.

To compare models on the same proposed moves, use the fixed seed-6 snapshots. The test sends only the images and
telemetry from before each move, plus the proposed action. The label comes from whether the replay move caused a nick.
Gemini Decision 1 approves at `P(clear) > 0.5`. Nemotron Super VL uses low reasoning effort and returns `ALLOW` or
`BLOCK`. Cases and API timings are saved under `runs/snapshot-eval/seed-6/`.

```sh
python3 snapshot_eval.py prepare  # once, from the saved seed-6 Nemotron replays
python3 snapshot_eval.py gemini   # GEMINI_API_KEY, GOOGLE_API_KEY or local .gemini-key
python3 snapshot_eval.py nvidia
python3 snapshot_eval.py report
```

Run `python3 gemini_episode.py 6` for a whole episode with Gemini Decision 1 and the local `.gemini-key`.
Gemini uses the two RGB views and telemetry to pick a cell and cut height. The local controller plans joint moves
and checks them against the nominal head. The simulator scores the actual motion against its hidden scalp. The
runner saves the score, replay, and decision transcript under `runs/seed-6/gemini-decisions/`. Use `--out DIR`
to run again in a new directory.

For a local rehearsal with Harbor, OpenCode and the RGB MCP server, install Docker and `uv`, then run
`python3 harbor_run.py --steps 8`. The default model is Gemini 3.1 Flash-Lite using `.gemini-key`. Omit `--steps`
for the full turn budget. The first run builds a Chromium image. Harbor saves the agent trajectory, simulator
score, and replay under `runs/harbor/`. This task uses seed 6 and gives OpenCode the three Haircut MCP tools.

## What is not modelled

Hair is rigid for scoring. The viewer's strand physics is visual. The blade rotates with the wrist and yaw, and a
cut is determined by the blade tip position. The simulator does not model skin deformation or cutting forces.

## Notes for the harness

- Run `mcp` (or `play`) as a separate process. The skin lives in that process's memory. Use the same seeds for every
  agent and for the reference; with no seed the server draws a random one.
- An agent that can import `haircut.py` or learn the seed can read the skin directly, so keep both out of its reach.
- `replay.html`, `run-SEED.html` and `run-SEED.json` reveal the skin or the score. The MCP server writes its run files
  during the episode, so start each episode's server in a directory the agent cannot read.
- `serve` listens on 127.0.0.1 only and is meant for a human; its `/state` endpoint returns the skin.
