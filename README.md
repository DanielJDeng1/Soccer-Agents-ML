# Soccer Agents ML

2v2 multi-agent reinforcement learning environment built in Unity with ML-Agents, MA-POCA (MultiAgent POsthumous Credit Assignment), and self-play.

Agents are trained with continuous movement controls, team rewards, and limited reward shaping.

## Environment

The values below reflect the current `TrainingEnvironment.prefab`.

- 2 teams with 2 agents each
- 29-value vector observation space
- 2 continuous actions
- Decision period of 5 `FixedUpdate` steps
- 210-second episode limit
- Randomized ball and agent spawn offsets

Each agent observes the ball state, both goals, its own motion and orientation, its teammate, both opponents, and the ball's normalized position on the pitch.

The policy outputs lateral and forward/backward movement. Movement is applied through the agent Rigidbody rather than direct transform movement.

## Rewards

Goal rewards are assigned through `SimpleMultiAgentGroup`:

```text
Goal scored:    +4
Goal conceded:  -4
```

Additional shaping includes:

- team reward for advancing the ball toward the opposing goal
- approach reward for the teammate closest to the ball
- penalty when teammates crowd the ball
- small per-step penalty

## Training

Training is configured in [`config.yaml`](config.yaml) with the POCA trainer and self-play.

```yaml
trainer_type: poca
batch_size: 2048
buffer_size: 20480
learning_rate: 0.0003
hidden_units: 512
num_layers: 2
max_steps: 10000000
```

Self-play snapshots and opponent swaps are configured every 20,000 steps.

## Metrics

`SoccerEnvController` records the following through ML-Agents `StatsRecorder`:

- goals per episode
- episodes ending in a goal
- ball touches
- completed passes
- interceptions

These are available in TensorBoard with the standard ML-Agents training metrics.

## Optional Loophole telemetry (experimental)

The existing TensorBoard stats remain unchanged. For offline reward-exploit analysis, an **opt-in** `SoccerTraceRecorder` component can export agent positions, ball positions, individual shaped-reward contributions, team-reward contributions, and episode outcomes.

1. In Unity, add `SoccerTraceRecorder` to the same GameObject as `SoccerEnvController`.
2. Check **Capture Enabled** on the recorder. The default is **off** so ordinary training is unaffected.
3. Run an episode. The CSV is written under `Application.persistentDataPath/Loophole/`.
4. Use the `episode` and `time_s` columns to align reward events with sampled player and ball positions. `sampleEveryFixedSteps` controls the optional position sampling interval.

The CSV event types are `position`, `reward`, and `outcome`. Reward rows identify the term (`step_penalty`, `ball_approach`, `teammate_crowding`, `ball_advance`, `goal_scored`, or `goal_conceded`). Outcome rows provide episode-level goal, touch, pass, and interception counts. Individual agent positions are sampled; reward events are recorded at the point they are applied. The trace does **not** include video recordings, checkpoint identifiers, or a validated reward-hacking detector.

This is a proposed instrumentation addition, not a finished Loophole implementation. It still needs compilation, scene verification, and a smoke test in Unity 6000.3.16f1 before use in an experiment.

## Files

- `Assets/Scripts/SoccerAgent.cs` — observations, movement, and per-agent reward shaping
- `Assets/Scripts/SoccerEnvController.cs` — team groups, scoring, resets, and training metrics
- `Assets/Prefabs/TrainingEnvironment.prefab` — serialized environment and agent settings
- `config.yaml` — POCA and self-play training configuration

## Setup

Current project versions:

```text
Unity:                 6000.3.16f1
com.unity.ml-agents:   4.1.0
Python:                3.10.12
mlagents:              1.1.0
```

The commands below use Windows PowerShell.

Clone the repository:

```powershell
git clone https://github.com/DanielJDeng1/Soccer-Agents-ML.git
cd Soccer-Agents-ML
```

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install mlagents==1.1.0
```

Open `Assets/Scenes/TestScene.unity` in Unity.

Start the trainer:

```powershell
mlagents-learn config.yaml --run-id=soccer_mapoca_v1
```

Press Play in Unity after the trainer begins listening.

## TensorBoard

```powershell
tensorboard --logdir=results
```

## Inference

After training, import the generated `.onnx` model from `results/<run-id>/` into Unity and assign it to the agents' `Behavior Parameters` component.

Set `Behavior Type` to `Inference Only` or `Default` to run the trained policy.

