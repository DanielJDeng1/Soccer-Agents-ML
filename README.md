Markdown
# Soccer Agents ML

Multi-agent reinforcement learning environment built in Unity using ML-Agents and MA-POCA (Multi-Agent Post-Processing Curriculum Assessment).

Agents learn competitive 2v2 soccer dynamics using continuous controls, multi-agent group rewards, and spatial shaping signals to prevent swarming and encourage tactical positioning.

## Project Structure

* **`Assets/Scripts/SoccerAgent.cs`**: Agent class handling observations, continuous physics controls, and individual proximity rewards.
* **`Assets/Scripts/SoccerEnvController.cs`**: Environment manager tracking match state, goal triggers, timer resets, and team-level group rewards via `SimpleMultiAgentGroup`.
* **`Assets/Prefabs/`**: Pre-configured environment prefab containing pitch geometry, goal triggers, and agent spawn points.
* **`config.yaml`**: PPO/MA-POCA training hyperparameters for ML-Agents.

## Environment Specifications

* **Vector Observation Space**: 28 floats
  * Ball position (relative, normalized) and linear velocity
  * Relative offsets to own goal and opponent goal
  * Agent linear velocity and forward vector
  * Relative positions of teammates and opponents
  * Pitch bounds normalized coordinates
* **Action Space**: Continuous (Size: 2)
  * `Actions[0]`: Lateral movement force (X-axis)
  * `Actions[1]`: Forward/Backward movement force (Z-axis)
* **Decision Frequency**: Every 5 FixedUpdate steps

## Setup & Installation

### Requirements

* Unity 2022.3 LTS or newer
* Python 3.10.x
* ML-Agents Python Package (`mlagents >= 0.30.0`)
* PyTorch

### Environment Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/DanielJDeng1/Soccer-Agents-ML.git](https://github.com/DanielJDeng1/Soccer-Agents-ML.git)
   cd Soccer-Agents-ML
Create and activate a Python virtual environment:

Bash
python -m venv ml_env
.\ml_env\Scripts\Activate.ps1
Install required Python packages:

Bash
pip install mlagents torch
Training
Open the project in Unity Editor and load Assets/Scenes/TestScene.unity.

Open your terminal, activate ml_env, and run:

Bash
mlagents-learn config.yaml --run-id=soccer_mapoca_v1 --force
When the terminal prompts Listening on port 5004, press Play in Unity.

TensorBoard metrics log automatically to the ./results folder. View training progress by running:

Bash
tensorboard --logdir=results
Inference
To run trained policies in Unity:

Locate the generated .onnx file in results/soccer_mapoca_v1/.

Drag the .onnx file into Assets/ML-Agents/.

Select the agent GameObjects in your scene prefab.

Drag the .onnx model asset into the Model field of the Behavior Parameters component.

Set Behavior Type to Inference Only or Default.
