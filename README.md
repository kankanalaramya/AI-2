# COSC-4117EL Assignment 2 — Warehouse Robot Grid World

## Project overview

This project models a warehouse robot operating in a grid-world environment. The robot must collect a package, take it to the packing station, and then return to the charging dock.

The project includes:
- A stochastic warehouse grid-world environment.
- A Pygame graphical user interface (GUI) for manual control and AI demonstrations.
- A model-based Value Iteration implementation.
- A model-free Q-Learning implementation.
- Experiment output files (charts and CSV summaries) saved in the `outputs/` folder when the experiment scripts are run.

## Project files

Keep these Python files together in the same project folder:

| File | Purpose |
|---|---|
| `warehouse_gridworld_domain_random.py` | Warehouse environment, grid layout, robot movement, rewards, and episode handling. |
| `warehouse_robot_gui.py` | Complete Pygame GUI for manual control, Value Iteration, and Q-Learning. |
| `value_iteration.py` | Model-based Value Iteration algorithm and related experiment code. |
| `q_learning.py` | Model-free Q-Learning algorithm and related experiment code. |
| `requirements.txt` | Python package dependencies. |
| `outputs/` | Generated experiment charts and CSV files. |

## Requirements

- Python 3.11 or later.
- NumPy.
- Matplotlib.
- Pygame (required for the graphical interface).

Install the dependencies from the project folder:

```bash
python -m pip install -r requirements.txt
```

If Pygame is not included in the requirements file or the GUI reports that it is missing, install it with:

```bash
python -m pip install pygame
```

## Run the complete graphical interface

1. Extract the project ZIP file.
2. Open the extracted project folder in Visual Studio Code.
3. Open the VS Code terminal. Make sure the terminal is in the folder containing `warehouse_robot_gui.py`.
4. Install the dependencies.
5. Start the GUI:

```bash
python warehouse_robot_gui.py
```

**Important:** Run `warehouse_robot_gui.py` to open the complete application. Running `warehouse_gridworld_domain_random.py` starts the environment module's own demonstration, not the integrated AI GUI.

All project modules (`warehouse_robot_gui.py`, `warehouse_gridworld_domain_random.py`, `value_iteration.py`, and `q_learning.py`) should remain in the same folder so Python can import them.

## GUI keyboard controls

| Key | Action |
|---|---|
| `M` | Switch to manual control mode. |
| `V` | Calculate and load the Value Iteration policy. |
| `Q` | Train and load the Q-Learning policy. |
| `SPACE` | Play or pause automatic AI movement. |
| `UP`, `DOWN`, `LEFT`, `RIGHT` | Move the robot manually, or move one step when AI playback is paused. |
| `1` | Set intended-action probability to 1.00. |
| `2` | Set intended-action probability to 0.90. |
| `3` | Set intended-action probability to 0.80. |
| `R` | Reset the current episode. |
| `N` | Generate the next seeded warehouse layout. |
| `+` / `-` | Increase / decrease AI playback speed. |
| `ESC` | Close the GUI. |

### Suggested GUI test

1. Run `python warehouse_robot_gui.py`.
2. Press `V` to calculate the Value Iteration policy.
3. When the policy is ready, press `SPACE` to start automatic movement.
4. Press `R` to reset the episode.
5. Press `Q` to train the Q-Learning policy.
6. When training is complete, press `SPACE` to start the learned policy.
7. Try `1`, `2`, and `3` to compare different intended-action probabilities. Changing the probability invalidates the currently loaded AI policy, so calculate/train the policy again after changing it.

Q-Learning training may take some time. Wait for the training message to finish before expecting the robot to move automatically.

## Run the algorithms from the terminal

Run these commands from the project folder:

```bash
python value_iteration.py
python q_learning.py
```

The scripts generate experiment results in the `outputs/` folder. Run the scripts before referring to their charts or CSV results in the assignment report.

## State representation

The tabular state is represented as:

```text
(row, column, has_package, delivered)
```

The two progress flags distinguish the robot's task stage. The same grid position can represent different situations depending on whether the robot has collected or delivered the package.

## Stochastic movement

The environment accepts an intended-action probability `p`. With probability `p`, the robot moves in the intended direction. The remaining probability is divided between the two perpendicular directions, each receiving `(1 - p) / 2`.

The GUI provides three probability settings:
- `p = 1.00`
- `p = 0.90`
- `p = 0.80`

## Episode termination and modeling note

The environment can end an episode when the robot returns to the dock after completing the delivery task, or when the maximum step limit is reached.

The tabular state listed above does not include the number of steps remaining. Therefore, the maximum-step limit is treated as an episode truncation by the tabular algorithms. A fully Markov finite-horizon formulation would include the remaining number of steps in the state. Explain this modeling choice in the report if it is relevant to the assignment requirements.

## Output files

The `outputs/` directory is used for generated experiment charts and CSV summaries. The exact files present depend on which experiment scripts have been run. Do not report results that you have not generated and checked.

## Troubleshooting

### `ModuleNotFoundError: No module named 'warehouse_gridworld_domain_random'`

- Confirm that `warehouse_robot_gui.py` and `warehouse_gridworld_domain_random.py` are in the same folder.
- Open that folder in VS Code using **File → Open Folder**.
- Open a new terminal in VS Code and run the GUI from the project folder:

```bash
python warehouse_robot_gui.py
```

### `ModuleNotFoundError` for NumPy, Matplotlib, or Pygame

Install the dependencies:

```bash
python -m pip install numpy matplotlib pygame
```

### The GUI opens but the robot does not move

- In manual mode, use the arrow keys.
- For automatic movement, press `V` or `Q` first, wait for the policy calculation/training to finish, and then press `SPACE`.
- If the episode has ended, press `R` to reset it.
- If you change the movement probability, calculate or train a policy again.

### The GUI closes immediately

Run it from the VS Code terminal rather than using an unrelated file or a different working directory:

```bash
python warehouse_robot_gui.py
```

Read the full error message printed in the terminal if it still fails.

## Academic integrity and AI acknowledgment

Follow the course instructor's academic-integrity and generative-AI acknowledgment requirements. Review, understand, test, and adapt the code yourself. Include the required AI-use acknowledgment and conversation/share link in the submitted report or code if required by the assignment instructions. Add the actual group member names and their contributions to the report. Do not claim results or testing that you have not personally completed.
