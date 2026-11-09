# COSC-4117EL Assignment 2 — Warehouse Robot Grid World

## Project overview

This project models a warehouse robot operating in a grid-world environment. The robot must collect a package, take it to the packing station, and then return to the charging dock.

Project repository: [kankanalaramya/AI-2](https://github.com/kankanalaramya/AI-2).

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

Create a virtual environment and install the dependencies from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

The GUI starts with layout seed 42 and `p = 0.80` (Parts 1 and 2). It supports resizing and maximizing. Buttons also respond to mouse clicks. A calculated/trained policy starts paused so you can inspect it before pressing Play.

## GUI keyboard controls

| Key | Action |
|---|---|
| `M` | Switch to manual control mode. |
| `V` | Calculate and load the Value Iteration policy. |
| `Q` | Train and load the Q-Learning policy. |
| `SPACE` | Play or pause automatic AI movement. |
| `UP`, `DOWN`, `LEFT`, `RIGHT` | Move in the specified direction while playback is paused. |
| `S` | Pause playback and take one step using the loaded policy. |
| `H` | Show or hide policy arrows for the current task stage. |
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

Q-Learning shows progress every 100 episodes. After training, press Play or Space to run the greedy policy with exploration disabled. Escape closes the window, including during training.

## Run the algorithms from the terminal

Run these commands from the project folder:

```bash
python value_iteration.py
python q_learning.py
```

The scripts generate experiment results in the `outputs/` folder. Run the scripts before referring to their charts or CSV results in the assignment report.

Q-Learning runs two configurations, three movement probabilities, and seeds 11, 22, and 33. Each run uses 2,500 training episodes and 100 greedy evaluation episodes. Layout seed 42 stays fixed; training seeds control motion and exploration. Each VI baseline uses the same layout, probability, discount factor, and evaluation seed as its Q-Learning comparison. The GUI is a demonstration and does not save these experiment files.

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

The environment ends an episode when the robot returns to the dock after completing the delivery task, or when the 120-step limit is reached. Every action consumes a step, including an invalid move. There is no wall-clock deadline; GUI playback speed does not change the episode limit.

The tabular state listed above does not include the number of steps remaining. The algorithms model the discounted task with successful return to the dock as its terminal state. The maximum-step limit stops an episode for training and evaluation, but Q-Learning still includes the estimated future value when updating at a timeout. It uses only the immediate reward when the task succeeds. Evaluation counts timeouts as failures. A fully Markov finite-horizon formulation would include the remaining number of steps in the state. Explain this modeling choice in the report.

## Output files

The `outputs/` directory is used for generated experiment charts and CSV summaries. The exact files present depend on which experiment scripts have been run. Do not report results that you have not generated and checked.

- `value_iteration_summary.csv`: convergence iterations and discounted start-state values at gamma 0.95.
- `value_iteration_action_changes.csv`: states whose selected actions differ across motion settings. Inspect these before explaining a route change; a change may also break a tie.
- `q_learning_experiment_results.csv`: per-run settings, seeds, mean evaluation rewards, and success rates for both methods.
- `*_episodes.csv`: each training episode's reward, success, and exploration probability.
- `*_greedy_path.csv`: states visited during the first evaluation episode, starting at step zero. A path ending away from the dock is a failed demonstration.
- Policy images: before pickup, carrying, and after delivery. Files named `config_*_vi_*` use the discount factor of the corresponding Q-Learning configuration.
- `*_learning.png`: episode rewards and aligned moving averages of reward and success.
- `algorithm_probability_comparison.png`: both methods' evaluation means, with standard deviations across the three seeds. These are not confidence intervals.

Run the focused algorithm checks with `python -m unittest -v test_algorithms.py`. Exclude `.venv/` and `__pycache__/` from the submission ZIP.

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

The supplied assignment requires AI-use acknowledgment and actual conversation/prompt links in **both code comments and the report**. Review, understand, test, and adapt the work yourself. An individual contribution breakdown is optional. If requesting a grade adjustment, document the actual agreement of all group members. Do not claim manual checks that you have not performed.

This version retains the AI-use acknowledgment and includes a detailed assistance
record in `report/AI_ASSISTANCE.md` (Appendix C of the PDF). Section 10 explains the
privacy reason for omitting conversation links and requests consideration of the
disclosure as an alternative. Instructor acceptance has not been confirmed. The
repository link does not replace the assignment's separate prompt-link requirement.

## Report and submission

The complete report is `report/COSC_4117EL_A2_G1_Report.pdf`; its editable source is
`report/REPORT.md`. The earlier `REPORT_TEMPLATE.md` is an outline, not the final report.
Verified experiment CSVs are included in `outputs/`.

Read **`SUBMISSION_CHECKLIST.md`**. Group details and the project repository are
filled in, the D2L standards review is confirmed, and chat links are currently
omitted. To rebuild after editing `report/submission_details.json`:

```bash
python -m pip install -r report/requirements.txt
python report/build_report.py
```

These extra packages are needed only to rebuild the report. The builder updates
the PDF and source-code attribution, then creates `COSC_4117EL_A2_G1.zip` with the
current details. It supports building with chat links omitted and reports the
remaining prompt-link requirement. Existing experiments do not need to be rerun for
identity/citation edits. The final ZIP excludes virtual environments, installed
packages, caches, and the bulk generated output charts.

For full evidence verification, `python report/analyze_results.py` reruns all 18
training configurations, checks the saved records and evaluation paths, and
recreates combined figures. If implementation/settings change, rerun both
algorithm scripts first and revise the report accordingly.
