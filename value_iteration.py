# BEGIN AI ACKNOWLEDGMENT
# AI-assisted project preparation: OpenAI ChatGPT / Codex.
# Scope and disclosure: report/REPORT.md, Section 10.
# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C).
# Project repository: https://github.com/kankanalaramya/AI-2
# Conversation links omitted for privacy; see the disclosure in Section 10.
# END AI ACKNOWLEDGMENT
"""
Value Iteration for COSC-4117EL Assignment 2.
Uses the supplied WarehouseGridWorld domain and its reward rules.

AI assistance acknowledgment: see the source comments above, report Section 10, and Appendix C for scope and disclosure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from warehouse_gridworld_domain_random import (
    WarehouseGridWorld, ACTIONS, ACTION_TO_DELTA,
    SHELF, CONGESTION, PICKUP_REWARD, DELIVERY_REWARD, DOCK_REWARD,
    CONGESTION_PENALTY, STEP_PENALTY, INVALID_MOVE_PENALTY
)

GAMMA = 0.95
THETA = 1e-5
PROBABILITIES = (1.00, 0.90, 0.80)
ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


def terminal_state(state, world):
    r, c, has_package, delivered = state
    return bool(delivered and (r, c) == world.dock_pos)


def state_list(world):
    """Valid floor cells × the three logically possible task-progress stages."""
    states = []
    for r in range(world.size):
        for c in range(world.size):
            if world.grid[r, c] != SHELF:
                states.extend([
                    (r, c, 0, 0),  # package not collected
                    (r, c, 1, 0),  # carrying package
                    (r, c, 1, 1),  # package delivered
                ])
    return states


def one_outcome(world, state, actual_action):
    """Return (next_state, reward, done) for one actual movement outcome."""
    r, c, has_package, delivered = state
    dr, dc = ACTION_TO_DELTA[actual_action]
    nr, nc = r + dr, c + dc
    reward = STEP_PENALTY

    if not (0 <= nr < world.size and 0 <= nc < world.size) or world.grid[nr, nc] == SHELF:
        nr, nc = r, c
        reward += INVALID_MOVE_PENALTY

    if world.grid[nr, nc] == CONGESTION:
        reward += CONGESTION_PENALTY

    if (nr, nc) == world.pickup_pos and not has_package:
        has_package = 1
        reward += PICKUP_REWARD

    if (nr, nc) == world.packing_pos and has_package and not delivered:
        delivered = 1
        reward += DELIVERY_REWARD

    done = False
    if (nr, nc) == world.dock_pos and delivered:
        reward += DOCK_REWARD
        done = True

    return (nr, nc, has_package, delivered), float(reward), done


def transition_outcomes(world, state, action, intended_probability):
    """Exact model P(s'|s,a), including probability mass for duplicate outcomes."""
    if not 0 <= intended_probability <= 1:
        raise ValueError("Probability must be in [0, 1].")
    perpendicular = {
        "up": ("left", "right"),
        "down": ("left", "right"),
        "left": ("up", "down"),
        "right": ("up", "down"),
    }
    a, b = perpendicular[action]
    raw = [
        (action, intended_probability),
        (a, (1.0 - intended_probability) / 2.0),
        (b, (1.0 - intended_probability) / 2.0),
    ]
    outcomes = []
    for actual, probability in raw:
        if probability > 0:
            ns, reward, done = one_outcome(world, state, actual)
            outcomes.append((probability, ns, reward, done))
    return outcomes


def value_iteration(world, intended_probability=0.80, gamma=GAMMA, theta=THETA):
    states = state_list(world)
    values = {s: 0.0 for s in states}
    policy = {s: "up" for s in states}
    iterations = 0

    while True:
        delta = 0.0
        new_values = values.copy()
        for state in states:
            if terminal_state(state, world):
                new_values[state] = 0.0
                policy[state] = None
                continue

            action_values = []
            for action in ACTIONS:
                total = 0.0
                for prob, next_state, reward, done in transition_outcomes(
                    world, state, action, intended_probability
                ):
                    future = 0.0 if done else values.get(next_state, 0.0)
                    total += prob * (reward + gamma * future)
                action_values.append(total)

            best_index = int(np.argmax(action_values))
            new_values[state] = action_values[best_index]
            policy[state] = ACTIONS[best_index]
            delta = max(delta, abs(new_values[state] - values[state]))

        values = new_values
        iterations += 1
        if delta < theta:
            break
        if iterations >= 10000:
            raise RuntimeError("Value Iteration did not converge within 10,000 iterations.")

    # Re-extract policy from the converged value function.
    for state in states:
        if terminal_state(state, world):
            policy[state] = None
            continue
        q_values = []
        for action in ACTIONS:
            q = 0.0
            for prob, ns, reward, done in transition_outcomes(world, state, action, intended_probability):
                q += prob * (reward + (0.0 if done else gamma * values.get(ns, 0.0)))
            q_values.append(q)
        policy[state] = ACTIONS[int(np.argmax(q_values))]

    return values, policy, iterations


def draw_policy(world, policy, filename, title, phase=(1, 0)):
    """Draw one task-progress phase: (has_package, delivered)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ListedColormap(["#f4f6f8", "#303844", "#f2b1a8", "#89b5ef", "#f2d270", "#91d4b1"])
    ax.imshow(world.grid, cmap=colors, vmin=0, vmax=5)
    has_package, delivered = phase
    for r in range(world.size):
        for c in range(world.size):
            cell = int(world.grid[r, c])
            if cell == SHELF:
                label = "#"
            elif cell == CONGESTION:
                label = "!"
            elif (r, c) == world.pickup_pos:
                label = "P"
            elif (r, c) == world.packing_pos:
                label = "D"
            elif (r, c) == world.dock_pos:
                label = "C"
            else:
                label = ""
            color = "white" if cell == SHELF else "black"
            ax.text(c, r - 0.1, label, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
            action = policy.get((r, c, has_package, delivered))
            if action:
                ax.text(c, r + 0.28, ARROWS[action], ha="center", va="center", fontsize=10, color="navy")
    phase_name = {(0, 0): "Before pickup", (1, 0): "Carrying package", (1, 1): "After delivery"}[phase]
    ax.set_title(title + f"\n{phase_name} policy")
    ax.set_xticks(range(world.size))
    ax.set_yticks(range(world.size))
    ax.set_xlabel("Column | P: pickup, D: packing, C: dock, !: congestion, #: shelf")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(-0.5, world.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.size, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5, color="gray")
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)
    world = WarehouseGridWorld(seed=42)
    print("Value Iteration | fixed layout seed=42")
    print("Pickup:", world.pickup_pos, "Packing:", world.packing_pos, "Dock:", world.dock_pos)
    summary = []
    all_policies = {}
    for p in PROBABILITIES:
        values, policy, iterations = value_iteration(world, p)
        all_policies[p] = policy
        for phase, suffix in [((0, 0), "before_pickup"), ((1, 0), "carrying"), ((1, 1), "after_delivery")]:
            out = f"outputs/value_iteration_p_{p:.2f}_{suffix}.png"
            draw_policy(world, policy, out, f"Value Iteration (p={p:.2f})", phase)
        start = world.get_state()
        summary.append((p, iterations, values[start]))
        print(f"p={p:.2f}: iterations={iterations}, V(start)={values[start]:.3f}")

    with open("outputs/value_iteration_summary.csv", "w", encoding="utf-8") as f:
        f.write("probability,iterations,start_state_value\n")
        for row in summary:
            f.write(",".join(map(str, row)) + "\n")

    # Identify a real state whose optimal action changes as motion becomes less reliable.
    changes = []
    for state in state_list(world):
        actions = [all_policies[p].get(state) for p in PROBABILITIES]
        if len(set(actions)) > 1:
            changes.append((state, *actions))
    with open("outputs/value_iteration_action_changes.csv", "w", encoding="utf-8") as f:
        f.write("state,p_1.00_action,p_0.90_action,p_0.80_action\n")
        for row in changes:
            f.write(f'"{row[0]}",{row[1]},{row[2]},{row[3]}\n')
    if changes:
        print("Example policy action change:", changes[0])
    else:
        print("No action changes found in this layout; inspect other seeded layouts before making the Part 3 claim.")
    print("Done. Results are in the outputs folder.")


if __name__ == "__main__":
    main()
