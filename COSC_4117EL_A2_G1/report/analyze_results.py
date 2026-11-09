# BEGIN AI ACKNOWLEDGMENT
# AI-assisted project preparation: OpenAI ChatGPT / Codex.
# Scope and disclosure: report/REPORT.md, Section 10.
# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C).
# Project repository: https://github.com/kankanalaramya/AI-2
# Conversation links omitted for privacy; see the disclosure in Section 10.
# END AI ACKNOWLEDGMENT
"""Reproduce the saved experiments and prepare evidence for the assignment report.
Run from the project folder: python report/analyze_results.py
AI assistance: ChatGPT assisted with this analysis; see submission_details.json.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from warehouse_gridworld_domain_random import WarehouseGridWorld, ACTIONS, SHELF
from value_iteration import value_iteration, transition_outcomes, terminal_state, state_list, ARROWS
from q_learning import train_q_learning, plot_learning

REPORT = ROOT / 'report'
FIGURES = REPORT / 'figures'
FIGURES.mkdir(exist_ok=True)
COLORS = ['#f4f6f8', '#303844', '#f2b1a8', '#89b5ef', '#f2d270', '#91d4b1']


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def action_values(world, state, p, gamma, values):
    scores = {}
    for action in ACTIONS:
        scores[action] = sum(prob * (reward + (0 if done else gamma * values[next_state]))
                             for prob, next_state, reward, done in transition_outcomes(world, state, action, p))
    return scores


def evaluate(world, policy, p, seed):
    rng = np.random.default_rng(seed)
    totals, successes, first_trace = [], [], []
    visited = set()
    for run in range(100):
        world.np_random = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        state = world.reset()
        total = 0
        trace = []
        while not world.done:
            visited.add(state)
            action = policy[state]
            result = world.step(action, intended_probability=p)
            total += result.reward
            trace.append({'step': world.steps, 'before': list(state), 'action': action,
                          'actual_action': result.info['actual_action'], 'state': list(result.state),
                          'reward': float(result.reward), 'total_reward': float(total),
                          'event': result.info['event'], 'congestion': bool(result.info.get('congestion'))})
            state = result.state
        totals.append(total)
        successes.append(result.info['event'] == 'return_to_dock')
        if run == 0:
            first_trace = trace
    return sum(totals) / 100, sum(successes) / 100, first_trace, visited


def policy_panel(ax, world, policy, phase, title, highlight=False):
    ax.imshow(world.grid, cmap=ListedColormap(COLORS), vmin=0, vmax=5)
    for row in range(10):
        for col in range(10):
            cell = world.grid[row, col]
            label = {2: '!', 3: 'P', 4: 'D', 5: 'C'}.get(int(cell), '')
            if label:
                ax.text(col, row - 0.17, label, ha='center', va='center', fontsize=13, weight='bold')
            action = policy.get((row, col, *phase))
            if action:
                ax.text(col, row + 0.23, ARROWS[action], ha='center', va='center', fontsize=14, color='#214b85')
    if highlight:
        ax.add_patch(Rectangle((4.5, 4.5), 1, 1, fill=False, edgecolor='#9c2c18', linewidth=3))
    ax.set_title(title, fontsize=15)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(labelsize=11, length=0)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which='minor', color='#bac4ce', linewidth=.5)
    ax.tick_params(which='minor', length=0)


def save_phases(world, policy, filename):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, phase, title in zip(axes, [(0, 0), (1, 0), (1, 1)], ['Before pickup', 'Carrying package', 'After delivery']):
        policy_panel(ax, world, policy, phase, title)
    fig.text(.5, .015, 'Coordinates: row, column.  P: pickup   D: packing   C: dock   !: congestion   Dark cells: shelves',
             ha='center', fontsize=12)
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def main():
    results = read_csv(ROOT / 'outputs/q_learning_experiment_results.csv')
    assert len(results) == 18
    records, summaries, selected = [], [], {}
    episode_data = {}
    for row in results:
        config, p, seed = row['configuration'], float(row['p']), int(row['seed'])
        gamma, alpha = float(row['gamma']), float(row['alpha'])
        decay = float(row['epsilon_decay'])
        prefix = f'{config}_p_{p:.2f}_seed_{seed}'
        saved = read_csv(ROOT / 'outputs' / (prefix + '_episodes.csv'))
        assert len(saved) == 2500
        world = WarehouseGridWorld(seed=42)
        values, vi_policy, iterations = value_iteration(world, p, gamma=gamma)
        q, rewards, successes, epsilons = train_q_learning(world, p, seed, alpha=alpha, gamma=gamma, epsilon_decay=decay)
        np.testing.assert_allclose(rewards, [float(r['reward']) for r in saved], rtol=0, atol=0)
        np.testing.assert_array_equal(successes, [int(r['success']) for r in saved])
        np.testing.assert_allclose(epsilons, [float(r['epsilon']) for r in saved], rtol=0, atol=1e-14)
        q_policy = {s: ACTIONS[int(np.argmax(q[s]))] for s in q if not terminal_state(s, world)}
        q_reward, q_success, trace, visited = evaluate(world, q_policy, p, seed + 500)
        vi_reward, vi_success, _, _ = evaluate(world, vi_policy, p, seed + 500)
        assert np.isclose(q_reward, float(row['evaluation_mean_reward']))
        assert np.isclose(vi_reward, float(row['value_iteration_evaluation_mean_reward']))
        assert q_success == float(row['evaluation_success_rate']) == 1.0
        assert vi_success == float(row['value_iteration_evaluation_success_rate']) == 1.0
        assert iterations == int(row['value_iteration_iterations'])
        assert gamma == float(row['value_iteration_gamma'])
        saved_path = read_csv(ROOT / 'outputs' / (prefix + '_greedy_path.csv'))
        states = [(*world.dock_pos, 0, 0)] + [tuple(t['state']) for t in trace]
        assert states == [tuple(int(r[k]) for k in ['row','column','has_package','delivered']) for r in saved_path]
        all_decisions = [s for s in q if not terminal_state(s, world)]
        agreement = sum(q_policy[s] == vi_policy[s] for s in all_decisions)
        visited_agreement = sum(q_policy[s] == vi_policy[s] for s in visited)
        near_best = 0
        for s in visited:
            scores = action_values(world, s, p, gamma, values)
            near_best += scores[q_policy[s]] >= max(scores.values()) - 1e-6
        success_ma = np.convolve(successes, np.ones(100) / 100, mode='valid')
        crossing = np.flatnonzero(success_ma >= .95)
        first_95 = int(crossing[0] + 100) if len(crossing) else None
        record = {'configuration': config, 'p': p, 'seed': seed, 'gamma': gamma,
                  'q_reward': q_reward, 'vi_reward': vi_reward, 'success_rate': q_success,
                  'last100_reward': float(np.mean(rewards[-100:])), 'first_95pct_success_window': first_95,
                  'all_state_agreement': agreement, 'all_state_count': len(all_decisions),
                  'visited_agreement': visited_agreement, 'visited_count': len(visited),
                  'visited_near_best': near_best, 'vi_iterations': iterations}
        records.append(record)
        episode_data[(config, p, seed)] = (rewards, successes)
        if seed == 11:
            selected[(config, p)] = q_policy
        if config == 'config_A' and p == .8 and seed == 11:
            (REPORT / 'greedy_demo_trace.json').write_text(json.dumps(trace, indent=2))
            plot_learning(rewards, successes, FIGURES / 'baseline_learning.png',
                          'Q-Learning config_A | p=0.80 | seed=11')
            save_phases(world, vi_policy, 'vi_phases.png')
            save_phases(world, q_policy, 'ql_phases.png')
        print('Reproduced:', prefix, 'agreement', f'{agreement}/{len(all_decisions)}', 'visited', f'{visited_agreement}/{len(visited)}', flush=True)
    for config in ['config_A', 'config_B']:
        for p in [1.0, .9, .8]:
            rows = [r for r in records if r['configuration'] == config and r['p'] == p]
            summaries.append({'configuration': config, 'p': p, 'gamma': rows[0]['gamma'],
                              'q_mean': float(np.mean([r['q_reward'] for r in rows])),
                              'q_sd': float(np.std([r['q_reward'] for r in rows])),
                              'vi_mean': float(np.mean([r['vi_reward'] for r in rows])),
                              'vi_sd': float(np.std([r['vi_reward'] for r in rows])),
                              'first95_min': min(r['first_95pct_success_window'] for r in rows),
                              'first95_max': max(r['first_95pct_success_window'] for r in rows),
                              'iterations': rows[0]['vi_iterations']})
    world = WarehouseGridWorld(seed=42)
    change = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, p in zip(axes, [1.0, .9, .8]):
        values, policy, _ = value_iteration(world, p)
        scores = action_values(world, (5, 5, 1, 1), p, .95, values)
        change.append({'p': p, 'state': [5,5,1,1], 'action': policy[(5,5,1,1)], 'values': scores})
        policy_panel(ax, world, policy, (1, 1), f'p = {p:.2f}', highlight=True)
    fig.text(.5, .015, 'After-delivery VI policies, gamma = 0.95.  Highlighted cell: (row 5, column 5).', ha='center', fontsize=12)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(FIGURES / 'motion_policies.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey='row')
    for column, p in enumerate([1.0,.9,.8]):
        for config, color, style in [('config_A','#245e98','-'),('config_B','#b76421','--')]:
            reward_lines, success_lines = [], []
            for seed in [11,22,33]:
                rewards, successes = episode_data[(config,p,seed)]
                reward_lines.append(np.convolve(rewards, np.ones(100)/100, mode='valid'))
                success_lines.append(np.convolve(successes, np.ones(100)/100, mode='valid'))
            for row, lines in enumerate([reward_lines,success_lines]):
                lines=np.array(lines)
                axes[row,column].plot(range(100,2501), lines.mean(axis=0), color=color, linestyle=style, label=config)
                axes[row,column].fill_between(range(100,2501),lines.min(axis=0),lines.max(axis=0),color=color,alpha=.13)
                axes[row,column].grid(alpha=.25)
                axes[row,column].tick_params(labelsize=12)
        axes[0,column].set_title(f'p = {p:.2f}',fontsize=16)
        axes[1,column].set_xlabel('Episode',fontsize=13)
        axes[1,column].set_ylim(0,1.02)
    axes[0,0].set_ylabel('100-episode mean reward',fontsize=13)
    axes[1,0].set_ylabel('100-episode success rate',fontsize=13)
    axes[0,0].legend(fontsize=12)
    fig.suptitle('Training means across three seeds; shading shows the seed range',fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES/'learning_comparison.png',dpi=180)
    plt.close(fig)
    evidence = {'records': records, 'summaries': summaries, 'action_change': change,
                'verification': {'training_runs_reproduced':18,'episode_records_reproduced':45000,
                                 'q_evaluation_episodes':1800,'vi_evaluation_episodes':1800,
                                 'saved_demo_paths_reproduced':18}}
    (REPORT/'analysis_summary.json').write_text(json.dumps(evidence,indent=2))
    print('Report evidence and figures written.', flush=True)


if __name__ == '__main__':
    main()
