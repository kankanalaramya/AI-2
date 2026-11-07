# COSC-4117EL Assignment 2
## Warehouse Robot Decision Making with Dynamic Programming and Reinforcement Learning

**Group:** [Enter group name and members]  
**Course:** COSC-4117EL-01 Artificial Intelligence  
**Term:** Fall 2026

> Replace every bracketed instruction with your own tested results. Do not submit invented measurements.

## 1. Introduction
[Describe the warehouse robot task and the purpose of comparing model-based planning with model-free learning.]

## 2. MDP Formulation
- **States:** `(row, column, has_package, delivered)`.
- **Actions:** up, down, left, right.
- **Transition model:** intended action probability `p`; each perpendicular action has probability `(1-p)/2`. Invalid movement leaves the robot in place.
- **Rewards:** pickup +25, delivery +40, return to dock +100, congestion -10, time step -1, invalid movement -3.
- **Discount factor:** 0.95 in the supplied implementation.
- **Terminal condition:** delivery completed and robot returned to dock; the environment also truncates an episode at its maximum step count.

Explain why position alone is not Markov: the reward and legal task progression depend on whether the package has been collected and delivered.

## 3. Part 1 - Value Iteration
Describe initialization, Bellman optimality update, stopping condition `theta = 1e-5`, `gamma = 0.95`, and policy extraction. Insert the iteration counts and policy images generated in `outputs/`. Explain that Value Iteration requires the transition probabilities and reward model to calculate expected returns.

## 4. Part 2 - Q-Learning
Describe the Q table, epsilon-greedy selection, update:
`Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]`.
State the two tested configurations:
- A: alpha=0.15, gamma=0.95, epsilon decay=0.995.
- B: alpha=0.30, gamma=0.90, epsilon decay=0.990.
State the seeds actually used (11, 22, 33), episode count (2500), and evaluation procedure (epsilon disabled; greedy actions). Insert reward and success-rate plots. Discuss approximate convergence and variability across seeds.

## 5. Part 3 - Stochastic Motion
Compare p=1.00, 0.90, 0.80. Insert the policy images and experiment summaries. Compare average evaluation reward, success rate, and policy behaviour. Identify a state where the selected action differs across p values by inspecting the generated policies; explain the route/congestion trade-off using the actual grid.

## 6. Value Iteration vs Q-Learning
[Compare the methods using the matched-layout, matched-p CSV results: the experiment CSV now includes Value Iteration's 100-episode stochastic evaluation reward and success rate alongside Q-Learning's. Compare these values and the policy visualizations. Explain why Q-Learning may differ due to finite training, exploration, random seeds, and sampling.]

## 7. Environment Changes
[Document the minimal change: optional `intended_probability` parameter in `step`; action outcomes sampled with p and the two perpendicular probabilities. Confirm the original grid, station placement, reward definitions, and Pygame display were retained.]

## 8. Limitations
The four-element state omits remaining steps; the environment's max-step cutoff is treated as an episode truncation by the learning algorithms. A strict finite-horizon Markov model would include remaining steps in the state.

## 9. Group Contributions
[Enter each member's actual contribution. If not applicable, state that.]

## 10. Generative AI Acknowledgment
[Follow the assignment's instruction: identify the AI tool, describe how it was used, and include the relevant ChatGPT conversation/share link(s) for prompts that generated or modified code/report text. Verify the current course policy and add citations in code comments and report.]
