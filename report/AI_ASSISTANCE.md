## Appendix C — AI Assistance Record

**Tool used:** OpenAI ChatGPT / Codex.

AI generated and revised program code, test and analysis scripts, figures, and substantial report text. This account summarizes the assistance recorded in this development session. It is a summary of actual work, not a verbatim prompt transcript. Any additional AI use outside this session should also be disclosed by the group.

### Work performed with AI assistance

| Area | Assistance and resulting artifacts |
|---|---|
| Requirements and explanations | Compared the supplied assignment, starter environment, report template, and rubric. Explained random seeds, task-progress flags, Value Iteration, Q-Learning, and termination versus truncation. |
| Algorithm review and revision | Reviewed the transition/reward model; revised training-seed handling, step-limit handling, Q-Learning targets, matched evaluation settings, and policy/learning plots in the algorithm files. |
| Graphical interface | Generated and revised `warehouse_robot_gui.py`, including resizing, scaled mouse input, task/status information, policy controls, and training progress/cancellation. |
| Tests and experiment execution | Generated `test_algorithms.py` and analysis helpers. Used tools to run algorithm checks, experiments, stochastic-motion sampling, and headless GUI checks. |
| Results and interpretation | Generated `report/analyze_results.py`, combined figures, policy-agreement measurements, the recorded greedy demonstration, and the action-change example at `(5, 5, 1, 1)`. |
| Report and packaging | Drafted and revised the report's explanations and analysis; generated the PDF builder, submission checklist, and ZIP packaging; inspected rendered pages and checked package contents. |

### Verification and evidence

The measurements were produced by executing the project code. The saved evidence includes 18 training runs with 45,000 episode records, 100 greedy evaluations per run, matched VI evaluations, and saved demonstration paths. Automated reproduction compared the new runs with those saved records. Five focused algorithm tests and the documented motion and GUI checks passed; details appear in Appendix A and `report/verification.txt`.

The original warehouse framework was supplied by the instructor, and its original ChatGPT acknowledgment remains in the source. This account does not attribute that original framework to the current AI-assisted session. Automated verification is recorded separately from any manual review performed by the group.

### Disclosure and responsibility

The reason for omitting conversation links and the proposed alternative disclosure are stated in Section 10. This account documents the scope of assistance; it does not establish instructor approval of an alternative citation format. The GitHub link identifies the project repository.

The group is responsible for understanding the algorithms, checking the submitted work, accurately reporting any additional AI use, and meeting the course's requirements. No individual member contribution is inferred from this AI-assistance record.
