# Finish the submission

The report content, experiment measurements, and figures are complete. The PDF is
`report/COSC_4117EL_A2_G1_Report.pdf`. Group 19's member details are filled in, and
the group has confirmed that the D2L programming standards are satisfied.
The individual contribution breakdown has been omitted as requested.
The project repository is [kankanalaramya/AI-2](https://github.com/kankanalaramya/AI-2).
Chat links are omitted for privacy, as requested. Section 10 records the reason,
and Appendix C provides a detailed AI-assistance account. Its editable source is
`report/AI_ASSISTANCE.md`. The assignment's prompt-link requirement remains
unresolved unless the instructor accepts this alternative; the disclosure does
not claim that acceptance has already been granted.

## 1. Fill one file

Edit `report/submission_details.json`:

| Field | What to enter | Where it appears |
|---|---|---|
| `group` | Your actual group name or number | Report cover |
| `members` | Actual names; add/remove member objects as needed | Cover and Section 9 |
| `student_id` | Optional, unless D2L standards require it | Beside the name |
| `contribution` | Optional; omitted for this submission. If supplied, describe actual work accurately | Section 9, only when supplied |
| `ai_tools`, `ai_use` | Review the supplied description; add any other tools/use by your groupmate | Section 10 |
| `repository_url` | Project GitHub repository; already filled in | Cover, Section 9, and Python comments |
| `ai_conversation_links` | Currently `[]` to omit chat links. Add actual share/prompt links later if available | Section 10 and Python comments, when supplied |
| `ai_link_omission_note` | Review the privacy explanation and proposed alternative disclosure for accuracy | Section 10 when chat links are omitted |
| `grade_adjustment_statement` | Optional; leave blank to omit it. If requesting an adjustment, describe the change and all members' actual agreement | Section 9, only when supplied |
| `d2l_programming_standards_checked` | Already `true`, based on your confirmation | Enables final ZIP generation |

An empty AI-link list omits that section without adding placeholder text. If you
add links later, use actual conversation/prompt links and check that they open
for the marker. The GitHub URL is stored separately as the project repository.

The assignment requires AI acknowledgment in **both source comments and the report**.
The builder inserts the supplied links into every submitted Python file. The
instructor's original template attribution remains separate.

## 2. Rebuild the PDF and final ZIP

From the project root, with your virtual environment activated:

```bash
python -m pip install -r report/requirements.txt
python report/build_report.py
```

On the machine used to prepare this report, the required PDF packages are already
available in the system's Anaconda Python, so this is also sufficient:

```bash
/home/uthp/anaconda3/bin/python report/build_report.py
```

Training does **not** need to be rerun for these edits. The builder updates the
PDF and source-code attribution, then creates **`COSC_4117EL_A2_G1.zip`** in the
project root, as named in the supplied assignment. Chat links may be omitted from
the build; ZIP creation does not mean the assignment's prompt-link requirement
has been met. Other missing required details are still reported.

The ZIP contains the runtime source files, focused tests, README, requirements,
PDF, editable report inputs, and compact evidence CSVs. It excludes `.venv`,
installed packages, caches, Git metadata, and the bulk generated output charts.
The embedded report figures remain included so the PDF can be rebuilt.

Open the regenerated PDF, check the cover and Sections 9–10, confirm that no
`[FILL:` markers remain, and inspect the ZIP contents before uploading.

## 3. Manual checks

**Already confirmed:** you have checked the separate D2L programming standards.
This is recorded as your confirmation, not an automated inspection of that document.
Before submission, review the report and make sure the group details and all AI
citations are accurate.

**Recommended desktop check:** automated GUI checks used a headless display, so
briefly run:

```bash
python warehouse_robot_gui.py
```

Maximize/resize the window, select VI, press Play, reset, train Q-Learning, and press
Play again. Confirm that the route completes pickup → packing → dock, the text is
readable, and clicks still work after resizing. This checks your actual desktop
window manager. No additional screenshots or experimental measurements are needed
for the current report.

## Already checked

- Five focused algorithm tests passed, including all 920 deterministic model outcomes.
- All 18 experiments reproduced: 45,000 training records and 18 saved greedy paths.
- Evaluation reproduced over 1,800 Q-Learning and 1,800 matched VI episodes.
- Motion sampling passed for 30,000 total draws.
- Headless GUI resize/click, single-step, completion/timeout, and cancellation checks passed.
- Figures and computed comparisons are backed by the submitted CSV/JSON evidence.
- The PDF was rendered and visually checked, including all six figures; updated pages are checked after report changes.
- Final ZIP creation was tested with temporary details; the extracted package passed all five algorithm tests. No test identities or URLs were written into your real submission.

The verification record is `report/verification.txt`.

To rerun the focused tests after any code changes:

```bash
python -m unittest -v test_algorithms.py
```

If you change the algorithms, environment, or experiment settings, rerun
`value_iteration.py`, `q_learning.py`, and `report/analyze_results.py`, then update
the report's results and interpretations before rebuilding. The prose contains
the verified measurements for the current configuration; the PDF builder alone
does not rerun experiments or automatically rewrite the analysis.

The assignment deadline is **November 11, 2026, 11:59 PM ET**. Submit the final ZIP
through the course submission system yourself.
