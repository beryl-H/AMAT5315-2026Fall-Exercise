## Purpose

This repository stores my weekly AMAT5315: Scientific Computing for Physicists
exercises and their Git history. Week 1 work goes in `week1/`, Week 2 in
`week2/`, Week 3 in `week3/`, and so on. As a student in the course, I work
through each week's exercise here by following the weekly learning sheet and
adding a solution plus tests.

The repository is public, so treat everything in it as publishable. Never add
passwords, API keys, access tokens, identity numbers, or private notes.

## Layout

- `weekN/` — one folder per exercise week (`week1/`, `week2/`, ...)
- `weekN/SPEC.md` — the task description for that week
- `weekN/` source and test files — the implementation and tests required by that week's learning sheet
- `setup.txt` — the tool versions recorded when I set up this environment

For example, `week1/` contains `SPEC.md`, `pi.py`, and `test_pi.py` for a
Monte Carlo estimate of pi.

## Working here

- Follow the weekly learning sheet for the week being worked on, together with
  that week's specification and existing files.
- Define correctness before implementation: establish what correct means from
  the learning sheet and specification before writing or changing the
  implementation.
- Derive tests from the specification, not from the implementation.
- Run verification independently using the commands required by that week's
  learning sheet. Do not weaken correctness criteria or tolerances merely to
  make a test pass.
- Use negative controls when required: deliberately introduce a known defect
  and confirm that the relevant test or check fails.
- Preserve meaningful Git history: keep the required order such as spec,
  failing test, implementation, sabotage, and fix.
- Put files in the locations required by the weekly learning sheet. Do not
  modify unrelated weeks or existing specifications unless the task requires it.
- Prefer small, readable implementations that follow the language,
  conventions, and structure required by the current week's learning sheet.
- Commit when I or the weekly learning sheet explicitly requires it, and
  preserve the required commit order.

## Repository marker

Memory probe: W1-MEMORY-5315
