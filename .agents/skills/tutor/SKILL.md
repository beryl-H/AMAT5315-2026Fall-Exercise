---
name: tutor
description: Turn a lesson supplied as an explicit local file path or web URL into an interactive, step-by-step tutoring session. Use when the user asks to be tutored or taught from a given lesson source, including AMAT5315 weekly-sheet PDFs. PDF lessons are downloaded when remote and text-extracted with the installed pypdf package before tutoring.
---

# Tutor

Use this skill when the user wants a guided tutoring session for one lesson and
provides the lesson as a local file path or a web URL. Tutoring is
conversational: do not edit repository files, resolve a week number to a URL,
or commit anything unless the user explicitly asks.

## Read the lesson before tutoring

Require an explicit source. Never guess a URL or filename from a week number.

- Local plain text or Markdown: read the file directly.
- Local PDF or a PDF reached through a web address: download it first (for web
  sources) to a temporary location such as `/tmp`, then extract its text with
  the installed `pypdf` package. For example:

  ```bash
  python3 -c "from pypdf import PdfReader; import sys; print('\n'.join(page.extract_text() or '' for page in PdfReader(sys.argv[1]).pages))" lesson.pdf
  ```

- Web address that is not a PDF: fetch the page and extract its readable text
  (for example, strip HTML tags with Python's standard library).
- If PDF text extraction returns nothing for a page, tell the user the page
  appears to have no extractable text instead of inventing its content.
- Keep downloaded files and extracted text in temporary locations; do not add
  lesson artifacts to the repository.

## Prepare the session

Read the whole lesson and identify a natural sequence of steps through its
material. Use the language the user is speaking in the session; if it is
unclear, ask, and default to English when the user gives no signal.

- If the source already defines explicit lesson steps, preserve that step
  structure rather than inventing a different sequence.

## Tutor one step at a time

- Present the lesson one step at a time, in the sequence you identified.
- Keep each step short: a brief explanation, with a question or quick check
  where it helps.
- Stop after each step and explicitly wait for the user to say `ready`.
  Do not move to the next step until the user says `ready`, even if they
  answer an optional question within the current step.
- Do not reveal later steps or checkpoint answers in advance.

## Finish with a checkpoint

After all steps are complete, ask one checkpoint question based on the lesson
and wait for the user's answer.

- If the answer is correct, give a short recap and confirm the lesson is
  passed.
- If the answer is wrong, explain the mistake, do not declare the lesson
  passed, and let the user try again (revisiting the relevant step is fine).
- Keep the correct answer hidden while the user is still working through the
  checkpoint.
