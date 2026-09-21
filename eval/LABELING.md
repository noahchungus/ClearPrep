# Labeling guide for the evaluation sets

Each sample answer is a short behavioral-interview answer written by the project author (NOT collected from real
candidates). Each is labeled by *content*, before the analyzer ever sees it:

* **situation**: the answer says when/where/what was going on (context that sets up a real story).
* **task**: it says what the speaker was responsible for or trying to achieve.
* **action**: it describes concrete steps the speaker took (alone or with a group they were part of).
* **result**: it states an outcome of the actions (a number, a change, feedback, or what happened at the end). A bare
  "it went well" with nothing else counts as a result only if it clearly refers to the outcome of the story.
* **quality**: `good` (a well-structured answer an interviewer would be happy with), `mid` (usable but with a clear
  gap), `weak` (vague, generic or missing most structure).

Labels were assigned while writing, using the definitions above, not by looking at analyzer output. The rules were tuned only
against `dev.json`; `test.json` is held out and was scored once after the rules were frozen (the numbers are on the app's "How scoring works" page).
Limits: one labeler (the author), small sets, and answers written to be realistic rather than sampled from real people.
