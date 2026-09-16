# Agent Rules: Verification & Scope

**Context:** this project is a life-safety flash-flood/cloudburst nowcasting
pipeline. These rules exist because two specific failure modes have already
occurred during development and need to not recur silently:
1. "task reported complete" being treated as equivalent to "task result verified"
2. the agent reading or acting outside the intended project scope without being asked

## 1. No metrics claim without an attached raw artifact

Any time a training run, evaluation, or benchmark is reported, the report must
include, in the same message:

- The literal script or command that produced the numbers (path + invocation) —
  not a paraphrase of what it printed.
- The full raw stdout for that run (or the directly relevant excerpt).
- Raw confusion-matrix counts (TP / FP / FN / TN) for every reported CSI / POD / FAR,
  not derived ratios alone.
- Row count and positive-event count for each test split, plus the split's
  random seed / grouping key (e.g. `event_grouped_split(random_state=...)`),
  so two runs can be confirmed to share the same split before their metrics
  are compared ("up from X").
- The decision threshold (tau) used, and an explicit statement of what data
  it was selected on (validation — never the test set being scored).
- The exact checkpoint evaluated (path + modified time, or hash), so a
  "background run just finished" claim can't silently score a stale or
  unrelated checkpoint.

If any of the above is missing, the claim is **provisional** and must be
labeled as such — not presented as a finished result.

## 2. Background-task completion needs the same evidence as a foreground run

Regardless of how a completed background task's output reaches the agent
(log file, injected system message, or otherwise), the agent still produces
a synchronous, reproducible artifact — a script the user can re-run, or a
log excerpt — before reporting metrics from it. The delivery mechanism does
not change the evidence bar.

## 3. Filesystem scope

- The agent operates only within the project root and its subdirectories,
  unless explicitly given a specific, named external path for a specific task.
- The agent does not browse into other applications' data stores (chat app
  caches, browser profiles, other users' folders, OS-level app data, etc.)
  to "find more context," even if it suspects something relevant lives there.
- If the agent believes an external file is relevant, it asks first and
  states why — it does not copy the file in unprompted and then report on it.

## 4. Provenance of external data

Any file, report, or number that did not originate from this project's own
tracked pipeline (forwarded by a third party, pulled from a chat app, found
elsewhere on disk) must be labeled with its source and treated as
**unverified input** — never as corroborating evidence for model performance
— until its methodology can be independently checked.

## 5. Self-verification scripts get the same scrutiny as the thing they verify

A script written specifically to audit a result (e.g. `re_eval.py`) is
itself code, and can contain its own bugs — a mismatched checkpoint, a
leaky split, a metric computed on the wrong column. Its source gets read at
least once before its output is treated as ground truth, not just its
printed result.
