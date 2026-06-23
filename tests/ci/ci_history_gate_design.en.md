# CI Metric History & Gate — Design

Draft, pre-implementation. Decisions only.

## What this is

Each training test produces a few numbers per run (e.g. gradient norm, a KL value). Today CI checks them against fixed thresholds someone hand-picked, or compares two runs inside the same test. We never compare a number against how that same test behaved on earlier commits.

This proposes: **save those numbers from every CI run, and use the history to catch regressions** — and to point at the PR that changed a number.

## 1. Why

Hand-picked thresholds (like "logprob diff <= 0.03") are rough: 

- Cannot detect subtle regression: if a bad pr make logp diff stably grows from 0.007 to 0.009, it's hard to detect the bug by hard gate (we might consider it as sanity check and increase the threshold).
- Hard to maintain: Each time when we add a new e2e test, we need to manually set the gate value.
- Hard to track: Cannot see recent trend of a metrics.
- The numbers are already sent to wandb every run, but nothing keeps them in a form we can compare or clean.

**Why not just read history from wandb?** wandb is a logging dashboard, not something we can treat as a clean baseline:

- No notion of "these past runs are the good ones to compare against."
- Removing a bad data point means deleting whole runs — clumsy and unauditable.
- Easy to hit wandb API access limit.

So: **wandb keeps receiving the numbers (unchanged), but the history we check against lives in our own store, and we never read back from wandb.**

## 2. Constraints

- The store is in the cloud — CI runs on many machines, so a local disk can't be shared.
- Each test is identified by its file path.
- Each saved row carries the commit, workflow and PR it came from.
- We can delete a wrong data point (by a workflow), and the next check uses the corrected history right away.
- Tolerate normal run-to-run noise (default: a number may move 20% before it's flagged).

Not this round: uploading arbitrary extra data; replacing the existing same-run checks; building a dashboard.

## 3. Data flow

```
during a test run:                  after the test passes:               our store (cloud DB)
  the run produces numbers
    --> already sent to wandb  ------------------------------------->  wandb   (we only write)
    --> also saved to a local file
                                     a check reads the local file,
                                     tags it with test / commit / PR,
                                     reads this test's past numbers  <-->  history
                                     compares (see section 5),
                                     saves this run's number         -->  new row
                                     fails CI if the comparison fails

  anyone, anytime  ------------------------------------------------->  see the trend / delete a bad point
```

- **When are the numbers collected?** While the test runs — they're produced during training, and we save one number per metric (e.g. its final value) when the run ends. Nothing is fetched afterward.
- **When does history fail CI?** Only after the test itself passes, and only once a test has at least 5 past good runs to compare against. Below 5, history isn't used yet (only the fixed limit in section 5 applies).
- **Which tests will be covered by historical metics check?** Short runs must be covered. Long regression tests still use wandb to track.

## 4. Where it's stored: Neon

Neon is a managed cloud Postgres database.

- Our data is tiny, so it fits Neon's free tier ($0), at most ($15/month)
- Being a database, we can average and query it easily, browse/edit it in a web UI, and delete a bad point with one update — a plain file store can't do this well.
- CI has no cloud database today, so something must be set up either way; Neon needs just one project and one password (stored as a CI secret).

One table: each row is one run's number for one (test, metric), plus commit, PR, and whether it counts as a "good run." Reading and cleaning are ordinary SQL.

## 5. The check: two layers

For each number we check two ways:

- **Fixed limit:** a hand-set "this should never happen" value. Catches gross breakage even for a brand-new test with no history. Given through `--ci-<metrics-name>` , but not always provided.
- **History check (needs ≥5 past good runs):** compares this run against the average of past good runs, and flags it if it moved more than 20%. This is the part that catches small regressions and gives the trend.

Notes:

- The 20% is a percentage by default. For numbers normally near zero (where a percentage is meaningless), we allow a small fixed wiggle instead.
- A run joins the "good runs" history only if it passed both checks — so a drifting run can't quietly drag the baseline along. Accepting a new normal is a manual step.

First metrics to check: train rollout logp abs diff, grad norm, raw rewards, and kl values on each steps.

## Open questions

- Whether a new test's first few baselines need a human to confirm? 
  - Prefer not in the 1st version.
- Who is allowed to delete data points? Also the CI dashboard? 
  - Start using it internally

