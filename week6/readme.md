# Graph-Based Recommendations with Personalized PageRank

Application: https://alinadmitrakova10.github.io/RecSys-LLMs/week6/

Source branch: https://github.com/alinadmitrakova10/RecSys-LLMs/tree/alina-06-personalized-pagerank/week6

The `gh-pages` publication branch serves the committed application and saved
results. Earlier homework paths are retained from the last published HW5 snapshot
(`40a20cc59e6ce017f59a33ec18b5b224af875ce2`). Reports, native AI-session exports,
lecture slides and local rendering files are excluded from Git and the site.

Course: LLM4Rec, HSE University
Instructor: Seungmin Jin
Task type: Individual assignment
Runtime: JavaScript and TensorFlow.js in a web browser

## Status and scope

This document defines the assignment. It does not claim that a new starter application, automated tests, or reference results on MovieLens have already been implemented.

Reuse the MovieLens 100K files and evaluation conventions from the previous matrix-factorization assignment. Do not overwrite the original `week5/` files. The directory and deadline for the new submission will be announced by the instructor.

GCN, GAT, temporal attention, and purchase-probability modeling are outside the required implementation.

## 1. Learning goals

- Construct a user-movie bipartite graph from positive training interactions.
- Explain row-normalized transitions, probability propagation, restart distributions, and stationary distributions.
- Implement PageRank and Personalized PageRank (PPR) with TensorFlow.js tensor operations.
- Turn movie-node scores into eligible Top-10 recommendations.
- Compare popularity, global PageRank, PPR, and your previous MF recommender under a shared retrieval protocol.
- Make an evidence-based deployment recommendation rather than assuming that a newer or more personalized algorithm is better.

## 2. Dataset and continuity with the previous assignment

Use `week5/u.data` and `week5/u.item`, and keep the previous per-user chronological 80%/10%/10% split. Do not overwrite the original `week5/` files. Treat missing ratings as unobserved, not as zero ratings.

Report the counts actually loaded by your application and all subsequent exclusions. The split is chronological within each user, not globally time-consistent; disclose that limitation and note that MovieLens records rating times, not viewing sessions.

## 3. Required implementation

### 3.1 Data preparation

Report users, movies, ratings, sparsity, positive training interactions, zero-degree users and movies, and the split record counts. The graph's node universe is the set of user IDs and movie IDs observed in the training split; map IDs to contiguous indices explicitly and never add a padding node to the graph or the recommendation pool.

```text
R[u, i] = 1 if user u rated movie i >= 4.0 in TRAINING
R[u, i] = 0 otherwise
```

A zero means no positive training edge; it is not a negative preference. Connect users and movies in both directions; there are no user-user or movie-movie edges.

### 3.2 Transition convention

```text
A = [[0, R],
     [R^T, 0]]
```

Normalize each nonzero row to obtain link-following probabilities. Notation: `alpha` is the probability of following a link, `1 - alpha` is the probability of restart, `v` is the normalized restart distribution, `p` is the column vector of node probabilities.

```text
p_next = alpha * P^T * p + (1 - alpha) * v
```

The transpose collects incoming probability for each destination; it does not reverse the graph's movement rules.

### 3.3 Zero-degree nodes

Do not divide by zero or silently discard probability mass. Implement the equivalent update without materializing dense dangling rows:

```text
p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
```

where `Q` has normalized rows for positive-degree nodes and zero rows otherwise, and `d` is the current mass on zero-degree nodes. Report zero-degree counts and the policy; it must be the same rule in global PageRank and PPR.

### 3.4 Global PageRank and PPR

Global PageRank: `v` is uniform over all graph nodes. PPR for user `u`: `v` is one-hot at that user's node. Initialize `p` to `v`. Stop when the L1 difference between consecutive distributions is below `1e-6`, or after 500 iterations. Report the iteration count, final residual, and whether convergence was reached; an iteration cap is not proof of convergence. Do not force a user's final score to equal the restart probability.

### 3.5 TensorFlow.js requirements

Use TensorFlow.js for the core batched propagation. JavaScript loops are acceptable for parsing, ID mapping, formatting, and small fixtures; they must not replace the main propagation with per-edge loops. You may avoid the full square adjacency matrix by implementing its two off-diagonal blocks with `R` and `R^T`, vector degree normalization, and `tf.matMul`. Await `tf.ready()`, report `tf.getBackend()`, manage tensors with `tf.tidy()` and dispose long-lived tensors, read results asynchronously where practical, never put an async callback inside `tf.tidy()`, and make sure repeated runs do not accumulate tensors. Record runtime and backend without comparing timings across devices. PPR is an iterative calculation, not neural-network training; no optimizer is required. Write code, comments and JSDoc in English and document inputs, outputs, indexing, assumptions and failure cases.

## 4. Verification before MovieLens experiments

### 4.1 Reference graph

Node order A, B, C, D; directed edges `A -> B`, `A -> C`, `B -> C`, `C -> A`, `D -> C`. A divides its link-following probability equally between B and C; B, C and D each have one outgoing link; there are no dangling nodes.

For alpha = 0.85 and uniform restart:

```text
p = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000]
ranking: C > A > B > D
```

For alpha = 0.85 and restart only at A:

```text
p = [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000]
ranking: A > C > B > D
```

These are numerical reference values for this specific test graph, not expected MovieLens outcomes.

### 4.2 Additional tests

Verify and report: normalized rows sum to 1; every restart distribution is nonnegative and sums to 1; iterations preserve total probability to numerical tolerance; stationary scores are nonnegative; one extra update produces approximately the same distribution; a separate zero-degree-node fixture preserves mass; graph direction and matrix orientation match; recommendation filtering removes the correct movies. Do not claim to have run checks you did not run.

## 5. From scores to Top-10 recommendations

Use all movies appearing in the training split as the candidate pool. For final TEST recommendations: extract movie-node scores only; exclude all movies the user rated in training or validation regardless of value; sort descending by score with ties broken by ascending movie ID; return up to 10; display IDs, titles, genres and scores. Report if fewer than 10 eligible candidates exist, and whether zero-score movies had to be included; do not claim that tie-broken zero-score candidates have demonstrated relevance. For VALIDATION recommendations exclude training-rated movies only. PPR scores are long-run graph-walk probabilities: not purchase probabilities, not predicted ratings, and not comparable in scale to MF scores.

## 6. Methods and experimental controls

Compare four methods: (1) Popularity — count distinct users with a positive training rating per movie; (2) Global PageRank with uniform restart; (3) PPR with restart at the target user; (4) MF — reuse your previous implementation and selected model. Do not alter the MF scoring definition to resemble PPR; state the model and checkpoint used.

### 6.1 Alpha selection

For PPR compare `alpha` in `{0.50, 0.70, 0.85, 0.95}` on validation Recall@10 and select one shared alpha for the full evaluation population; do not use test outcomes. Break exact ties by the smaller alpha. Use the same alpha for the primary global PageRank comparison and disclose this control; this is not a claim that global PageRank was independently optimized. MF keeps the previous validation-RMSE checkpoint rule; disclose that MF and PPR use different validation objectives.

### 6.2 One-factor-at-a-time demonstration

Choose two users from the common test cohort and keep their IDs fixed. Show three PPR settings: baseline (user A, alpha = 0.85), change seed only (user B, alpha = 0.85), change alpha only (user A, alpha = 0.50). Compare both changed settings with the baseline and never attribute a difference to one factor after changing both. Do not assume larger alpha always improves novelty, or that the restart seed must be the highest-scoring node.

## 7. Retrieval evaluation and popularity evidence

### 7.1 Shared evaluation cohort

Use the same candidate rules and the same users for all four methods. A user is evaluable when the user appears in training with at least one positive training edge, the selected MF implementation can score the required training-movie candidates, and at least one relevant validation/test movie remains eligible under the appropriate filtering rules. Apply the shared intersection for the primary comparison, disclose any additional restriction, and report how many users are excluded. This is a warm-user evaluation; it does not establish cold-start performance.

### 7.2 Recall@10

A relevant movie has rating >= 4.0 in the evaluation split and must belong to the training-movie candidate pool and pass that split's previously-rated exclusion rule.

```text
Recall@10 = relevant eligible movies retrieved / all relevant eligible movies
```

Report macro-average Recall@10 over the shared cohort, the cohort size, and exclusion counts; separately count relevant movies absent from the training candidate pool and users without eligible relevant movies. Do not compute rating RMSE for PPR.

### 7.3 Recommendation popularity

```text
popularity[i] = number of distinct users with training rating >= 4.0 for movie i
```

For each reported Top-10 list include each movie's popularity and the list's mean popularity, and for the two selected users report the number of common movies in their Top-10 lists within each method (descriptive evidence, not proof of better personalization). Do not treat unknown test movies as dislikes or popularity as platform-wide impressions.

## 8. Business & Algorithmic Analysis - exactly ONE question

### Should the platform keep MF, replace it with PPR, or use both?

Recommend one deployment strategy — keep MF as the main recommender, replace it with PPR, or use the two methods in different recommendation sections — supported by Recall@10 on the common cohort, the two users' Top-10 lists with titles and genres, their training histories, the training-set popularity of the recommended movies, and the observed effects of changing the restart user and alpha separately.

Do not assume a different list is better personalization, that less popular movies are automatically better discoveries, or that PPR scores are purchase probabilities. If you propose separate sections, name each section's purpose and explain why your results support the assignment. State what the offline results cannot establish about current-session intent, viewing completion, retention or revenue, and propose one online experiment before rollout specifying the intervention, primary outcome, at least one guardrail, and how user-level assignment avoids contamination. No extra business essay is required.

## 9. Deliverables

Submit: the JavaScript application and its source files with TensorFlow.js used in the core propagation; `README.md` with exact run instructions and dependency versions actually used; `verification.json` or an equivalent machine-readable test-result export; `results.csv` with retrieval metrics, common-cohort size, exclusions and selected settings; `recommendations.csv` with user ID, method, alpha where applicable, rank, movie ID, title, genres, score and training popularity; a course-format `report.pdf` covering implementation evidence and the single business question; the unmodified AI-session export required by the course, if that requirement remains in force. Preserve the original dataset and previous assignment files, and state any incomplete component, failed test, nonconverged run or unavailable prior MF output.

Confirm the current grading guide, submission directory, deadline and AI-export requirement with the instructor before publishing the final assignment. Do not invent an official numerical rubric or a deadline.

## 10. Evidence and citation integrity

Cite the original dataset and verify all paper titles, authors, years and URLs. Do not fabricate runtime measurements, experiment results, tests or online business effects. Make sure tables match the application's actual exports, and separate measured observations, proposed explanations, and proposed deployment actions.

Dataset provenance: MovieLens 100K, GroupLens, University of Minnesota. The previous assignment provides the citation: F. Maxwell Harper and Joseph A. Konstan, "The MovieLens Datasets: History and Context," ACM Transactions on Interactive Intelligent Systems, 2015. Verify the bibliographic record before including it in the submitted report.

## 11. Final checklist

- [ ] Same MovieLens files and split as the previous assignment.
- [ ] Graph built from positive training ratings only.
- [ ] All observed training/validation ratings used for final exclusion.
- [ ] Alpha consistently means link-following probability.
- [ ] Matrix orientation and dangling-mass handling tested.
- [ ] No unintended padding node or dataset mutation.
- [ ] TensorFlow.js propagation and memory management implemented.
- [ ] Shared evaluation cohort and candidates across methods.
- [ ] Validation used for selection; test used only for final evaluation.
- [ ] Popularity calculated from training data only.
- [ ] One setting changed per demonstration.
- [ ] PPR scores not presented as ratings or purchase probabilities.
- [ ] One evidence-based business decision, with limitations and an online experiment.
- [ ] All missing work and verification failures disclosed.

---

## Local run instructions (added for this starter)

TensorFlow.js is loaded from the CDN in `index.html`; d3 is vendored locally.

Serve the folder over HTTP (the data layer uses `fetch`, so `file://` will not work):

    cd week6
    python3 -m http.server 8000

Then open `http://localhost:8000/index.html`.

## Local HW5 MF continuity and persistence

The MF method is derived from `alina-05-matrix-factorization` commit
`40a20cc59e6ce017f59a33ec18b5b224af875ce2`, `week5/script.js` (Git blob
`2d9bfb017b38bebcf7420b8e890709a3c2c358af`). The neighboring `week5` worktree
is not its source. Original-run weights were unavailable: this is a **new seeded
retraining of the previous method**, not recovery of the historical result.

- 16 factors, `trainMean + dot(user, item)`, Adam, 20 epochs, learning rate 0.01,
  regularization 0.05, batch size 64.
- Loss uses the mean squared values of the gathered batch vectors, not the norm
  sum of all factors. Initialization uses `tf.randomNormal(0, 0.05)` with user seed
  20261013 and item seed 20261014. Fisher-Yates uses mulberry32 seed 20261015 and
  resets the ordered indices each epoch. TF.js is 4.22.0; runtime backend is recorded.
- RMSE predictions are clipped to [1,5]; ranking scores are **not clipped**.
  Checkpoint selection/restoration uses validation RMSE, exact ties selecting the
  earliest epoch. Test RMSE is measured only after restoration; it is not used
  for selection. Rating RMSE retains all test records, including movies absent
  from train; the future retrieval scorer uses only the 1611 train movies.
- On first MF use, the application tries memory, then IndexedDB, then the local
  `mf-checkpoint.json` artifact. Only absence of both saved sources permits a
  fresh run. Concurrent callers share preparation; a failed/incompatible load
  raises an error and does not silently retrain. Rebuilding data indices
  invalidates an in-memory model.
- Checkpoints include settings, source identity, selected epoch, complete epoch
  history, backend/version, data and split hashes, factor dimensions, checksum
  and raw score probes. A load verifies compatibility, all saved probe scores
  (absolute tolerance 1e-5 for float32 backend differences) and validation RMSE.
  IndexedDB is origin-specific; the JSON artifact also works with a fresh browser
  profile or a different local-server port.

Verification uses Node and the existing Playwright dependency (or set
`PLAYWRIGHT_MODULE` to an installed Playwright package). Source parity is checked
against the Git object itself on small examples. No Recall@10 or alpha search is run.

```powershell
node week6/check-mf.cjs --fixtures-only
node week6/check-mf.cjs --train-once
node week6/check-mf.cjs
```

The training command creates `week6/mf-checkpoint.json` and `week6/mf-run.json`
only if no checkpoint artifact exists. Re-running it reuses the artifact and
does not overwrite the experiment. The default command requires existing weights.
`mf-run.json` contains measured metrics and persistence checks; old report numbers
and the old HW5 Recall@10 are never used as current results.

## Retrieval protocol and the earlier fixed-alpha preview

`movieScores`, asynchronous `recommendTopK` and asynchronous `recallAtK` now use
the shared implementation in `retrieval.js`. Candidates are train movies only;
validation excludes train-rated movies, and test excludes train+validation-rated
movies at every rating value. Ranking uses score descending/movieId ascending.
Zero graph probability is valid. Missing, nonnumeric or non-finite eligible
scores are explicit failures, not zeros or silent ranking exclusions.

The common cohort is verified against actual MF scorer output for the whole
train-movie pool. Reports include each exclusion reason, user IDs, the unique
union and overlaps. Relevant records outside the candidate pool and distinct
unseen movie IDs are reported separately. Recall is the unweighted mean of
per-user recalls. Provisional selectors are marked as pending MF eligibility
until the saved model has been checked; they are not an evaluation cohort.

**This stage never retrains MF.** The application loads the selected checkpoint
from IndexedDB or `mf-checkpoint.json`; missing/incompatible weights are an
error. The earlier `--train-once` instructions describe how the existing artifact
was produced, not permission to repeat training during retrieval preparation.
`mf.js`, the checkpoint and the previous MF run record remain unchanged.

The final TEST action now uses the saved/one-pass comparison described at the
end of this document. Validation selection is loaded, not repeated, for that
comparison. The earlier eight-list alpha=0.85 preview remains saved separately.
The two-user example button now displays exactly the three controlled PPR
settings and train-only profiles described at the end of this document.

```powershell
node week6/check-retrieval.cjs --fixtures-only
node week6/check-retrieval.cjs
```

The second command checks the browser, saved MF loading (zero training runs),
known-answer fixtures and independent filtering/ranking, then writes:

- `recommendations-alpha085.json`: eight complete lists, score meanings, mean
  train popularity, overlaps, cohort exclusions and MF checkpoint identity.
- `recommendations-alpha085.csv`: 80 recommendation rows with IDs, titles,
  genres, scores, training popularity, split and list mean popularity.
- `recommendations-summary-alpha085.csv`: per-list mean popularity/zero counts
  and common-movie counts for the two users within each method.
- `retrieval-verification.json`: actual test results and checks that full
  MovieLens Recall, alpha selection and MF training were not run.

In-page self-checks use isolated known-answer fixtures, including rank 10/11,
macro 5/6 versus pooled 3/4, overlapping exclusions and invalid-versus-zero
scores. An empty recommendation implementation fails these fixtures. All errors
are FAIL; no implementation failure is converted to PENDING.

## Completed validation-only alpha selection

The same 846-user validation intersection and 1611 train-movie candidates were
used at every alpha. Validation excludes every train-rated movie, regardless of
rating. Popularity and graph construction still use train only. MF is loaded
from the unchanged saved checkpoint, with **zero training runs**. This
validation-only stage itself did not run test retrieval or produce a report;
the subsequent test evaluation is recorded separately below.

Measured macro validation Recall@10:

| Alpha | Recall@10 | Accepted users | Alpha processing time (seconds) |
|---|---:|---:|---:|
| **0.50** | **0.11448315913208877** | 846/846 | 71.2365 |
| 0.70 | 0.10479695076410430 | 846/846 | 157.8238 |
| 0.85 | 0.09002077155336492 | 846/846 | 341.9312 |
| 0.95 | 0.07375030387909676 | 846/846 | 2436.1233 |

The selected value is **0.50**. Exact ties select the smaller alpha. It is stored
in `HW6_ALPHA_SELECTION` and `alpha-validation.json`, independently of the example
slider. The later primary PR/PPR comparison calls `comparisonAlpha()`, which
requires a complete compatible grid; example buttons and the old alpha=0.85
preview remain separate. Refreshing the page validates and displays the saved
grid rather than rerunning it. Failed/incomplete grids cannot announce a winner.

### Timing, batching and strict convergence

- Scalar PPR at alpha=0.85 took 1.64–2.03 seconds on the three initial users.
  `batch-ppr.js` therefore reuses the two normalized TensorFlow.js blocks with
  batches of 16 users. Rows are user distributions, so `p_row * Q` is exactly the
  transpose of the README column-vector convention `Q^T * p_column`.
- Every row starts at its own one-hot v. Its dangling mass returns to that same
  v. Each row freezes at its own first measured L1 difference below 1e-6; the cap
  is 500 updates per attempt. TF.js 4.22.0/WebGL was used for the initial grid.
- Batch and ordinary PPR were compared at all four alphas on small fixtures and
  real users 1, 2 and 943. Live score differences were below 5e-9 and Top-10 IDs
  matched. A small float32 plateau fixture is deliberately a negative test:
  capped results are rejected, not called converged.
- The initial alpha=0.95 batch pass had 27 capped users. One ordinary WebGL PPR
  retry per capped user recovered 24. The three remaining users were **54, 310,
  595**. Before a CPU retry, user 1 qualified CPU against WebGL at alpha=0.95:
  identical Top-10 and maximum score difference 6.9849193e-9, both converged.
  Only those three remaining users were then run on CPU, converging in 284,
  284 and 285 updates. The formula, start, tolerance, cap and user population
  were not changed. The complete grid was **not rerun**.
- All 30 failed WebGL attempts (27 batch + 3 ordinary) remain in the per-user
  attempt history with backend, iterations and residual. Accepted results at
  alpha=0.95 comprise 819 batch/WebGL, 24 ordinary/WebGL and 3 ordinary/CPU runs.
  No failed result contributed hits or a zero Recall to the aggregate.
- Total measured selection/recovery processing time was **3019.6011 seconds**
  (about 50 min 20 s), including retries and CPU qualification, excluding idle
  time. Initial scalar/batch qualification is timed separately. These are local
  timings, not cross-device performance comparisons. Tensor counts were 2 before
  and after the batch and CPU checks: only the two saved MF factor variables.

Validation exclusions are 97 unique users: 1 without positive train edges and
97 without eligible relevant validation movies, with 1 overlapping user. No MF
coverage exclusion occurred. Four relevant validation user-movie records were
outside the train pool. Reason counts are not summed as separate people.

### Artifacts and verification without repeating the expensive run

- `alpha-validation.json`: complete grid, selected value, exact cohort and
  identities, exclusion reasons, timings, solver metadata and all per-user
  evidence. Each user has hits, relevant count/IDs, recommended IDs, accepted
  backend, iterations, residual and all retry attempts where applicable.
- `alpha-validation.csv`: four summary rows with Recall, exclusions, timing,
  convergence, mixed-backend provenance and failed-attempt counts.
- `alpha-validation-users.csv`: 3384 rows, allowing independent macro Recall
  and intersection checks without PPR. Backend and complete attempt histories
  are exported too.
- `alpha-benchmark.json`, `alpha-batch-verification.json` and
  `alpha-cpu-verification.json`: measured scalar speed, equivalence/resource
  checks, and the CPU control/recovery evidence.
- `alpha-result-verification.json`: saved-result integrity, recomputed macro
  Recalls, incomplete-grid rejection, slider independence and reload checks.
- `alpha-validation-progress.json`: historical initial-pass progress/failure
  evidence, not the authoritative final result and never a source of a winner.

```powershell
# Fast integrity/UI checks; no grid or PPR recomputation.
node week6/check-alpha.cjs --verify-saved
# Existing regression suites, preserving prior preview/model artifacts.
node week6/check-data-graph.cjs
node week6/check-pagerank.cjs
node week6/check-retrieval.cjs --verify-only
node week6/check-mf.cjs --load-only
```

`--benchmark-only` and `--checks-only` measure scalar speed or repeat numerical
qualification. `--run-validation` explicitly starts a new expensive grid.
`--retry-capped` and `--retry-cpu` document the sparse recovery steps used for
this run; neither is required to verify the saved complete result. The loader
rechecks data/MF/cohort identity, per-user hits and convergence, all macro means
and the winner. Saved MF compatibility includes its original split fingerprints;
test relevance or test retrieval outcomes are never used to choose alpha.

## Final TEST comparison (separate from validation selection)

`test-comparison.js` evaluates the same **861 users** for popularity, uniform
global PageRank, user-seeded PPR and the saved HW5 MF. The 1611 train movies are
the only candidates; every train/validation-rated movie is excluded independently
of rating. Eligible test ratings >=4 define relevance. The metric is the
unweighted mean of each user's Recall@10, not a pooled hits/relevant fraction.

PR and PPR use **alpha=0.50 from the verified `alpha-validation.json`** through
`comparisonAlpha()`, never the example slider. This is a shared comparison
parameter for global PR, which was **not independently tuned**. MF keeps its
validation-RMSE-selected checkpoint 1, from the previously saved seeded HW5 run.
Its retrieval Recall below is newly measured under this protocol; no historical
HW5 Recall was copied or substituted.

| Method | Macro test Recall@10 | Users | Measured method time (s) |
|---|---:|---:|---:|
| Popularity | 0.07602291343770759 | 861 | 0.4309 |
| Global PageRank | 0.07969324149418144 | 861 | 1.3494 |
| PPR | 0.11218853219486066 | 861 | 79.3690 |
| Saved HW5 MF | 0.02560857416094039 | 861 | 2.2183 |

The one-pass run took **83.4506 seconds**, including preparation, ranking and
statistics, before separate saved-result verification. Popularity uses JavaScript;
PR/PPR/MF used TensorFlow.js 4.22.0/WebGL on this machine. Global PR was calculated
once (19 updates, residual 5.43586793e-7). PPR reused one block-transition engine
for **54 batches** of at most 16 users; all 861 users converged in 21–22 updates,
with maximum residual 9.99436907e-7. No nonconverged user was omitted or assigned
zero Recall. Only the two MF variables remained after propagation (2 -> 2 tensors).

Exactly **3444 lists** were ranked, one per method/user. Hits, Recall and mean
training popularity are computed from each list together. Aggregated popularity
and the final two-user examples reuse these stored lists rather than running PPR
or ranking again. Mean popularity is the macro mean of per-list means.

The shared eligibility check excludes **82 unique users**: 1 without positive
train edges and 82 without eligible relevant test movies, with one overlapping
user; no user is excluded for MF score coverage. There are 13 relevant test
user-movie records outside the train pool. Full reason sets, excluded user IDs
and overlaps are in the JSON. These reason counts are not additive.

### Final artifacts and page behavior

- `test-results.json`: settings, data/split/cohort hashes, alpha source/hash,
  MF source/run/checkpoint/settings/hash, runtimes, convergence and all four
  methods' per-user recommended IDs/scores/popularities, eligible relevant IDs,
  hits, Recall and list means. It also holds the complete final Top-10 metadata
  for users 1 and 2, plus within-method overlap counts.
- `results.csv`: four method metrics with common-cohort size, overlapping
  exclusions, saved alpha source, backend, time, convergence and MF provenance.
- `recommendations.csv`: 80 final recommendation rows with titles, genres,
  scores, training popularity, list mean popularity and two-user overlap.
- `recommendations-summary.csv`: eight per-list means, zero-score counts and
  the common movie IDs/counts for users 1 and 2.
- `test-verification.json`: actual fast-verification, corruption-rejection and
  reload/instrumentation results.

The page automatically loads and validates the saved test result after data and
the saved alpha have been checked. Its final table and final user lists are
separate from the interactive slider/examples. The explicit TEST button reuses
the saved result; it does not recalculate it. The default results/recommendations
exports now use the final test artifacts. The alpha=0.85 example files remain
unchanged and separate (`recommendations-alpha085.*` and its summary).

Fast verification rechecks identical groups, settings and hashes, eligible
relevant IDs, train+validation filters, ID uniqueness, stored score/ID ordering,
hits, all user/macro means, training popularity, convergence and user 1/2
titles/genres/overlaps. It uses stored results, not a new graph propagation or
new ranked lists. Deliberately corrupted duplicates, rated movies, ordering,
cohorts, alpha, MF checkpoint, macro Recall and convergence are rejected.

```powershell
# Default/fast mode: no full TEST, alpha search, PPR propagation or MF training.
node week6/check-test.cjs --verify-saved
# The original explicit one-time calculation; refuses if test-results.json exists.
node week6/check-test.cjs --run-test
# Focused old regressions; preserve the prior artifacts.
node week6/check-data-graph.cjs
node week6/check-pagerank.cjs
node week6/check-retrieval.cjs --verify-only
```

Browser verification instruments the actual functions: the initial test action
used one global PR and one PPR engine/54 batches, at 0.50 even with the example
slider set to 0.95. After a real refresh, any new MovieLens propagation would
throw; restoring and showing the result passed with zero calls. Data, previous
homework files, saved MF and every alpha-selection JSON/CSV remain unchanged.
Business conclusions and the report remain outside this step. The next section
records the separate controlled PPR demonstration without repeating evaluation.

## Controlled PPR examples and train-only profiles

The page's controlled-settings section and `ppr-settings-observations.md` present
exactly three PPR examples from README section 6.2:

| Setting | User | Alpha | Mean train popularity | Common with baseline | Zero scores |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 0.85 | 245.5 | reference | 0 |
| change-user | 2 | 0.85 | 254.9 | 0/10 | 0 |
| change-alpha | 1 | 0.50 | 228.4 | 7/10 | 0 |

Only the two changed settings are compared with the baseline. There is no
comparison between user 2/0.85 and user 1/0.50 that would change both factors.
The train graph, train candidate universe and test exclusion rule are unchanged.
Changing the target user also changes that user's rated-item exclusion set,
under the same rule; this is noted explicitly rather than attributing every list
difference to personalization. Final selected alpha/test metrics remain separate.

The 0.85 lists come from the verified `recommendations-alpha085.json`; user 1 at
0.50 comes from the final cached `test-results.json`. No new MovieLens propagation
is needed. Source hashes, current train/candidate identity and every saved ID,
title, genre, score, filter and popularity are verified. The example slider does
not control these three fixed settings.

User 1's profile uses **217 train ratings, 131 positive**; user 2's uses **49 and
32**. Only train records contribute to the summaries and concrete rated-title
examples. Each distinct positive movieId contributes one count to each of its
genres, including unknown, so genre shares use 131/32 positive films and can sum
above 100%. Multi-genre assignment totals are 280 and 62. Different movie IDs
with the same title are not merged. Genres describe these histories; they are
not features in the PPR rating-edge graph.

Artifacts owned by this demonstration:

- `ppr-settings-examples.json`: exact settings, 30 full recommendation records,
  two baseline comparisons, train summaries, observations and source/history hashes.
- `ppr-settings-examples.csv`: IDs, titles, genres, walk probabilities, popularity,
  means and zero counts for the three Top-10 lists.
- `ppr-settings-comparisons.csv`: common, appeared and disappeared IDs, list means
  and zero counts for the two baseline comparisons.
- `ppr-train-histories.csv`: **266 complete train rows**, with rating, timestamp,
  metadata, positive flag and an explicit train split label.
- `ppr-train-summary.csv`, `ppr-train-genres.csv`: two user summaries and all
  positive-film genre counts/shares with their denominators.
- `ppr-settings-observations.md`: short English observations, concrete train-title
  links, deltas and limitations. Different lists are not proof of better quality;
  lower popularity alone is not a useful-discovery result, and zero probability
  is not evidence of relevance. No business decision is made here.
- `ppr-settings-verification.json`: actual saved-source, filter, three-setting,
  comparison, mean, zero-score, train-origin, reload and regression checks.

```powershell
node week6/check-ppr-profiles.cjs --build
node week6/check-ppr-profiles.cjs --verify-saved
node week6/check-data-graph.cjs
node week6/check-retrieval.cjs --fixtures-only
```

The build command only combines checked caches and train records; verification
does not regenerate previous results. Browser instrumentation forbids real
propagation, training, alpha selection and test evaluation during this check.
The page restores the saved three settings, observations and summaries after
refresh; the existing final table/lists also stay intact. Nine corrupted-example
cases are rejected. Known-answer retrieval tests explicitly specify their own
reference alpha so moving the example slider cannot change the fixtures.
