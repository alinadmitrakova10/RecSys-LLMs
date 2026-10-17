# Should we replace our MF with PPR?

## Proposed decision

I propose replacing our saved MF with PPR in the main personalized Top-10 row for warm users. This needs an online experiment before deployment. PPR's measured retrieval advantage motivates the proposal; a separate MF discovery section has no demonstrated benefit yet.

## What the offline evidence supports

On the same 861 eligible test users, macro Recall@10 was **0.07602291 for popularity, 0.07969324 for global PageRank, 0.11218853 for PPR, and 0.02560857 for MF**. Candidates and train+validation exclusions were identical. Global PR shared PPR's selected alpha; it was not independently optimized.

Our MF is the saved seeded retraining of HW5 (`40a20cc`), checkpoint epoch 1 chosen by validation RMSE. PPR used validation Recall@10. This compares specific models with different selection objectives, not MF and PPR in general. MF ranks unclipped scores; PPR scores are long-run graph-walk probabilities, not ratings or purchase probabilities.

User 1's PPR list includes **Fargo** (Crime/Drama/Thriller; training popularity 377); MF offers **To Catch a Thief** (Comedy/Romance/Thriller; 34). Train ratings connect both to familiar genres: **The Godfather 5/5**, **Pulp Fiction 4/5**, and **When Harry Met Sally... 5/5**.

For user 2, PPR's **Return of the Jedi** (Action/Adventure/Romance/Sci-Fi/War; 339) connects to **Star Wars 5/5** in train. MF's **Cinema Paradiso** (Comedy/Drama/Romance; 81) shares Drama/Romance with **Sense and Sensibility 5/5** and **Emma 5/5**. These are descriptive genre connections, not causal explanations of graph walks. All profile examples come from train.

Final mean list popularity for users 1/2 is **228.4/226.7 for PPR** and **104.2/68.3 for MF**. Popularity counts distinct train users rating a film at least 4. Lower popularity does not establish useful discovery. Each method's two lists share one film: Schindler's List (popularity), Star Trek: First Contact (global PR), Trainspotting (PPR), and The Wrong Trousers (MF). Different lists do not prove better quality.

## Separate controlled observations

Final alpha is **0.50**. The separate examples are **(user 1, 0.85), (user 2, 0.85), (user 1, 0.50)**. Changing the user at 0.85 gives 0/10 overlap and mean popularity 245.5 to 254.9. Lowering alpha for user 1 retains 7/10, with mean 245.5 to 228.4: Trainspotting, Sense and Sensibility, and E.T. replace Scream, Air Force One, and Titanic. This single contrast does not establish a general alpha trade-off.

Filtering partly explains user differences: user 2 already rated Fargo 5/5 and Contact 3/5 in train, so both are excluded regardless of rating. A different restart is not the only explanation.

## Limits and one online experiment

Evidence covers only warm users and train films. Chronology is per user, not globally consistent. Timestamps record ratings, not viewing. Offline Recall cannot establish current-session interest, completion, retention, or revenue.

Run one A/B experiment: control keeps the MF row; treatment replaces that row with PPR (alpha=0.50). Keep placement, other rows, and candidate/filter rules identical. Assign eligible users randomly once by stable authenticated user ID, and keep assignments across sessions/devices so nobody sees both variants.

The single primary metric is the **28-day recommendation-completion rate**: the proportion of all assigned users who start a film from the tested row and watch at least 90% of it within their first 28 days after assignment. Non-clickers remain in the denominator. The guardrail is p95 recommendation-serving latency against a limit agreed before testing. Rollout requires supported primary-metric improvement with acceptable latency. This expected benefit is a hypothesis, not a measured uplift.

## Existing evidence files

Numbers and examples were checked against `results.csv`, `recommendations.csv`, `recommendations-summary.csv`, `ppr-train-histories.csv`, `ppr-train-summary.csv`, and `ppr-settings-observations.md`. No experiment, model, or saved metric was rerun or changed.
