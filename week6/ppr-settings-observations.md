# PPR settings and train-only profiles

These are fixed examples, separate from validation alpha selection and final test metrics.

| Setting | User | Alpha | Mean train popularity | Zero scores |
|---|---:|---:|---:|---:|
| baseline | 1 | 0.85 | 245.5 | 0 |
| change-user | 2 | 0.85 | 254.9 | 0 |
| change-alpha | 1 | 0.50 | 228.4 | 0 |

## Train-only summaries

Each distinct movieId with a train rating >=4 contributes once to every listed genre, including unknown. Multi-genre films therefore contribute to several counts; shares use positive-film count and need not sum to 100%. Different movie IDs are kept even when titles coincide.

- **User 1:** 217 train ratings; 131 positive ratings. Drama: 61; Comedy: 39; Action: 36; Sci-Fi: 27; Thriller: 25; Romance: 22; Adventure: 15; War: 13; Crime: 11; Horror: 6; Children's: 5; Musical: 5; Animation: 4; Documentary: 3; Mystery: 3; Western: 2; Fantasy: 1; Film-Noir: 1; unknown: 1.
  Examples: Return of the Jedi (1983) (#181), 5/5; Godfather, The (1972) (#127), 5/5; Postino, Il (1994) (#14), 5/5; Back to the Future (1985) (#204), 5/5; Pulp Fiction (1994) (#56), 4/5; When Harry Met Sally... (1989) (#216), 5/5.
- **User 2:** 49 train ratings; 32 positive ratings. Drama: 18; Comedy: 11; Romance: 11; Action: 5; Crime: 5; Thriller: 3; Adventure: 2; Sci-Fi: 2; War: 2; Animation: 1; Children's: 1; Film-Noir: 1.
  Examples: Star Wars (1977) (#50), 5/5; Fargo (1996) (#100), 5/5; Godfather, The (1972) (#127), 5/5; Contact (1997) (#258), 3/5; Sense and Sensibility (1995) (#275), 5/5; Emma (1996) (#283), 5/5.

## Two comparisons with the same baseline

### change-user
- Common: 0/10 — none.
- Appeared: Return of the Jedi (1983) (#181), Silence of the Lambs, The (1991) (#98), Raiders of the Lost Ark (1981) (#174), Twelve Monkeys (1995) (#7), Pulp Fiction (1994) (#56), Rock, The (1996) (#117), Empire Strikes Back, The (1980) (#172), Dead Man Walking (1995) (#9), Independence Day (ID4) (1996) (#121), Shawshank Redemption, The (1994) (#64).
- Disappeared: Fargo (1996) (#100), Contact (1997) (#258), English Patient, The (1996) (#286), Schindler's List (1993) (#318), One Flew Over the Cuckoo's Nest (1975) (#357), Casablanca (1942) (#483), Scream (1996) (#288), Air Force One (1997) (#300), Star Trek: First Contact (1996) (#222), Titanic (1997) (#313).
- Mean training popularity: 245.5 -> 254.9; zero scores: 0 -> 0.

### change-alpha
- Common: 7/10 — Fargo (1996) (#100), Contact (1997) (#258), English Patient, The (1996) (#286), Schindler's List (1993) (#318), One Flew Over the Cuckoo's Nest (1975) (#357), Casablanca (1942) (#483), Star Trek: First Contact (1996) (#222).
- Appeared: Trainspotting (1996) (#475), Sense and Sensibility (1995) (#275), E.T. the Extra-Terrestrial (1982) (#423).
- Disappeared: Scream (1996) (#288), Air Force One (1997) (#300), Titanic (1997) (#313).
- Mean training popularity: 245.5 -> 228.4; zero scores: 0 -> 0.

## Observations

- Changing the target from user 1 to user 2 at alpha=0.85 keeps 0/10 movies. The mean training popularity changes from 245.5 to 254.9. This describes these two lists, not a quality improvement.
- The target-user change also changes which rated IDs are filtered, under the same train+validation exclusion rule. For example, Return of the Jedi (1983) (5/5 in train) cannot appear for user 1, while Return of the Jedi appears for user 2. Fargo (1996) (5/5 in train) and Contact (1997) (3/5 in train) are filtered for user 2 regardless of rating. These filtering facts alone can create list differences.
- User 1 has 217 train ratings, 131 of them positive. Drama has 61 positive films, Comedy 39, Action 36 and Sci-Fi 27. Contact and Star Trek: First Contact share Sci-Fi with Back to the Future (1985) (5/5 in train); Fargo shares Crime/Drama with Godfather, The (1972) (5/5 in train) and Pulp Fiction (1994) (4/5 in train).
- User 2 has 49 train ratings, 32 positive. Drama has 18 positive films, Romance 11 and Comedy 11. Return of the Jedi and The Empire Strikes Back have a concrete title/genre connection to Star Wars (1977) (5/5 in train); Pulp Fiction shares Crime/Drama with Fargo (1996) (5/5 in train) and Godfather, The (1972) (5/5 in train).
- Changing only alpha for user 1 from 0.85 to 0.50 retains 7/10 movies. Trainspotting (Drama), Sense and Sensibility (Drama/Romance), and E.T. (Children's/Drama/Fantasy/Sci-Fi) appear; Scream, Air Force One and Titanic disappear. Drama tags increase from 7 to 9. The mean training popularity decreases from 245.5 to 228.4.
- Sense and Sensibility's Drama/Romance tags are already represented by Postino, Il (1994) (5/5 in train) and When Harry Met Sally... (1989) (5/5 in train). These are descriptive genre connections, not traced causal explanations of the walks: PPR uses positive rating edges, not genre features.
- Zero-score counts (baseline, change-user, change-alpha): 0, 0, 0. In general, a tie-broken zero-score movie has not demonstrated relevance. Different lists do not prove better quality, and lower training popularity alone does not establish a useful discovery.
- The three settings are fixed controlled examples. They do not choose or replace the separately saved final alpha=0.50. Only train records explain the profiles; held-out ratings are used for exclusion checks, not profile evidence.

## Checkable files

See `ppr-settings-examples.csv` for the three full Top-10 lists and `ppr-settings-comparisons.csv` for deltas. Full histories are in `ppr-train-histories.csv`; counts and genre distributions are in `ppr-train-summary.csv` and `ppr-train-genres.csv`. JSON records source hashes and history hashes. No MovieLens propagation, MF training or population metric was rerun.
