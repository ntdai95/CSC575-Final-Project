# Lyrics-Based Auto-Tagging (CSC 575, DP1_D)

Predicting the tags a listener would attach to a song from its lyrics alone, using Music4All.

## Project Design Specification

- Project Title: Lyrics-based auto-tagging of music (DP1_D)
- Tools: Python, Jupyter Notebook, NumPy, pandas, matplotlib, scikit-learn, NLTK
- Datasets: Music4All, <https://sites.google.com/view/contact4music4all>

## 1. Project Overview

Music auto-tagging means predicting descriptive labels for a track: genres like `shoegaze`,
moods like `melancholic`, settings like `workout`. Almost all published work starts from audio.
We want to know how far the lyrics alone get you. We use TF-IDF and bag-of-words features with
the classical multi-label classifiers in scikit-learn, not a neural architecture.

## 2. Problem Statement

Given the lyrics of a song as plain text, predict a binary vector over a vocabulary of `N`
tags. This is multi-label, not multi-class: a song can be `indie`, `female vocalists` and
`chill` at once.

- Measure how well a linear classifier over lyrics predicts Last.fm tags, per tag
- Find which tags are learnable from lyrics and which are not
- Vary vocabulary size `N` and see how the picture changes
- Compare against a model built on the six audio descriptors in the metadata

Non-goals: no recommender, no audio deep learning, no tag generation.

## 3. Motivation

Lyrics and audio carry different information, and the difference should show up unevenly across
tags. Hu and Downie [1] found that across 18 mood categories, lyric features significantly
outperformed audio spectral features in seven, and audio outperformed every lyric feature type
in one. We want to see whether that holds on a much larger and messier tag set.

## 4. Dataset Description

Music4All [2]. Access is by request to `contact4music4all@gmail.com`.

| Property | Figure |
|---|---|
| Songs | 109,269 (30-second clips) |
| Artists | 16,269 |
| Unique genre tags | 853 |
| Unique Last.fm user tags | 19,541 |
| Languages in the lyrics | 46 |
| Metadata fields per song | 16 |

The 16 fields include `lang`, `genres`, `tags`, and six Spotify audio descriptors
(`danceability`, `energy`, `key`, `mode`, `valence`, `tempo`). Our target is `tags`.

- Many tags describe the listener rather than the music (`seen live`, `albums i own`, artist
  names); Lamere [3] documents this kind of noise. Objective 3 measures how far the frequency
  distribution tails off.
- A zero usually means nobody applied the tag, not that it is false, so a correct prediction
  can be scored as a false positive and precision reads low.
- We filter to English on `lang`, assigned by LangDetect.
- Artists have characteristic vocabularies and tags, so we add an artist-disjoint split.

Lyrics are copyrighted, so the repository holds code and derived results only.

## 5. Proposed Methodology

- Clean lyrics into two versions: structural (line breaks kept, for style and rhyme features)
  and vectoriser (lowercased, markers like `[Chorus]` stripped). Filter to English.
- Drop exact duplicate lyrics, then near-duplicates at Jaccard similarity 0.9 or above on
  lowercased token sets, reporting how many pairs were removed. Heavily rewritten covers fall
  under that cutoff, and two different artists covering one song are not caught by the
  artist-disjoint split either, so some song-level overlap may remain.
- Normalise tags, merge aliases, exclude listener tags and artist names, then take the top `N`.
- Two seeded splits, 70/15/15 train/validation/test: random and artist-disjoint. Even splits
  are harder for multi-label data [4], so report per-tag positive counts for each.
- The tag vocabulary is chosen once, from the songs in the training portion of both splits, so
  neither split's held-out labels inform it and both are scored over the same tags. Vectorisers
  and feature scalers are fit on training, thresholds tuned on validation, and the test set
  scored once.
- Hyperparameters come from a grid search on validation macro-F1, with the grids in the
  repository.
- Features: bag-of-words and TF-IDF, plus a dense block from Mayer et al. [5]: text statistics,
  part-of-speech ratios, repetition, and phoneme rhymes via `nltk.corpus.cmudict`.
  `WordsPerMinute` is dropped since the metadata has no song duration.
- Models: binary relevance over Multinomial Naive Bayes, logistic regression and linear SVM,
  with classifier chains compared against binary relevance as the second label strategy. The
  dense block is standardised so it has negatives and goes only to the latter two; Naive Bayes
  runs on sparse features. Three references to beat: the per-tag majority class, a seeded
  `DummyClassifier(strategy="stratified")` per tag, and always-predict-positive, which on
  sparse tags scores better than the other two and is the real F1 floor.
- Audio comparison: same splits, six descriptors, logistic regression, `key` one-hot encoded.

## 6. Tools and Technologies

| Tool | Use |
|---|---|
| Python, Jupyter Notebook | Development environment, as in the course notebooks |
| NumPy, pandas | Data handling |
| scikit-learn [6] | Vectorisers, classifiers, metrics |
| NLTK | Tokenisation, POS tagging, `cmudict` for rhymes |
| matplotlib | Figures |

We are working from the scikit-learn docs on multi-label classification, text feature
extraction and model evaluation. The Chapter 1 notebooks, *A music and sound exposition of the
Discrete Fourier Transform*, parts 1-9, use the same libraries. Part 5 covers the DFT and part
6 covers short-time processing and spectrograms, which is where the optional audio work would
start. Appendix B of the MIR textbook is the reference for project structure.

## 7. Evaluation Metrics

Label-wise accuracy is inflated by sparse labels, so we report it once for the majority
reference and work from F1 elsewhere.

- Per-tag precision, recall, F1 and support
- Micro- and macro-averaged precision, recall and F1. The macro set is fixed from the songs in
  the training portion of both splits, the same pool the vocabulary comes from: tags with at
  least 50 positives there. Nothing is removed from it afterwards, so both splits are averaged
  over the identical tag set. A tag that happens to have no test positives still scores, at 0
  under `zero_division=0`, and we report how many of those there were
- Average precision per tag
- 95% bootstrap confidence intervals, 1,000 resamples, clustered on artist in both splits,
  since songs by one artist are not independent. One comparison is primary, the three
  classifiers at `N` = 50 on the random split, and the rest are read as exploratory, since an
  interval excluding zero means less once many comparisons are made
- Label cardinality and density at each `N` [7]
- Training and prediction time at 10%, 25%, 50% and 100% of the data
- Scores by tag category (genre, mood, instrumentation, other). The other bucket is reported on
  its own and not merged into the category comparison; it does not change the macro set
- Top-weighted terms per tag

Default thresholds are 0.5 for Naive Bayes and logistic regression and 0 for `LinearSVC`, which
has no `predict_proba`. The thresholds are then tuned per tag to maximise validation F1,
keeping the default for any tag with no validation positives.

## 8. Timeline

| Week | Dates | Objectives |
|---|---|---|
| 1 | Oct 19-23 | Dataset obtained, repository set up, §9 reading split three ways |
| 2 | Oct 26-30 | Data pipeline, tag vocabulary, target matrix, splits |
| 3 | Nov 2-6 | TF-IDF features, Naive Bayes baseline, first per-tag results |
| 4 | Nov 9-13 | **Progress report.** Logistic regression and SVM, three-way comparison |
| 5 | Nov 16-20 | Vocabulary sweep, timing runs, tag categories, audio comparison, classifier chains |
| 6 | Nov 23-27 | Style and rhyme features, threshold tuning, error analysis, figures |
| 7 | Nov 30-Dec 4 | **Final report** in ISMIR LaTeX format, presentation |

Stretch: sentence embeddings, multilingual lyrics, mel-spectrogram features.

## 9. Team Roles

- **A: Data, features and the lyrics models.** Tan Dai Ngo. Loading, cleaning, tag vocabulary,
  target matrix, splits, text and style features, the three classifiers, timing, thresholds.
- **B: Multi-label strategy and statistics.** Tan Dai Ngo & Maki Akinkunmi. Hyperparameter protocol, bootstrap
  intervals, and the binary relevance against classifier chains comparison.
- **C: Audio comparison, analysis and reporting.** Maki Akinkunmi. Descriptor baseline, tag
  category rubric, error analysis, figures, ISMIR LaTeX conversion.

Code review, written reports, and presentation are shared.

## Tan Dai Ngo

Workstream A: data, features and the lyrics models.

### Objective 1: Music4All lyrics and tag pipeline (Oct 30)

- PI1 (basic): load the metadata, `tags` and lyrics files and reconcile the row counts against
  the published figures (109,269 songs, 16,269 artists, 19,541 tags). Record any mismatch in
  `docs/dataset_notes.md`
- PI2 (basic): clean lyrics into the structural and vectoriser versions, filter to English on
  `lang`, and report how many songs survive each step
- PI3 (expected): normalise tags, apply the alias merge and exclusion list, and build the
  target matrix for `N` = 20, 50 and 100. Report label cardinality and density [7] before and
  after exclusion
- PI4 (expected): drop exact duplicates and near-duplicates at Jaccard 0.9, build both 70/15/15
  splits, select the shared tag vocabulary from the training intersection, and report per-tag
  positive counts for each split
- PI5 (advanced): compare the two splits on artist coverage and label distribution, and
  estimate how many near-duplicate pairs the Jaccard filter missed by hand-checking a sample

### Objective 2: Lyrics features and the three classifiers (Nov 27)

- PI1 (basic): extract bag-of-words counts with `CountVectorizer` and train Multinomial Naive
  Bayes per tag under binary relevance. Report micro-F1 on validation
- PI2 (basic): switch to TF-IDF and sweep `min_df`, `max_df` and n-gram range on validation,
  writing every configuration to a results CSV
- PI3 (expected): train logistic regression and a linear SVM under the same wrapper, compare
  all three against the three reference classifiers on micro and macro F1, and report the score
  drop from the random split to the artist-disjoint one
- PI4 (expected): add the style features and phoneme rhymes from Mayer et al. [5], report the
  macro-F1 change, and measure vectorisation, training and prediction time per classifier at
  10%, 25%, 50% and 100% of the training data
- PI5 (advanced): compare TF-IDF against a pretrained sentence-embedding representation of the
  lyrics, and test the macro-F1 difference on the rare tags, meaning the bottom quartile by
  training support fixed before the run, with a paired bootstrap interval

### Objective 3: Tag vocabulary and decision thresholds (Dec 4)

- PI1 (basic): plot the tag frequency distribution and report how many of the 19,541 tags apply
  to fewer than 100 songs, and how many the exclusion list removes
- PI2 (basic): produce the per-tag results table (precision, recall, F1, average precision,
  support) for the best model at `N` = 50, and name the five strongest and five weakest tags
- PI3 (expected): tune per-tag thresholds on validation, using probabilities for Naive Bayes
  and logistic regression and `decision_function` scores for the linear SVM, and report the
  change in macro-F1 with bootstrap intervals
- PI4 (expected): extract the top-weighted terms per tag and write up what drives each one,
  flagging tags where the model keys on artist names or language, and report macro-F1 and macro
  average precision within each tag category
- PI5 (advanced): test mood against genre on the prevalence-matched pairs from Workstream C
  with a paired bootstrap interval on the mean per-tag F1 difference

## Tan Dai Ngo & Maki Akinkunmi

Workstream B: multi-label strategy and statistics.

### Objective 1: Hyperparameter protocol and model selection (Nov 13) - Tan Dai Ngo

- PI1 (basic): write the grid search that tunes each classifier on validation macro-F1 and
  writes every configuration to a results CSV
- PI2 (basic): define the parameter grids for Naive Bayes, logistic regression and the linear
  SVM, and keep them in the repository
- PI3 (expected): run the search for all three classifiers at `N` = 50 and report the chosen
  settings with the validation score that selected them
- PI4 (expected): keep the search inside the training and validation portions, and document
  which quantities are fitted on which portion
- PI5 (advanced): re-run the selection on micro-F1 and macro average precision and report
  whether the winning classifier changes with the metric

### Objective 2: Uncertainty estimates (Nov 20) - Tan Dai Ngo

- PI1 (basic): implement the 95% bootstrap interval with 1,000 resamples and report it with
  each score in the comparison tables
- PI2 (basic): cluster the resampling on artist in both splits
- PI3 (expected): run the primary comparison, the three classifiers at `N` = 50 on the random
  split, with intervals on every figure
- PI4 (expected): implement the paired bootstrap on score differences, and report the interval
  for each pair of models at `N` = 50
- PI5 (advanced): compare artist-clustered intervals against song-level ones and report the
  difference in width

### Objective 3: Multi-label strategy comparison (Dec 4) - Maki Akinkunmi

- PI1 (basic): write the label-strategy wrapper so binary relevance and classifier chains run
  through one interface and emit the same results format
- PI2 (basic): count how many distinct label combinations occur at `N` = 20, 50 and 100
- PI3 (expected): implement classifier chains and compare against binary relevance on micro and
  macro F1 with paired intervals
- PI4 (expected): measure what the chains cost in training and prediction time against binary
  relevance
- PI5 (advanced): test whether the chain order matters by running several random orders and
  reporting the spread

## Maki Akinkunmi

Workstream C: audio comparison, analysis and reporting.

### Objective 1: Audio-descriptor baseline (Nov 20)

- PI1 (basic): load the six Spotify descriptors (`danceability`, `energy`, `key`, `mode`,
  `valence`, `tempo`) and report how many songs have complete values for all six descriptors
- PI2 (basic): one-hot encode `key` as a pitch class, standardise the continuous descriptors,
  and impute missing values with the training median plus an indicator column
- PI3 (expected): train logistic regression under the same binary-relevance wrapper and the
  same splits, and produce the per-tag table in the shared format
- PI4 (expected): compare the descriptor model against the lyrics models tag by tag and report
  which tags each model does better on, with intervals
- PI5 (advanced): extract mel-spectrogram features from the 30-second clips with librosa,
  following part 6 of the Chapter 1 notebooks, and compare against the six descriptors

### Objective 2: Tag category rubric and annotation (Nov 27)

- PI1 (basic): write the rubric for sorting tags into genre, mood, instrumentation and other,
  and keep it in the repository
- PI2 (basic): annotate the `N` = 100 vocabulary against the rubric, which covers the `N` = 20
  and 50 sets too, and publish the grouping as a CSV
- PI3 (expected): have a second team member annotate the same vocabulary independently, each
  picking one primary category, and report Cohen's kappa on those primary labels
- PI4 (expected): build the prevalence-matched mood and genre pairs that Objective 3 PI4 in
  Workstream A uses, pairing each mood tag with the genre tag closest in training prevalence
  within 20% relative, and report how many mood tags have no partner
- PI5 (advanced): record secondary categories for tags that fit two groups, and report how much
  the category scores move when those tags are counted in both

### Objective 3: Error analysis, figures and the report (Dec 4)

- PI1 (basic): produce the per-tag F1 chart and the category score chart from the results CSVs
- PI2 (basic): pull the worst-scoring tags and read a sample of their lyrics to describe what
  the model misses
- PI3 (expected): cross-reference the top-weighted terms from Workstream A against the error
  cases and separate failures that look like tag noise from failures that look like model error
- PI4 (expected): convert the README into the ISMIR LaTeX template with the bibliography as a
  `.bib` file
- PI5 (advanced): add a scatter plot of per-tag F1 against training prevalence, with the tag
  categories coloured separately

## References

[1] X. Hu and J. S. Downie, "When lyrics outperform audio for music mood classification: a
feature analysis," ISMIR, Utrecht, 2010, pp. 619-624.

[2] I. A. P. Santana et al., "Music4All: a new music database and its applications," IWSSIP,
Niterói, 2020, pp. 399-404.

[3] P. Lamere, "Social tagging and music information retrieval," *J. New Music Research*, vol.
37, no. 2, pp. 101-114, 2008.

[4] K. Sechidis, G. Tsoumakas, and I. Vlahavas, "On the stratification of multi-label data,"
ECML PKDD, Athens, 2011, pp. 145-158.

[5] R. Mayer, R. Neumayer, and A. Rauber, "Rhyme and style features for musical genre
classification by song lyrics," ISMIR, Philadelphia, 2008, pp. 337-342.

[6] F. Pedregosa et al., "Scikit-learn: machine learning in Python," *J. Machine Learning
Research*, vol. 12, pp. 2825-2830, 2011.

[7] G. Tsoumakas and I. Katakis, "Multi-label classification: an overview," *Int. J. Data
Warehousing and Mining*, vol. 3, no. 3, pp. 1-13, 2007.

[8] D. Turnbull, L. Barrington, D. Torres, and G. Lanckriet, "Semantic annotation and retrieval
of music and sound effects," *IEEE TASLP*, vol. 16, no. 2, pp. 467-476, 2008.

[9] T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, and P. Lamere, "The million song dataset,"
ISMIR, Miami, 2011, pp. 591-596.

[10] J. P. G. Mahedero, Á. Martínez, P. Cano, M. Koppenberger, and F. Gouyon, "Natural language
processing of lyrics," ACM Multimedia, Singapore, 2005, pp. 475-478.

[11] R. Mayer and A. Rauber, "Musical genre classification by ensembles of audio and lyrics
features," ISMIR, Miami, 2011, pp. 675-680.

[12] X. Hu, J. S. Downie, and A. F. Ehmann, "Lyric text mining in music mood classification,"
ISMIR, Kobe, 2009, pp. 411-416.

[13] C. McKay, J. A. Burgoyne, J. Hockman, J. B. L. Smith, G. Vigliensoni, and I. Fujinaga,
"Evaluating the genre classification performance of lyrical features relative to audio,
symbolic and cultural features," ISMIR, Utrecht, 2010, pp. 213-218.

[14] M. Fell and C. Sporleder, "Lyrics-based analysis and classification of music," COLING,
Dublin, 2014, pp. 620-631.

[15] A. Tsaptsinos, "Lyrics-based music genre classification using a hierarchical attention
network," ISMIR, Suzhou, 2017, pp. 694-701.

[16] K. Choi, G. Fazekas, M. Sandler, and K. Cho, "Convolutional recurrent neural networks for
music classification," ICASSP, New Orleans, 2017, pp. 2392-2396.
