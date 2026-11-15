# Lyrics-Based Auto-Tagging (CSC 575, DP1_A)

Predicting the tags a listener would attach to a song from its lyrics alone.

## Project Design Specification

- Project Title: Lyrics-based auto-tagging of music
- Default Project: DP1_A
- Tools: Python, Jupyter Notebook, NumPy, pandas, matplotlib, scikit-learn, NLTK
- Datasets: Music4All-EN [2], the English lyrics subset of Music4All, from
  <https://drive.google.com/file/d/1Z9GiOCju9go-TnnJA1g5mQQsNbYlblz0/view>: 84,103 tracks,
  11,877 artists, 14,507 Last.fm tags, 671 genre tags, full lyrics, and six Spotify audio
  descriptors per track. No audio clips, so everything here is text plus metadata. The data is
  not committed to this repository; download it and unzip it to `music4all-en/`
- Team: Tan Dai Ngo (data, features, models), Maki Akinkunmi (evaluation, audio, reporting)
- Approach: TF-IDF and bag-of-words features, binary relevance over Naive Bayes, logistic
  regression and a linear SVM, scored per tag at `N` = 20, 50 and 100 tags
- Splits: two seeded 70/15/15 splits, random and artist-disjoint, sharing one vocabulary

### Related Work

Turnbull et al. [6] model music and words with a multi-class Naive Bayes approach, treating
semantic annotation as one M-class problem rather than M binary one-versus-all problems, where M
is the size of the vocabulary. Binary relevance is the second of those routes and it is the one
we take, so we keep per-tag scores and tune thresholds. Levy and Sandler [3] derive a semantic
space from social tags and observe that such tags unavoidably contain a great deal of irrelevant
content, which matters because many of the 14,507 tags here describe the listener rather than
the music. The Million Song Dataset [7] set the scale of tagging research, but Music4All [2]
ships lyrics and tags together.

On lyrics, Kleedorfer et al. [8] clustered song lyrics to detect recurring topics, and Mayer et
al. [4] added rhyme and text statistics over 397 song lyrics in ten genres; we take their line
statistics but not the phoneme work. Mayer and Rauber [9] combined audio and lyrics through
classifier ensembles rather than concatenating features. Hu et al. [10] found part-of-speech
features and function words are not as good as bag-of-words for mood, and Hu and Downie [1] then
showed lyric features significantly outperformed audio spectral features in seven of 18 mood
categories, with audio ahead in one. McKay et al. [11] ranked lyrical features weakest against
audio, symbolic and cultural ones for genre. Fell and Sporleder [12] modelled style and
structure, and Tsaptsinos [13] used a hierarchical attention network on a 117-genre dataset and
a reduced 20-genre one, where the attention network won on 117 but a plain LSTM beat it on 20.
Choi et al. [14] introduced a convolutional recurrent network for music tagging. Rennie et al.
[15] document the poor assumptions of multinomial Naive Bayes on text, Fan et al. [16] describe
the LIBLINEAR solver behind our linear SVM, and the implementations come from scikit-learn [5].
None of these ask, tag by tag, which of the Last.fm tags on Music4All lyrics can predict as the
vocabulary size varies.

### Tan Dai Ngo

- Objective: Music4All lyrics and tag pipeline (Oct 30)
  - (basic) load the five CSVs and the 84,103 lyrics files, reconcile the row counts
  - (basic) clean lyrics, strip blank lines, drop exact duplicates
  - (expected) normalise tags, apply the exclusion list, build the target matrix
  - (expected) build both splits, pick the shared vocabulary, report per-tag counts
  - (advanced) compare the splits on artist coverage and lost test positives
- Objective: Features and the three classifiers (Nov 20)
  - (basic) bag-of-words plus Naive Bayes per tag, micro-F1 on validation
  - (basic) switch to TF-IDF, sweep min_df, max_df and n-gram range
  - (expected) add logistic regression and linear SVM, compare against both references
  - (expected) time training and prediction at 10, 25, 50 and 100 percent of the data
  - (advanced) add the line statistics from Mayer et al. [4], test the rare tags
- Objective: Vocabulary sweep and thresholds (Dec 4)
  - (basic) plot tag frequency, count how many of the 14,507 tags fall under 100 songs
  - (basic) per-tag table at `N` = 50, name the five best and worst tags
  - (expected) run the comparison at `N` = 20, 50 and 100, where support on the shared
    training rows runs from 9,818 songs down to 238
  - (expected) tune per-tag thresholds on validation, report the macro-F1 change
  - (advanced) extract top-weighted terms per tag, flag artist-name leakage

### Maki Akinkunmi

- Objective: Evaluation module (Nov 13)
  - (basic) write per-tag precision, recall, F1 and support scoring to CSV
  - (basic) add micro and macro averages plus label-wise accuracy
  - (expected) add average precision per tag and its macro mean
  - (expected) add a 95% bootstrap interval on the classifier comparison
  - (advanced) check how much the interval widens when clustered on artist
- Objective: Audio descriptor comparison (Nov 27)
  - (basic) load the six Spotify descriptors, report how many songs have all six
  - (basic) one-hot encode key, standardise the rest, impute the median
  - (expected) train logistic regression on the same splits, produce the per-tag table
  - (expected) compare against the lyrics models tag by tag
  - (advanced) write the category rubric, annotate `N` = 100, score by category
- Objective: Error analysis, figures and the report (Dec 4)
  - (basic) produce the per-tag F1 chart and tag frequency plot
  - (basic) read lyrics from the worst tags and describe what is missed
  - (expected) separate tag noise from model error using the top terms
  - (expected) convert the README into the ISMIR LaTeX template
  - (advanced) scatter per-tag F1 against prevalence, coloured by category

Progress report due Nov 13. Final report and presentation Dec 4. Code review, the reading
above and the presentation are shared. Stretch items if the schedule holds: classifier chains,
a sentence-embedding comparison, and words-per-minute from the `duration_ms` field. The subset
ships no audio, so spectrogram work is out of scope.

### References

[1] X. Hu and J. S. Downie, "When lyrics outperform audio for music mood classification: a feature analysis," ISMIR, Utrecht, 2010, pp. 619-624.

[2] I. A. P. Santana, F. Pinhelli, J. Donini, L. Catharin, R. B. Mangolin, Y. M. G. da Costa, V. D. Feltrim, and M. A. Domingues, "Music4All: a new music database and its applications," IWSSIP, Niteroi, 2020.

[3] M. Levy and M. Sandler, "A semantic space for music derived from social tags," ISMIR, Vienna, 2007.

[4] R. Mayer, R. Neumayer, and A. Rauber, "Rhyme and style features for musical genre classification by song lyrics," ISMIR, Philadelphia, 2008, pp. 337-342.

[5] F. Pedregosa et al., "Scikit-learn: machine learning in Python," J. Machine Learning Research, 12, pp. 2825-2830, 2011.

[6] D. Turnbull, L. Barrington, and G. Lanckriet, "Modeling music and words using a multi-class naive Bayes approach," ISMIR, Victoria, 2006.

[7] T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, and P. Lamere, "The million song dataset," ISMIR, Miami, 2011, pp. 591-596.

[8] F. Kleedorfer, P. Knees, and T. Pohle, "Oh oh oh whoah! Towards automatic topic detection in song lyrics," ISMIR, Philadelphia, 2008, pp. 287-292.

[9] R. Mayer and A. Rauber, "Music genre classification by ensembles of audio and lyrics features," ISMIR, Miami, 2011, pp. 675-680.

[10] X. Hu, J. S. Downie, and A. F. Ehmann, "Lyric text mining in music mood classification," ISMIR, Kobe, 2009, pp. 411-416.

[11] C. McKay, J. A. Burgoyne, J. Hockman, J. B. L. Smith, G. Vigliensoni, and I. Fujinaga, "Evaluating the genre classification performance of lyrical features relative to audio, symbolic and cultural features," ISMIR, Utrecht, 2010, pp. 213-218.

[12] M. Fell and C. Sporleder, "Lyrics-based analysis and classification of music," COLING, Dublin, 2014, pp. 620-631.

[13] A. Tsaptsinos, "Lyrics-based music genre classification using a hierarchical attention network," ISMIR, Suzhou, 2017, pp. 694-701.

[14] K. Choi, G. Fazekas, M. Sandler, and K. Cho, "Convolutional recurrent neural networks for music classification," ICASSP, New Orleans, 2017.

[15] J. D. M. Rennie, L. Shih, J. Teevan, and D. R. Karger, "Tackling the poor assumptions of naive Bayes text classifiers," ICML, Washington DC, 2003.

[16] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin, "LIBLINEAR: a library for large linear classification," J. Machine Learning Research, 9, pp. 1871-1874, 2008.
