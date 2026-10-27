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

Turnbull et al. [8] frame annotation and retrieval as one supervised multiclass, multilabel
problem over CAL500, arguing against one-versus-all because it "results in binary decisions for
each word"; binary relevance, which we use, is that approach, so we keep per-tag scores and
tune thresholds. Lamere [3] surveys social tagging and its noise, which matters because many
of the 14,507 tags here describe the listener rather than the music. The Million Song Dataset [9] set
the scale of tagging research, but Music4All [2] ships lyrics and tags together.

On lyrics, Mahedero et al. [10] applied standard NLP tooling early, and Mayer et al. [5] added
rhyme and text statistics over 397 lyrics in ten genres; we take their line statistics but not
the phoneme work. Mayer and Rauber [11] fused classifier outputs across audio and lyrics rather
than concatenating features. Hu et al. [12] found plain bag-of-words beat part-of-speech
features for mood, and Hu and Downie [1] then showed lyric features significantly outperformed
audio spectral features in seven of 18 mood categories, with audio ahead in one. McKay et al.
[13] ranked lyrical features weakest against audio, symbolic and cultural ones for genre. Fell
and Sporleder [14] modelled style and structure, and Tsaptsinos [15] used a hierarchical
attention network on 117 and 20 genres, beating an LSTM on 117 but not on 20. Choi et al. [16]
introduced a convolutional recurrent network for audio tagging. For the multi-label machinery
we take binary relevance and the cardinality and density measures from Tsoumakas and Katakis
[7], Sechidis et al. [4] show even splits are harder than in the single-label case, and the
implementations come from scikit-learn [6]. None of these ask, tag by tag, which of the
Last.fm tags on Music4All lyrics can predict as the vocabulary size varies.

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
  - (advanced) add the line statistics from Mayer et al. [5], test the rare tags
- Objective: Vocabulary sweep and thresholds (Dec 4)
  - (basic) plot tag frequency, count how many of the 14,507 tags fall under 100 songs
  - (basic) per-tag table at `N` = 50, name the five best and worst tags
  - (expected) run the comparison at `N` = 20, 50 and 100, where support runs from 22,652
    songs down to 530
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

[1] X. Hu and J. S. Downie, "When lyrics outperform audio for music mood classification," ISMIR, Utrecht, 2010, pp. 619-624.

[2] I. A. P. Santana et al., "Music4All: a new music database and its applications," IWSSIP, Niteroi, 2020, pp. 399-404.

[3] P. Lamere, "Social tagging and music information retrieval," J. New Music Research, 37(2), pp. 101-114, 2008.

[4] K. Sechidis, G. Tsoumakas, and I. Vlahavas, "On the stratification of multi-label data," ECML PKDD, Athens, 2011, pp. 145-158.

[5] R. Mayer, R. Neumayer, and A. Rauber, "Rhyme and style features for musical genre classification by song lyrics," ISMIR, Philadelphia, 2008, pp. 337-342.

[6] F. Pedregosa et al., "Scikit-learn: machine learning in Python," J. Machine Learning Research, 12, pp. 2825-2830, 2011.

[7] G. Tsoumakas and I. Katakis, "Multi-label classification: an overview," Int. J. Data Warehousing and Mining, 3(3), pp. 1-13, 2007.

[8] D. Turnbull, L. Barrington, D. Torres, and G. Lanckriet, "Semantic annotation and retrieval of music and sound effects," IEEE TASLP, 16(2), pp. 467-476, 2008.

[9] T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, and P. Lamere, "The million song dataset," ISMIR, Miami, 2011, pp. 591-596.

[10] J. P. G. Mahedero, A. Martinez, P. Cano, M. Koppenberger, and F. Gouyon, "Natural language processing of lyrics," ACM Multimedia, Singapore, 2005, pp. 475-478.

[11] R. Mayer and A. Rauber, "Musical genre classification by ensembles of audio and lyrics features," ISMIR, Miami, 2011, pp. 675-680.

[12] X. Hu, J. S. Downie, and A. F. Ehmann, "Lyric text mining in music mood classification," ISMIR, Kobe, 2009, pp. 411-416.

[13] C. McKay, J. A. Burgoyne, J. Hockman, J. B. L. Smith, G. Vigliensoni, and I. Fujinaga, "Evaluating the genre classification performance of lyrical features relative to audio, symbolic and cultural features," ISMIR, Utrecht, 2010, pp. 213-218.

[14] M. Fell and C. Sporleder, "Lyrics-based analysis and classification of music," COLING, Dublin, 2014, pp. 620-631.

[15] A. Tsaptsinos, "Lyrics-based music genre classification using a hierarchical attention network," ISMIR, Suzhou, 2017, pp. 694-701.

[16] K. Choi, G. Fazekas, M. Sandler, and K. Cho, "Convolutional recurrent neural networks for music classification," ICASSP, New Orleans, 2017, pp. 2392-2396.
