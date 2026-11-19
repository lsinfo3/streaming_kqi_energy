# Appendix B: Hyperparameter Search Spaces

This is appendix material from the paper *Energy-Aware Assessment and
Energy Modeling for Encrypted Video KQI Prediction at the Edge*, cut from the submitted paper
for space. The content and all numbers are unchanged; only the LaTeX markup has been converted
to Markdown.

**Table B.1.** Architecture search ranges. 2<sup>*a*:*b*</sup> denotes powers of two with integer
exponents from *a* to *b*; *L* is the input sequence length.

| Family | Search dimensions |
|---|---|
| RF | Trees 10–1000 (step 10); depth 2<sup>2:6</sup>; leaves 2<sup>6:12</sup>; minimum leaf samples 1–16; feature fraction 0.1–0.8. |
| LGBM | Trees 10–5000 (step 10); depth 2<sup>2:8</sup>; leaves 2<sup>4:7</sup>; learning rate 10<sup>−5</sup>–10<sup>−1</sup>. |
| FFNN | 1–4 layers; width 2<sup>5:11</sup>. |
| CNN | 1–3 layers; kernels 2<sup>4:6</sup>; kernel length 2–min(*L*, 10). |
| LSTM | 1–3 layers; hidden size 2<sup>4:8</sup>. |
| TF | 1–3 layers; heads 2<sup>1:3</sup>; model width 2<sup>5:8</sup>; feed-forward width 2<sup>6:10</sup>. |

Neural searches use learning rates 10<sup>−5</sup>–10<sup>−1</sup> (log scale), training batch
sizes 2<sup>7:12</sup>, and dropout 0–0.5 in steps of 0.1; stalling positive-class weights range
from 1 to 20. Multiwindow inputs use a maximum window of 5, 10, 20, or 30 s, with or without
feature selection; sequence models may also use a single 0.5 or 1 s window. Sequence histories
contain 5–60 one-second vectors in steps of five, or 10–120 half-second vectors in steps of ten.

## Source note

Two descriptions of the search space exist in the paper sources: this appendix table, and an
earlier table that is commented out in `sections/methodology.tex` (around lines 28–61). **This
document uses the appendix version**, for two reasons.

First, it is a strict superset of the commented-out table everywhere the two overlap: the
ranges agree for every parameter of every family, and the appendix additionally states the
step sizes (trees in steps of 10, dropout in steps of 0.1, sequence histories in steps of five
or ten), enumerates the admissible maximum windows (5, 10, 20, 30 s) rather than writing
"5–30 s", and gives the CNN kernel length explicitly.

Second, the two disagree on one entry, and the appendix is the consistent one. The commented-out
table lists the CNN under a shared row "Layer / hidden size (2<sup>*x*</sup>)" with the range
2–10, alongside its own "Kernels (2<sup>*x*</sup>) 4–6" row. The appendix gives the CNN kernels
as 2<sup>4:6</sup>, which matches, and the kernel length as 2–min(*L*, 10), which is a linear
range capped by the sequence length — not a power-of-two hidden size. The "2–10" of the
commented-out table is therefore the kernel length placed in a power-of-two row and without the
*L* cap, i.e. mislabelled.

One detail appears only in the commented-out table: it records a sequence length of 1 for RF,
LGBM and FFNN, that is, those families consume a single feature vector rather than a history.
Section III-C of the paper states this in prose, so nothing is lost by using the appendix
version.
