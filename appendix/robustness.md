# Appendix A: Robustness Checks

This is appendix material from the paper *Energy-Aware Assessment and
Energy Modeling for Encrypted Video KQI Prediction at the Edge*, cut from the submitted paper
for space. The content and all numbers are unchanged; only the LaTeX markup has been converted
to Markdown.

## Outlier filtering

Table A.1 compares each measurement series with and without the energy-based IQR filter used in
the main analysis. W1 batch-32 covers the 80 Raspberry Pi and 96 Jetson configurations with
valid outputs, W2 the 21/22 valid candidates and their reference series, and W3 the four traffic
views at 100 streams with and without dissection.

**Table A.1.** Effect of energy-outlier filtering: excluded/total runs and the maximum absolute
change in a series mean relative to the unfiltered mean.

| Workload | Raspberry Pi | Jetson |
|---|---|---|
| W1, batch 1 | 123/2880; 1.33 % | 58/2880; 0.88 % |
| W1, batch 32 | 83/2400; 6.27 % | 200/2880; 5.12 % |
| W2, paired series | 5/1260; 0.23 % | 19/1320; 0.17 % |
| W3, 100 streams | 4/240; 0.22 % | 2/240; 0.01 % |

For unbatched W1, removing the filter changes median whole-device savings from 71.8 % to 71.7 %
on the Raspberry Pi and leaves the Jetson result at 19.4 %. Subtracting the twelve-hour idle
baseline, *E* − *P*<sub>idle</sub>*T*, gives median savings of 75.8 % and 28.6 %; these are
savings per completed W1 job and do not remove the baseline of an always-on monitor. For W2,
filtering changes any candidate-minus-reference mean by at most 1.87 J (Raspberry Pi) and
2.14 J (Jetson).

## Transfer of the workstation energy proxy to the edge

**Table A.2.** Spearman correlation between the workstation energy objective of the selected
trials and their W1 whole-device edge energy (16 configurations per family and device).

| Family | Raspberry Pi | Jetson |
|---|---|---|
| RF | 0.87 | 0.40 |
| LGBM | 0.91 | 0.86 |
| FFNN | 0.25 | 0.12 |
| CNN | 0.74 | 0.56 |
| LSTM | 0.91 | 0.91 |
| TF | 0.76 | 0.63 |

Table A.2 shows how well the workstation energy objective used during selection transfers to the
edge. LGBM and LSTM transfer consistently, FFNN poorly, and RF depends strongly on the execution
platform, where Jetson trees run as compiled GPU tensor operations. Workstation-based selection
is therefore a useful heuristic for some families, but it does not guarantee edge-optimal
configurations.
