# v5 energy-model revision

The v5 analysis uses the original thesis measurements through a local copy of the data loader. Earlier paper versions and their analysis remain unchanged.

## Accounting correction

The live residual is the W2 inference-minus-reference power difference minus the actual W2 prediction rate multiplied by the unbatched W1 incremental energy per prediction. W2 power and prediction rate are bootstrapped together. A residual is retained only when its bootstrap 95% interval is positive; unresolved residuals are set to zero. Batched projections reuse the unbatched residual and substitute batch-32 prediction energy.

For the representative heavy predictor, the Pi residual is 0.2787 W (95% interval 0.2317–0.3225 W); the Jetson residual is 0.0166 W (interval includes zero, therefore modeled as zero). Both light residuals are unresolved. This is an effective decomposition, not a causal hardware attribution.

## Reproduction

Run `python model/audit.py` from the v5 directory to regenerate `model/audit_results.json`. The script verifies offset accounting and scenario consistency and records crossover intervals, nominal energy, batching savings, and sensitivity results. Percentile arrays are ordered 5th/50th/95th unless explicitly labeled CI95.

Run `python figures_src/redraw.py --model-only` to regenerate only the scaling figure, then `latexmk -pdf main.tex` to rebuild the paper. NumPy, pandas, SciPy and Matplotlib are required.

## Narrative order

III.G defines the model, parameter-estimation procedure, capacity assumptions, and validation/sensitivity design without referring forward to Table IV. IV.F follows the measured energy results and reports the parameter values and checks. IV.G then applies those values to concurrent monitoring. Table IV remains with calibration results; it is not required to understand methodology. The text before III.G is unchanged.
