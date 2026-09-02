# ELECTRE-T2B Code

Computational code for the paper "An improved elimination and choice translating reality method integrating bootstrap-based thresholds and Bonferroni mean for urban sustainability assessment".

## Dependencies

- Python 3.9+
- numpy, pandas, scipy

## Structure

```
chapter3/   Method implementation and examples
  electre_t2b.py        core computations (thresholds, weights, concordance, credibility, comprehensive index)
  example_threshold.py  Example 1: thresholds and weights
  example_bm.py         Example 2: interaction parameters
chapter4/   Simulation validation
  threshold_robustness.py  Section 4.1 threshold estimation robustness (4050 combinations, B=5000, 1000 replications)
  rank_reversal.py         Section 4.2 rank reversal (972 combinations, 1000 replications)
chapter5/   Case study
  case_study.py   Tables 4 and 7
  sensitivity.py  Table 8
  comparative.py  Table 9
chapter6/   Discussion
  threshold_analysis.py  Section 6.2 threshold ratios
data/       case data data_sc.xlsx
```

## Run

Run from the corresponding directory, e.g., `python chapter5/case_study.py`.

The simulations in Chapter 4 are computationally heavy (4050/972 combinations times 1000 replications) and take a long time. All random numbers use the fixed seed SEED=2026, so the results are reproducible.