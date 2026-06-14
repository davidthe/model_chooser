# Sample report for `348_3758c_default`

This report is based on the result artifacts already present in the repository.
It summarizes the four model outputs that were available for the sample image.

## Scope
- Ground truth lines: 28
- Models evaluated: 4
- Line-level comparisons: 112

## Model summary
```text
                        model  pll_score   wer   cer  levenshtein_distance  lines
                    italian_7    125.931 0.336 0.079                 4.500     28
              sinai_no_voc_61    157.511 0.644 0.219                12.286     28
                prenumeranten    201.468 1.075 0.609                33.179     28
italian_7_retrained_bnf150_6p    228.806 1.467 0.518                27.679     28
```

## Best scores on this sample
- Best PLL: `italian_7`
- Best WER: `italian_7`
- Best CER: `italian_7`
- Best Levenshtein: `italian_7`

## Correlation matrix
```text
                      pll_score    wer    cer  levenshtein_distance
pll_score                 1.000  0.859  0.813                 0.810
wer                       0.859  1.000  0.832                 0.820
cer                       0.813  0.832  1.000                 0.994
levenshtein_distance      0.810  0.820  0.994                 1.000
```

## Note
The repo is now prepared for a full corpus evaluation run, but the runtime visible in this session does not expose a GPU, so the complete dataset sweep has not been finished here.