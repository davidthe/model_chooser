# Mean Dataset Correlation Report

Run: `full_run_20260617`

This report computes a correlation matrix separately for each handwriting dataset, then averages those matrices element by element. Each dataset therefore contributes equally, unlike a pooled correlation table where datasets with more pages or model rows can dominate.

## Summary
| Variant | Datasets | PLL-WER | PLL-CER | PLL-Lev | WER-CER | CER-Lev |
| --- | --- | --- | --- | --- | --- | --- |
| base_valid_all_pages | 17 | 0.408 | 0.426 | 0.499 | 0.720 | 0.954 |
| base_valid_after_worst20_trim | 11 | 0.605 | 0.640 | 0.697 | 0.809 | 0.951 |
| high_confidence_after_worst20_trim | 11 | 0.461 | 0.496 | 0.565 | 0.785 | 0.863 |

## Recommended Mean Correlation Table
Variant: `base_valid_after_worst20_trim`

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.605 | 0.640 | 0.697 |
| WER | 0.605 | 1.000 | 0.809 | 0.792 |
| CER | 0.640 | 0.809 | 1.000 | 0.951 |
| Levenshtein | 0.697 | 0.792 | 0.951 | 1.000 |

Dataset counts contributing to each cell:

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 11 | 11 | 11 | 11 |
| WER | 11 | 11 | 11 | 11 |
| CER | 11 | 11 | 11 | 11 |
| Levenshtein | 11 | 11 | 11 | 11 |

## All Variants

### base_valid_all_pages
|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.408 | 0.426 | 0.499 |
| WER | 0.408 | 1.000 | 0.720 | 0.670 |
| CER | 0.426 | 0.720 | 1.000 | 0.954 |
| Levenshtein | 0.499 | 0.670 | 0.954 | 1.000 |

### high_confidence_after_worst20_trim
|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.461 | 0.496 | 0.565 |
| WER | 0.461 | 1.000 | 0.785 | 0.787 |
| CER | 0.496 | 0.785 | 1.000 | 0.863 |
| Levenshtein | 0.565 | 0.787 | 0.863 | 1.000 |