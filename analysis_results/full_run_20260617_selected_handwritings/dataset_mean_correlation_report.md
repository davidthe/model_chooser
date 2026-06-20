# Mean Dataset Correlation Report

Run: `full_run_20260617_selected_handwritings`

This report computes a correlation matrix separately for each handwriting dataset, then averages those matrices element by element. Each dataset therefore contributes equally, unlike a pooled correlation table where datasets with more pages or model rows can dominate.

## Summary
| Variant | Datasets | PLL-WER | PLL-CER | PLL-Lev | WER-CER | CER-Lev |
| --- | --- | --- | --- | --- | --- | --- |
| base_valid_all_pages | 7 | 0.785 | 0.759 | 0.772 | 0.854 | 0.979 |
| base_valid_after_worst20_trim | 7 | 0.785 | 0.759 | 0.772 | 0.854 | 0.979 |
| high_confidence_after_worst20_trim | 7 | 0.649 | 0.676 | 0.713 | 0.845 | 0.928 |

## Recommended Mean Correlation Table
Variant: `base_valid_after_worst20_trim`

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.785 | 0.759 | 0.772 |
| WER | 0.785 | 1.000 | 0.854 | 0.841 |
| CER | 0.759 | 0.854 | 1.000 | 0.979 |
| Levenshtein | 0.772 | 0.841 | 0.979 | 1.000 |

Dataset counts contributing to each cell:

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 7 | 7 | 7 | 7 |
| WER | 7 | 7 | 7 | 7 |
| CER | 7 | 7 | 7 | 7 |
| Levenshtein | 7 | 7 | 7 | 7 |

## All Variants

### base_valid_all_pages
|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.785 | 0.759 | 0.772 |
| WER | 0.785 | 1.000 | 0.854 | 0.841 |
| CER | 0.759 | 0.854 | 1.000 | 0.979 |
| Levenshtein | 0.772 | 0.841 | 0.979 | 1.000 |

### high_confidence_after_worst20_trim
|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.649 | 0.676 | 0.713 |
| WER | 0.649 | 1.000 | 0.845 | 0.846 |
| CER | 0.676 | 0.845 | 1.000 | 0.928 |
| Levenshtein | 0.713 | 0.846 | 0.928 | 1.000 |