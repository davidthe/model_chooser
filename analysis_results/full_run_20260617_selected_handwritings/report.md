# Selected Handwriting Run Report

Run: `full_run_20260617_selected_handwritings`
Source run: `full_run_20260617`

This report focuses on the requested handwriting sets and reuses the completed OCR predictions.

## Selected Handwriting Sets
- BNF 150 Amitai Tanchuma 2 (`bnf150amitaitanchuma2`)
- BNF 150 test 2 (`bnf150test2`)
- Huntington Amitai Tanchuma 2 (`huntingtonamitaitanchuma2`)
- Munich Staatsbibliothek test 2 (`munichstaatsbibliothektest2`)
- Sefer Haikarim BNF 740 experiment (`seferhaikarimbnf740experimen`)
- Vatican 34 Midrash Tanchuma (`vatican34midrashtanchu`)
- Vatican 44 Amitai Midrash Tanchuma (`vatican44amitaimidrashtanchu`)

## Scope
- Selected handwriting sets: 7
- Source model-image rows: 217
- Base-valid rows: 103
- Rows used after current page-trim flag: 103
- Images used after trim: 15
- Prepared line rows for these sets: 3388

## Overall 4-Method Correlation Matrix
|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.348 | 0.245 | 0.731 |
| WER | 0.348 | 1.000 | 0.908 | 0.692 |
| CER | 0.245 | 0.908 | 1.000 | 0.742 |
| Levenshtein | 0.731 | 0.692 | 0.742 | 1.000 |

## PLL Correlation by Handwriting
| Handwriting | Pages | Rows | PLL-WER | PLL-CER | PLL-Lev | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Vatican 44 Amitai Midrash Tanchuma | 2 | 14 | 0.980 | 0.904 | 0.914 | 0.932 |
| Sefer Haikarim BNF 740 experiment | 2 | 14 | 0.876 | 0.915 | 0.913 | 0.901 |
| BNF 150 test 2 | 2 | 14 | 0.921 | 0.767 | 0.762 | 0.817 |
| Vatican 34 Midrash Tanchuma | 2 | 14 | 0.647 | 0.890 | 0.891 | 0.809 |
| BNF 150 Amitai Tanchuma 2 | 3 | 21 | 0.909 | 0.736 | 0.734 | 0.793 |
| Munich Staatsbibliothek test 2 | 2 | 12 | 0.448 | 0.650 | 0.744 | 0.614 |
| Huntington Amitai Tanchuma 2 | 2 | 14 | 0.717 | 0.453 | 0.446 | 0.539 |

## Model Ranking on Selected Sets
| Model | Mean PLL | Mean WER | Mean CER | Mean Lev | Pages | Sets |
| --- | --- | --- | --- | --- | --- | --- |
| italian 7 | 4446.721 | 0.564 | 0.287 | 395.667 | 15 | 7 |
| sephardi | 4474.706 | 0.597 | 0.294 | 424.000 | 15 | 7 |
| ashkenazy | 4576.780 | 0.611 | 0.268 | 439.200 | 15 | 7 |
| biblia9 | 5162.977 | 0.742 | 0.378 | 611.467 | 15 | 7 |
| vat44 | 5166.927 | 0.728 | 0.341 | 539.667 | 15 | 7 |
| sinai no voc 61 | 5558.636 | 0.837 | 0.393 | 749.769 | 13 | 6 |
| prenumeranten | 5784.055 | 0.998 | 0.675 | 1143.867 | 15 | 7 |

## Per-Handwriting 4-Method Matrices

### BNF 150 Amitai Tanchuma 2
Pages: 3; rows: 21

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.909 | 0.736 | 0.734 |
| WER | 0.909 | 1.000 | 0.903 | 0.899 |
| CER | 0.736 | 0.903 | 1.000 | 0.998 |
| Levenshtein | 0.734 | 0.899 | 0.998 | 1.000 |

### BNF 150 test 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.921 | 0.767 | 0.762 |
| WER | 0.921 | 1.000 | 0.919 | 0.917 |
| CER | 0.767 | 0.919 | 1.000 | 0.999 |
| Levenshtein | 0.762 | 0.917 | 0.999 | 1.000 |

### Huntington Amitai Tanchuma 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.717 | 0.453 | 0.446 |
| WER | 0.717 | 1.000 | 0.895 | 0.898 |
| CER | 0.453 | 0.895 | 1.000 | 0.999 |
| Levenshtein | 0.446 | 0.898 | 0.999 | 1.000 |

### Munich Staatsbibliothek test 2
Pages: 2; rows: 12

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.448 | 0.650 | 0.744 |
| WER | 0.448 | 1.000 | 0.765 | 0.662 |
| CER | 0.650 | 0.765 | 1.000 | 0.909 |
| Levenshtein | 0.744 | 0.662 | 0.909 | 1.000 |

### Sefer Haikarim BNF 740 experiment
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.876 | 0.915 | 0.913 |
| WER | 0.876 | 1.000 | 0.884 | 0.879 |
| CER | 0.915 | 0.884 | 1.000 | 0.998 |
| Levenshtein | 0.913 | 0.879 | 0.998 | 1.000 |

### Vatican 34 Midrash Tanchuma
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.647 | 0.890 | 0.891 |
| WER | 0.647 | 1.000 | 0.659 | 0.676 |
| CER | 0.890 | 0.659 | 1.000 | 0.953 |
| Levenshtein | 0.891 | 0.676 | 0.953 | 1.000 |

### Vatican 44 Amitai Midrash Tanchuma
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.980 | 0.904 | 0.914 |
| WER | 0.980 | 1.000 | 0.954 | 0.955 |
| CER | 0.904 | 0.954 | 1.000 | 0.998 |
| Levenshtein | 0.914 | 0.955 | 0.998 | 1.000 |