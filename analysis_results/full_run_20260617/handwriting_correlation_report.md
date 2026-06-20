# Per-Handwriting Correlation Report

Run: `full_run_20260617`

This report computes Pearson correlation matrices between the four scoring methods for each handwriting set: PLL, WER, CER, and Levenshtein. Rows are model-image comparisons. The main section uses the same worst-20% page trim from the current run, and the appendix keeps the all-page base-valid view for comparison.

## Scope
- Handwriting sets in source table: 20
- Model-image rows in source table: 658
- Base-valid rows: 351
- Pages removed by worst-20% diagnostic trim: 11

## PLL Correlation Summary
| Handwriting | Pages | Rows | PLL-WER | PLL-CER | PLL-Lev | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Vatican 44 Amitai Midrash Tanchuma | 2 | 14 | 0.980 | 0.904 | 0.914 | 0.932 |
| Sefer Haikarim BNF 740 experiment | 2 | 14 | 0.876 | 0.915 | 0.913 | 0.901 |
| BNF 150 test 2 | 2 | 14 | 0.921 | 0.767 | 0.762 | 0.817 |
| Vatican 34 Midrash Tanchuma | 2 | 14 | 0.647 | 0.890 | 0.891 | 0.809 |
| BNF 150 Amitai Tanchuma 2 | 3 | 21 | 0.909 | 0.736 | 0.734 | 0.793 |
| Munich Staatsbibliothek test 2 | 2 | 12 | 0.448 | 0.650 | 0.744 | 0.614 |
| Huntington Amitai Tanchuma 2 | 2 | 14 | 0.717 | 0.453 | 0.446 | 0.539 |
| Vat Ebr 34 | 5 | 35 | 0.023 | 0.749 | 0.712 | 0.495 |
| Bodleian MS Huntington 115 | 9 | 59 | 0.479 | 0.438 | 0.541 | 0.486 |
| Munich 229 Amitai Tanchuma 2 | 2 | 14 | 0.403 | 0.267 | 0.556 | 0.409 |
| Bodleian Oppenheim Add. 4-128 | 10 | 64 | 0.249 | 0.270 | 0.454 | 0.324 |

## Per-Handwriting Matrices After Trim

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

### Bodleian MS Huntington 115
Pages: 9; rows: 59

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.479 | 0.438 | 0.541 |
| WER | 0.479 | 1.000 | 0.903 | 0.894 |
| CER | 0.438 | 0.903 | 1.000 | 0.987 |
| Levenshtein | 0.541 | 0.894 | 0.987 | 1.000 |

### Bodleian Oppenheim Add. 4-128
Pages: 10; rows: 64

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.249 | 0.270 | 0.454 |
| WER | 0.249 | 1.000 | 0.908 | 0.867 |
| CER | 0.270 | 0.908 | 1.000 | 0.967 |
| Levenshtein | 0.454 | 0.867 | 0.967 | 1.000 |

### Huntington Amitai Tanchuma 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.717 | 0.453 | 0.446 |
| WER | 0.717 | 1.000 | 0.895 | 0.898 |
| CER | 0.453 | 0.895 | 1.000 | 0.999 |
| Levenshtein | 0.446 | 0.898 | 0.999 | 1.000 |

### Munich 229 Amitai Tanchuma 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.403 | 0.267 | 0.556 |
| WER | 0.403 | 1.000 | 0.939 | 0.906 |
| CER | 0.267 | 0.939 | 1.000 | 0.898 |
| Levenshtein | 0.556 | 0.906 | 0.898 | 1.000 |

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

### Vat Ebr 34
Pages: 5; rows: 35

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.023 | 0.749 | 0.712 |
| WER | 0.023 | 1.000 | 0.169 | 0.161 |
| CER | 0.749 | 0.169 | 1.000 | 0.753 |
| Levenshtein | 0.712 | 0.161 | 0.753 | 1.000 |

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

## Appendix: All Base-Valid Pages

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

### Bodleian MS Huntington 115
Pages: 9; rows: 59

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.479 | 0.438 | 0.541 |
| WER | 0.479 | 1.000 | 0.903 | 0.894 |
| CER | 0.438 | 0.903 | 1.000 | 0.987 |
| Levenshtein | 0.541 | 0.894 | 0.987 | 1.000 |

### Bodleian Oppenheim Add. 4-128
Pages: 12; rows: 77

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.188 | 0.181 | 0.345 |
| WER | 0.188 | 1.000 | 0.903 | 0.858 |
| CER | 0.181 | 0.903 | 1.000 | 0.967 |
| Levenshtein | 0.345 | 0.858 | 0.967 | 1.000 |

### Constantinople 1520
Pages: 1; rows: 7

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.194 | -0.473 | -0.413 |
| WER | 0.194 | 1.000 | -0.494 | -0.539 |
| CER | -0.473 | -0.494 | 1.000 | 0.996 |
| Levenshtein | -0.413 | -0.539 | 0.996 | 1.000 |

### Huntington Amitai Tanchuma 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.717 | 0.453 | 0.446 |
| WER | 0.717 | 1.000 | 0.895 | 0.898 |
| CER | 0.453 | 0.895 | 1.000 | 0.999 |
| Levenshtein | 0.446 | 0.898 | 0.999 | 1.000 |

### Matteo Zz MS Oxford Huntington 2
Pages: 1; rows: 7

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | -0.072 | -0.284 | -0.281 |
| WER | -0.072 | 1.000 | 0.937 | 0.939 |
| CER | -0.284 | 0.937 | 1.000 | 1.000 |
| Levenshtein | -0.281 | 0.939 | 1.000 | 1.000 |

### Midrash Hagadol Dvarim
Pages: 1; rows: 7

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | -0.188 | 0.457 | 0.764 |
| WER | -0.188 | 1.000 | 0.470 | 0.188 |
| CER | 0.457 | 0.470 | 1.000 | 0.893 |
| Levenshtein | 0.764 | 0.188 | 0.893 | 1.000 |

### Munich 229 Amitai Tanchuma 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.403 | 0.267 | 0.556 |
| WER | 0.403 | 1.000 | 0.939 | 0.906 |
| CER | 0.267 | 0.939 | 1.000 | 0.898 |
| Levenshtein | 0.556 | 0.906 | 0.898 | 1.000 |

### Munich Staatsbibliothek test 2
Pages: 2; rows: 12

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.448 | 0.650 | 0.744 |
| WER | 0.448 | 1.000 | 0.765 | 0.662 |
| CER | 0.650 | 0.765 | 1.000 | 0.909 |
| Levenshtein | 0.744 | 0.662 | 0.909 | 1.000 |

### Neubauer 147 Amitai Tanchuma 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.067 | -0.013 | 0.057 |
| WER | 0.067 | 1.000 | 0.901 | 0.902 |
| CER | -0.013 | 0.901 | 1.000 | 0.996 |
| Levenshtein | 0.057 | 0.902 | 0.996 | 1.000 |

### Oppenheim Add. Fol. 3 test 2
Pages: 1; rows: 7

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.442 | 0.319 | 0.316 |
| WER | 0.442 | 1.000 | 0.917 | 0.916 |
| CER | 0.319 | 0.917 | 1.000 | 1.000 |
| Levenshtein | 0.316 | 0.916 | 1.000 | 1.000 |

### Parma 3122 test 2
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.013 | 0.298 | 0.448 |
| WER | 0.013 | 1.000 | 0.659 | 0.391 |
| CER | 0.298 | 0.659 | 1.000 | 0.891 |
| Levenshtein | 0.448 | 0.391 | 0.891 | 1.000 |

### Sefer Haikarim BNF 740 experiment
Pages: 2; rows: 14

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | 0.876 | 0.915 | 0.913 |
| WER | 0.876 | 1.000 | 0.884 | 0.879 |
| CER | 0.915 | 0.884 | 1.000 | 0.998 |
| Levenshtein | 0.913 | 0.879 | 0.998 | 1.000 |

### Vat Ebr 34
Pages: 6; rows: 42

|  | PLL | WER | CER | Levenshtein |
| --- | --- | --- | --- | --- |
| PLL | 1.000 | -0.092 | 0.733 | 0.742 |
| WER | -0.092 | 1.000 | 0.124 | 0.042 |
| CER | 0.733 | 0.124 | 1.000 | 0.734 |
| Levenshtein | 0.742 | 0.042 | 0.734 | 1.000 |

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