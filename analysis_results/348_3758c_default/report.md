# Model Chooser Sample Report

This note matches the PDF report and summarizes the sample result set that was already present in the repository.

## What the data says
On `348_3758c_default`, `italian_7` is the best model across PLL, WER, CER, and Levenshtein distance. That agreement across metrics is a good sign: the PLL score is not drifting away from the GT-based measures.

## Correlation snapshot
- PLL vs WER: 0.859
- PLL vs CER: 0.813
- PLL vs Levenshtein: 0.810

## Scope
- Sample lines: 28
- Models compared: 4
- Line-level comparisons: 112

The full-corpus OCR run is still a separate job; this PDF is the clean, documented sample report.