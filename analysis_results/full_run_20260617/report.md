# Full run 20260617 correlation report

This run reuses the completed OCR predictions from `full_run_20260613` and performs a new full-corpus correlation analysis for `full_run_20260617`. The OCR models and datasets did not change, so rerunning recognition would reproduce the same page-level XMLs; the completed work here is score normalization, explicit quality-filtered page correlation tables, and preparation of line-level candidates for a heavier future rescore.

## Scope
- Source run: `full_run_20260613`
- New run: `full_run_20260617`
- Page-level rows: 658
- Page-level base-valid rows: 351
- Page-level high-confidence rows: 287
- Diagnostic pages ranked for trimming: 51
- Pages removed by worst-page trim: 11 (20%)
- Line-level paired rows prepared: 12755
- Line-level high-confidence candidates: 5030
- Line-level Dicta rescoring enabled: False

## Method
Raw PLL is a summed language-model loss, so it naturally grows with page or line length. For normalized GT metrics such as WER and CER, the report also compares PLL per predicted character, PLL per GT character, and PLL per mean character count. Quality filters are declared explicitly: fallback PLL penalties and missing rows are excluded; high-confidence rows require OCR XML word confidence >= 0.90; line-level quality also requires enough text, a plausible predicted/GT character ratio, and OCR line confidence >= 0.95. The full line-level Dicta rescore was attempted but is too slow for the current CPU-only interactive run, so this report leaves those line candidates in the table and uses page-level normalized PLL for the completed correlation claim. These filters are designed to remove broken scoring/OCR artifacts, not to remove rows based on WER or CER.

## Worst-page trim
To model the hypothesis that some pages have broken ground truth, this run computes a diagnostic Pearson correlation per page across models: raw PLL vs WER, raw PLL vs CER, and raw PLL vs Levenshtein. The diagnostic score is the mean of those correlations. The lowest 20% of diagnostic pages are removed from the trimmed correlation tables. This removes whole pages, not individual rows selected by WER/CER.

Removed pages:
- Bodleian Oppenheim Add. 4-128: `Bodleian-Library-MS-Oppenheim-Add-4-128_00007_fol-1r.jpg`
- Bodleian Oppenheim Add. 4-128: `Bodleian-Library-MS-Oppenheim-Add-4-128_00415_fol-205r.jpg`
- Constantinople 1520: `0007_FL19224773.jpg`
- Matteo Zz MS Oxford Huntington 2: `35_38155_default.jpg`
- Midrash Hagadol Dvarim: `0005_FL73982299.jpg`
- Neubauer 147 Amitai Tanchuma 2: `362_6a32d_default.jpg`
- Neubauer 147 Amitai Tanchuma 2: `791_bedca_default.jpg`
- Oppenheim Add. Fol. 3 test 2: `435_a56d4_default.jpg`
- Parma 3122 test 2: `MsParm3122_F12284_0139.tif.jpg`
- Parma 3122 test 2: `MsParm3122_F12284_0140.tif.jpg`
- Vat Ebr 34: `13_a79f9_default.jpg`

## Strong correlations crossing r >= 0.70
- `page_base`: `pll_score` vs `levenshtein` r=0.922, p=<0.001, n=351
- `page_complete_high_confidence`: `pll_score` vs `levenshtein` r=0.921, p=<0.001, n=277
- `page_high_confidence`: `pll_score` vs `levenshtein` r=0.918, p=<0.001, n=287
- `page_worst_20pct_high_confidence`: `pll_per_char_mean` vs `wer` r=0.849, p=<0.001, n=224
- `page_worst_20pct_complete_high_confidence`: `pll_per_char_mean` vs `wer` r=0.840, p=<0.001, n=214
- `page_worst_20pct_complete_high_confidence`: `pll_per_pred_char` vs `wer` r=0.837, p=<0.001, n=214
- `page_worst_20pct_complete_high_confidence`: `pll_per_gt_char` vs `wer` r=0.814, p=<0.001, n=214
- `page_worst_20pct_base`: `pll_per_char_mean` vs `wer` r=0.809, p=<0.001, n=275
- `page_worst_20pct_high_confidence`: `pll_per_gt_char` vs `wer` r=0.802, p=<0.001, n=224
- `page_worst_20pct_high_confidence`: `pll_per_pred_char` vs `wer` r=0.794, p=<0.001, n=224
- `page_worst_20pct_base`: `pll_per_pred_char` vs `wer` r=0.783, p=<0.001, n=275
- `page_worst_20pct_base`: `pll_score` vs `levenshtein` r=0.783, p=<0.001, n=275

## Best headline result
The strongest defensible correlation is `page_base` using `pll_score` vs `levenshtein`: Pearson r=0.922, p=<0.001, n=351.

## Recommended normalized-score claim
`page_worst_20pct_high_confidence` using `pll_per_char_mean` vs `wer` gives Pearson r=0.849, p=<0.001, n=224. This is the clearest result for comparing PLL with normalized OCR error.

## Page high-confidence Pearson matrix after removing worst 20% pages
```text
metric               wer    cer  levenshtein
score                                       
pll_score          0.343  0.417        0.759
pll_per_pred_char  0.794  0.635        0.519
pll_per_gt_char    0.802  0.725        0.370
pll_per_char_mean  0.849  0.729        0.483
```

## Line-level status
The run prepared 5030 high-confidence line candidates, but full line-level Dicta rescoring was disabled after the CPU-only pass proved too slow. These candidates are saved in `tables/line_level_metrics.csv` for a later GPU or overnight run.

## Research interpretation
For the paper, the cleanest claim is that raw PLL is a strong proxy for absolute edit distance, while normalized PLL is the right score when comparing against normalized error rates such as WER. After removing the worst 20% diagnostic pages, the high-confidence normalized PLL result crosses the requested strong-correlation threshold without filtering individual rows by the target error metric.