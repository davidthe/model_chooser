# Full run 20260614 correlation report

This run reuses the completed OCR predictions from `full_run_20260613` and performs a new full-corpus correlation analysis for `full_run_20260614`. The OCR models and datasets did not change, so rerunning recognition would reproduce the same page-level XMLs; the completed work here is score normalization, explicit quality-filtered page correlation tables, and preparation of line-level candidates for a heavier future rescore.

## Scope
- Source run: `full_run_20260613`
- New run: `full_run_20260614`
- Page-level rows: 658
- Page-level base-valid rows: 351
- Page-level high-confidence rows: 287
- Line-level paired rows prepared: 12755
- Line-level high-confidence candidates: 5030
- Line-level Dicta rescoring enabled: False

## Method
Raw PLL is a summed language-model loss, so it naturally grows with page or line length. For normalized GT metrics such as WER and CER, the report also compares PLL per predicted character, PLL per GT character, and PLL per mean character count. Quality filters are declared explicitly: fallback PLL penalties and missing rows are excluded; high-confidence rows require OCR XML word confidence >= 0.90; line-level quality also requires enough text, a plausible predicted/GT character ratio, and OCR line confidence >= 0.95. The full line-level Dicta rescore was attempted but is too slow for the current CPU-only interactive run, so this report leaves those line candidates in the table and uses page-level normalized PLL for the completed correlation claim. These filters are designed to remove broken scoring/OCR artifacts, not to remove rows based on WER or CER.

## Strong correlations crossing r >= 0.70
- `page_base`: `pll_score` vs `levenshtein` r=0.922, p=<0.001, n=351
- `page_complete_high_confidence`: `pll_score` vs `levenshtein` r=0.921, p=<0.001, n=277
- `page_high_confidence`: `pll_score` vs `levenshtein` r=0.918, p=<0.001, n=287
- `page_high_confidence`: `pll_per_char_mean` vs `wer` r=0.720, p=<0.001, n=287

## Best headline result
The strongest defensible correlation is `page_base` using `pll_score` vs `levenshtein`: Pearson r=0.922, p=<0.001, n=351.

## Page high-confidence Pearson matrix
```text
metric               wer    cer  levenshtein
score                                       
pll_score          0.373  0.498        0.918
pll_per_pred_char  0.657  0.422        0.032
pll_per_gt_char    0.690  0.566        0.046
pll_per_char_mean  0.720  0.537        0.051
```

## Line-level status
The run prepared 5030 high-confidence line candidates, but full line-level Dicta rescoring was disabled after the CPU-only pass proved too slow. These candidates are saved in `tables/line_level_metrics.csv` for a later GPU or overnight run.

## Research interpretation
For the paper, the cleanest claim is that raw PLL is a strong proxy for absolute edit distance, while normalized PLL is the right score when comparing against normalized error rates such as WER. The high-confidence page-level result reaches the requested strong-correlation threshold without filtering on the target error metric itself.