# Full run 20260617 summary

- Page rows: 658
- Base-valid page rows: 351
- High-confidence page rows: 287
- Pages removed by worst-page trim: 11
- Line rows prepared: 12755
- High-confidence line candidates: 5030
- Line-level Dicta rescoring enabled: False

## Best correlations crossing r >= 0.70
- page_base | pll_score vs levenshtein: r=0.922, p=<0.001, n=351
- page_complete_high_confidence | pll_score vs levenshtein: r=0.921, p=<0.001, n=277
- page_high_confidence | pll_score vs levenshtein: r=0.918, p=<0.001, n=287
- page_worst_20pct_high_confidence | pll_per_char_mean vs wer: r=0.849, p=<0.001, n=224
- page_worst_20pct_complete_high_confidence | pll_per_char_mean vs wer: r=0.840, p=<0.001, n=214
- page_worst_20pct_complete_high_confidence | pll_per_pred_char vs wer: r=0.837, p=<0.001, n=214
- page_worst_20pct_complete_high_confidence | pll_per_gt_char vs wer: r=0.814, p=<0.001, n=214
- page_worst_20pct_base | pll_per_char_mean vs wer: r=0.809, p=<0.001, n=275