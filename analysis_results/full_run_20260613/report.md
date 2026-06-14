# Full handwriting report

This report summarizes the full `datasets/` corpus and compares all OCR models on every handwriting style.

## Scope
- Handwriting styles: 20
- Images: 94
- OCR models: 7
- Model-image comparisons: 658
- Valid comparisons used for statistics: 351
- Fallback-penalty rows excluded: 111
- Missing-score rows excluded: 196

## Method
For each dataset, I ran the OCR models on every image, computed PLL from the language model, and computed WER, CER, and Levenshtein distance against the ground truth text. Lower PLL is better in this pipeline. Rows with the fallback penalty or missing scores were excluded from the statistical summaries. The main correlation matrix is the equal-weight mean of the per-dataset Pearson matrices, so larger datasets do not dominate the result.

## Data quality
- BNF 150 Amitai Tanchuma 2: valid=21, penalty=0, missing=0
- BNF 150 test 2: valid=14, penalty=0, missing=0
- Bodleian Oppenheim Add. 4-128: valid=77, penalty=56, missing=0
- Constantinople 1520: valid=7, penalty=0, missing=0
- Huntington Amitai Tanchuma 2: valid=14, penalty=0, missing=56
- Matteo Zz MS Oxford Huntington 2: valid=7, penalty=14, missing=0
- Midrash Hagadol Dvarim: valid=7, penalty=0, missing=0
- Bodleian MS Huntington 115: valid=59, penalty=4, missing=0
- Munich 229 Amitai Tanchuma 2: valid=14, penalty=0, missing=0
- Munich Staatsbibliothek test 2: valid=12, penalty=2, missing=0
- Neubauer 147 Amitai Tanchuma 2: valid=14, penalty=0, missing=0
- Oppenheim Add. Fol. 3 test 2: valid=7, penalty=7, missing=0
- Parma 3122 test 2: valid=14, penalty=0, missing=0
- Print edition Mantoue 1544: valid=0, penalty=7, missing=0
- Sefer Haikarim BNF 740 experiment: valid=14, penalty=0, missing=0
- Vat 44: valid=0, penalty=7, missing=77
- Vat Ebr 34: valid=42, penalty=0, missing=7
- Vatican 34 Midrash Tanchuma: valid=14, penalty=0, missing=0
- Vatican 44 Amitai Midrash Tanchuma: valid=14, penalty=0, missing=56
- Vayber Taytsh experiment: valid=0, penalty=14, missing=0

## Overall ranking by mean PLL
- italian 7: PLL=5156.402, WER=0.723, CER=0.356, Lev=741.200
- sephardi: PLL=5250.843, WER=0.749, CER=0.369, Lev=762.300
- ashkenazy: PLL=5312.538, WER=0.769, CER=0.388, Lev=807.184
- sinai no voc 61: PLL=5625.200, WER=0.890, CER=0.464, Lev=957.604
- biblia9: PLL=5799.872, WER=0.873, CER=0.454, Lev=898.865
- vat44: PLL=5805.886, WER=0.831, CER=0.416, Lev=840.941
- prenumeranten: PLL=6326.002, WER=1.006, CER=0.689, Lev=1272.980

## Best model by handwriting style
- BNF 150 Amitai Tanchuma 2: PLL=sephardi, WER=italian 7, CER=italian 7, Lev=italian 7
- BNF 150 test 2: PLL=italian 7, WER=italian 7, CER=italian 7, Lev=italian 7
- Bodleian Oppenheim Add. 4-128: PLL=ashkenazy, WER=italian 7, CER=italian 7, Lev=italian 7
- Constantinople 1520: PLL=sinai no voc 61, WER=vat44, CER=vat44, Lev=ashkenazy
- Huntington Amitai Tanchuma 2: PLL=sephardi, WER=sephardi, CER=italian 7, Lev=italian 7
- Matteo Zz MS Oxford Huntington 2: PLL=prenumeranten, WER=italian 7, CER=italian 7, Lev=italian 7
- Midrash Hagadol Dvarim: PLL=italian 7, WER=vat44, CER=sinai no voc 61, Lev=sinai no voc 61
- Bodleian MS Huntington 115: PLL=italian 7, WER=italian 7, CER=italian 7, Lev=italian 7
- Munich 229 Amitai Tanchuma 2: PLL=ashkenazy, WER=ashkenazy, CER=ashkenazy, Lev=ashkenazy
- Munich Staatsbibliothek test 2: PLL=ashkenazy, WER=vat44, CER=ashkenazy, Lev=ashkenazy
- Neubauer 147 Amitai Tanchuma 2: PLL=sinai no voc 61, WER=italian 7, CER=italian 7, Lev=italian 7
- Oppenheim Add. Fol. 3 test 2: PLL=sephardi, WER=italian 7, CER=sephardi, Lev=italian 7
- Parma 3122 test 2: PLL=sephardi, WER=vat44, CER=vat44, Lev=vat44
- Sefer Haikarim BNF 740 experiment: PLL=sephardi, WER=italian 7, CER=italian 7, Lev=italian 7
- Vat Ebr 34: PLL=italian 7, WER=vat44, CER=italian 7, Lev=ashkenazy
- Vatican 34 Midrash Tanchuma: PLL=italian 7, WER=italian 7, CER=italian 7, Lev=italian 7
- Vatican 44 Amitai Midrash Tanchuma: PLL=biblia9, WER=biblia9, CER=biblia9, Lev=biblia9

## Mean correlation matrix across handwriting styles
             pll_score    wer    cer  levenshtein
pll_score        1.000  0.408  0.426        0.499
wer              0.408  1.000  0.720        0.670
cer              0.426  0.720  1.000        0.954
levenshtein      0.499  0.670  0.954        1.000

## Pooled correlation matrix across all rows
             pll_score    wer    cer  levenshtein
pll_score        1.000  0.325  0.447        0.922
wer              0.325  1.000  0.850        0.480
cer              0.447  0.850  1.000        0.661
levenshtein      0.922  0.480  0.661        1.000

The strongest average model is `italian 7`. The correlation results show that PLL stays positively aligned with the GT-based scores, and the equal-weight mean matrix is the fairest summary when the goal is to compare handwriting styles.