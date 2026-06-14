# Full evaluation summary

- Datasets evaluated: 20
- Images evaluated: 94
- Recognition models evaluated: 7
- Per-sample comparisons: 658

## Model ranking by mean PLL
- biblia9: PLL=26207.338, WER=0.875, CER=0.453, Lev=885.515
- vat44: PLL=27677.887, WER=0.843, CER=0.421, Lev=828.394
- prenumeranten: PLL=27878.042, WER=0.999, CER=0.687, Lev=1233.788
- italian 7: PLL=28514.905, WER=0.728, CER=0.351, Lev=712.500
- sephardi: PLL=28595.697, WER=0.754, CER=0.362, Lev=730.758
- ashkenazy: PLL=29985.165, WER=0.791, CER=0.426, Lev=808.939
- sinai no voc 61: PLL=31715.842, WER=0.881, CER=0.465, Lev=890.136

## Best model by handwriting
- BNF 150 Amitai Tanchuma 2: PLL=sephardi, WER=italian 7, CER=italian 7, Lev=italian 7
- BNF 150 test 2: PLL=italian 7, WER=italian 7, CER=italian 7, Lev=italian 7
- Bodleian Oppenheim Add. 4-128: PLL=biblia9, WER=italian 7, CER=italian 7, Lev=italian 7
- Constantinople 1520: PLL=sinai no voc 61, WER=vat44, CER=vat44, Lev=ashkenazy
- Huntington Amitai Tanchuma 2: PLL=sephardi, WER=sephardi, CER=italian 7, Lev=italian 7
- Matteo Zz MS Oxford Huntington 2: PLL=sinai no voc 61, WER=italian 7, CER=italian 7, Lev=italian 7
- Midrash Hagadol Dvarim: PLL=italian 7, WER=vat44, CER=sinai no voc 61, Lev=sinai no voc 61
- Bodleian MS Huntington 115: PLL=vat44, WER=italian 7, CER=italian 7, Lev=italian 7
- Munich 229 Amitai Tanchuma 2: PLL=ashkenazy, WER=ashkenazy, CER=ashkenazy, Lev=ashkenazy
- Munich Staatsbibliothek test 2: PLL=ashkenazy, WER=vat44, CER=ashkenazy, Lev=ashkenazy
- Neubauer 147 Amitai Tanchuma 2: PLL=sinai no voc 61, WER=italian 7, CER=italian 7, Lev=italian 7
- Oppenheim Add. Fol. 3 test 2: PLL=sephardi, WER=italian 7, CER=italian 7, Lev=italian 7
- Parma 3122 test 2: PLL=sephardi, WER=vat44, CER=vat44, Lev=vat44
- Print edition Mantoue 1544: PLL=ashkenazy, WER=sephardi, CER=sinai no voc 61, Lev=sinai no voc 61
- Sefer Haikarim BNF 740 experiment: PLL=sephardi, WER=italian 7, CER=italian 7, Lev=italian 7
- Vat 44: PLL=ashkenazy, WER=sephardi, CER=sephardi, Lev=sephardi
- Vat Ebr 34: PLL=italian 7, WER=vat44, CER=italian 7, Lev=ashkenazy
- Vatican 34 Midrash Tanchuma: PLL=italian 7, WER=italian 7, CER=italian 7, Lev=italian 7
- Vatican 44 Amitai Midrash Tanchuma: PLL=biblia9, WER=biblia9, CER=biblia9, Lev=biblia9
- Vayber Taytsh experiment: PLL=sinai no voc 61, WER=sinai no voc 61, CER=sinai no voc 61, Lev=sinai no voc 61

## Pearson correlations
             pll_score    wer    cer  levenshtein
pll_score        1.000  0.053  0.050       -0.001
wer              0.053  1.000  0.838        0.450
cer              0.050  0.838  1.000        0.611
levenshtein     -0.001  0.450  0.611        1.000
