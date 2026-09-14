# Rule-based prototypes

Two rule families, both learned on TRAIN only and scored on the held-out family-disjoint TEST split from `cnn_vit_pipeline/cohort.py`:

**(a) mined mnemonic n-gram rules.** 2-, 3- and 4-grams over the `mn/` mnemonic streams (first 30,000 mnemonics per file), mined level-wise with Apriori downward closure at document support >= 40 training files, scored by lift and odds ratio toward ransomware under a Jeffreys (alpha=0.5) prior, then reduced to a compact set by greedy set cover with a train-precision floor of 0.80.

**(b) hand-written behaviour signatures.** 23 crypto-loop, enumeration and anti-analysis patterns over mnemonics and operands in the `.asm` (first 80,000 instructions). Presence rules have no free parameter; density rules get one threshold, picked on train by maximising that single rule's own F1.

Decision thresholds for the weighted scorers come from the **val fold** (`cohort.add_val_fold`, 10% of train held out by group), not from the rows the rules were mined on and not from test.

**(c)** one calibration baseline -- mnemonic 1-3-gram TF-IDF + linear SVM / logistic regression -- is reported alongside, because a rule set is only interesting relative to the cheapest strong classical model on the same tokens. Section 5 audits it, because it turned out to be the best result in the whole project.

> **Correction, and what it invalidates.** An earlier version of these tables was wrong. `MnemCorpus` stores each file's mnemonic stream as `int16` to keep the corpus in memory; `ngram_rules._codes` then computed n-gram codes by `c * 2048 + next`, which **overflows int16 silently** for any leading mnemonic whose id is 16 or above. The miner cast to `int64` first and was correct; the scorer (`hit_matrix`) did not. So every mined rule was scored against a wrapped-around code: all 3- and 4-grams matched nothing at all, and 2-grams matched only when their first mnemonic was one of the 15 earliest-seen in the corpus (`add sub cmp inc push mov ...`). The visible symptom was a **top-by-lift table in which every rule showed zero test hits** -- including `xorps movlpd`, present in 264 of 796 training ransomware and 4 of 1,003 training goodware, which cannot plausibly fire on none of 491 test files. The cast now lives inside `_codes` and `tests/test_graph2vec_rules.py` pins it (`test_codes_are_dtype_independent`, `test_hit_matrix_fires_for_high_id_mnemonics`, `test_hit_matrix_agrees_with_mining_on_int16_corpus`, which re-derives every mined document frequency from the hit matrix). Every `ngram_rules`, `behaviour_rules`-weighted and `combined` number below is from the rerun; the `calibration_baseline` rows never touched the hit matrix and are unchanged.

## 1. Test results

### mendeley

test 491 (129 goodware / 362 ransomware); majority-class accuracy 0.7373.

| track | model | rules | acc | bal_acc | macro_f1 | auc | recall_ran | fpr |
|---|---|---|---|---|---|---|---|---|
| ngram_rules | any_hit | 14 | 0.7984 | 0.6362 | 0.6558 | 0.6362 | 0.9779 | 0.7054 |
| ngram_rules | rule_votes | 36 | 0.7882 | 0.8189 | 0.7634 | 0.7891 | 0.7541 | 0.1163 |
| ngram_rules | weighted_logodds | 36 | 0.7943 | 0.8256 | 0.7699 | 0.7887 | 0.7597 | 0.1085 |
| behaviour_rules | crypto_signature_any_hit | 5 | 0.3422 | 0.2944 | 0.3018 | 0.2944 | 0.3950 | 0.8062 |
| behaviour_rules | weighted_logodds | 23 | 0.7739 | 0.7294 | 0.7197 | 0.6950 | 0.8232 | 0.3643 |
| combined | behaviour+ngram_rules_logreg | 59 | 0.8208 | 0.8485 | 0.7967 | 0.8934 | 0.7901 | 0.0930 |
| calibration_baseline | mnemonic_tfidf_1_3+LinearSVC | 74738 | 0.9695 | 0.9568 | 0.9603 | 0.9778 | 0.9834 | 0.0698 |
| calibration_baseline | mnemonic_tfidf_1_3+LogReg | 74738 | 0.9756 | 0.9610 | 0.9680 | 0.9871 | 0.9917 | 0.0698 |

### balanced

test 489 (127 goodware / 362 ransomware); majority-class accuracy 0.7403.

| track | model | rules | acc | bal_acc | macro_f1 | auc | recall_ran | fpr |
|---|---|---|---|---|---|---|---|---|
| ngram_rules | any_hit | 12 | 0.7444 | 0.6357 | 0.6436 | 0.6357 | 0.8619 | 0.5906 |
| ngram_rules | rule_votes | 41 | 0.5256 | 0.6336 | 0.5225 | 0.7569 | 0.4088 | 0.1417 |
| ngram_rules | weighted_logodds | 41 | 0.5358 | 0.6098 | 0.5266 | 0.7392 | 0.4558 | 0.2362 |
| behaviour_rules | crypto_signature_any_hit | 5 | 0.5051 | 0.6070 | 0.5019 | 0.6070 | 0.3950 | 0.1811 |
| behaviour_rules | weighted_logodds | 23 | 0.5808 | 0.6581 | 0.5704 | 0.7807 | 0.4972 | 0.1811 |
| combined | behaviour+ngram_rules_logreg | 64 | 0.6953 | 0.7303 | 0.6698 | 0.8302 | 0.6575 | 0.1969 |
| calibration_baseline | mnemonic_tfidf_1_3+LinearSVC | 86677 | 0.8282 | 0.8508 | 0.8023 | 0.9285 | 0.8039 | 0.1024 |
| calibration_baseline | mnemonic_tfidf_1_3+LogReg | 86677 | 0.8221 | 0.8441 | 0.7956 | 0.9226 | 0.7983 | 0.1102 |


## 2. The selected n-gram rule set

### mendeley

81,127 n-grams cleared support ({'2': 6987, '3': 25525, '4': 48615}); 3,282 kept as candidates; greedy set cover selected 14 ransomware-leaning and 22 goodware-leaning rules, covering 791/796 training ransomware.


**Selected ransomware rules**

| rule (mnemonic n-gram) | n | train R/G docs | lift | odds | train P | train R | test P | test R |
|---|---|---|---|---|---|---|---|---|
| `adc mov mov` | 3 | 592/88 | 8.4300 | 29.9700 | 0.8710 | 0.7440 | 0.8028 | 0.6298 |
| `int3 push mov push` | 4 | 617/150 | 5.1700 | 19.5100 | 0.8040 | 0.7750 | 0.8916 | 0.6133 |
| `add adc mov` | 3 | 557/121 | 5.7800 | 16.9100 | 0.8220 | 0.7000 | 0.7840 | 0.5414 |
| `add test jne push` | 4 | 569/141 | 5.0700 | 15.2600 | 0.8010 | 0.7150 | 0.9418 | 0.4917 |
| `mov adc mov` | 3 | 522/79 | 8.2800 | 22.1400 | 0.8690 | 0.6560 | 0.9576 | 0.6243 |
| `mov mov adc` | 3 | 493/59 | 10.4500 | 25.8100 | 0.8930 | 0.6190 | 0.9488 | 0.5635 |
| `mul add` | 2 | 513/101 | 6.3700 | 16.1100 | 0.8360 | 0.6440 | 0.8694 | 0.6436 |
| `sbb mov mov mov` | 4 | 484/53 | 11.4100 | 27.5400 | 0.9010 | 0.6080 | 0.9202 | 0.4779 |
| `mov xor div mov` | 4 | 496/94 | 6.6200 | 15.9000 | 0.8410 | 0.6230 | 0.8559 | 0.5414 |
| `add mov adc` | 3 | 492/114 | 5.4200 | 12.5600 | 0.8120 | 0.6180 | 0.9444 | 0.6575 |
| `push call add call` | 4 | 452/95 | 5.9700 | 12.5000 | 0.8260 | 0.5680 | 0.9371 | 0.4116 |
| `xor sub sbb` | 3 | 374/26 | 17.8000 | 32.7000 | 0.9350 | 0.4700 | 0.8986 | 0.1713 |
| `add mov mov shr` | 4 | 361/60 | 7.5300 | 12.9500 | 0.8570 | 0.4540 | 0.8400 | 0.3481 |
| `dec or inc` | 3 | 278/51 | 6.8100 | 9.9300 | 0.8450 | 0.3490 | 0.9688 | 0.2569 |


**Selected goodware rules**

| rule (mnemonic n-gram) | n | train R/G docs | lift | odds | train P | train R | test P | test R |
|---|---|---|---|---|---|---|---|---|
| `int3 mov push push` | 4 | 103/677 | 0.1900 | 0.0700 | 0.8680 | 0.6750 | 0.1019 | 0.1240 |
| `neg sbb not and` | 4 | 39/631 | 0.0800 | 0.0300 | 0.9420 | 0.6290 | 0.8101 | 0.4961 |
| `neg sbb not` | 3 | 43/633 | 0.0900 | 0.0300 | 0.9360 | 0.6310 | 0.8125 | 0.5039 |
| `xor lea xor` | 3 | 68/634 | 0.1400 | 0.0500 | 0.9030 | 0.6320 | 0.3922 | 0.1550 |
| `and jmp xor` | 3 | 162/684 | 0.3000 | 0.1200 | 0.8090 | 0.6820 | 0.4662 | 0.5349 |
| `call jmp mov test` | 4 | 140/663 | 0.2700 | 0.1100 | 0.8260 | 0.6610 | 0.2563 | 0.4729 |
| `je call jmp` | 3 | 149/667 | 0.2800 | 0.1200 | 0.8170 | 0.6650 | 0.8306 | 0.7984 |
| `int3 xor ret` | 3 | 101/634 | 0.2000 | 0.0800 | 0.8630 | 0.6320 | 0.1059 | 0.0698 |
| `call jmp mov call` | 4 | 160/653 | 0.3100 | 0.1400 | 0.8030 | 0.6510 | 0.3583 | 0.5194 |
| `xor inc cmp jne` | 4 | 124/624 | 0.2500 | 0.1100 | 0.8340 | 0.6220 | 0.6897 | 0.4651 |
| `call mov lea call` | 4 | 141/623 | 0.2900 | 0.1300 | 0.8150 | 0.6210 | 0.2785 | 0.5116 |
| `mov lea call test` | 4 | 129/607 | 0.2700 | 0.1300 | 0.8250 | 0.6050 | 0.1169 | 0.2093 |
| `xor call jmp` | 3 | 43/524 | 0.1000 | 0.0500 | 0.9240 | 0.5220 | 0.2708 | 0.2016 |
| `call lea lea call` | 4 | 49/527 | 0.1200 | 0.0600 | 0.9150 | 0.5250 | 0.3931 | 0.4419 |
| `xor lock` | 2 | 33/497 | 0.0800 | 0.0400 | 0.9380 | 0.4960 | 0.0769 | 0.1008 |
| `lea ret` | 2 | 131/538 | 0.3100 | 0.1700 | 0.8040 | 0.5360 | 0.2924 | 0.6822 |
| `js lea` | 2 | 124/526 | 0.3000 | 0.1700 | 0.8090 | 0.5240 | 0.1786 | 0.1550 |
| `lea mov lea call` | 4 | 112/487 | 0.2900 | 0.1700 | 0.8130 | 0.4860 | 0.1775 | 0.2326 |
| `jmp lea mov call` | 4 | 90/471 | 0.2400 | 0.1400 | 0.8400 | 0.4700 | 0.3144 | 0.4729 |
| `xor mov add pop` | 4 | 63/456 | 0.1800 | 0.1000 | 0.8790 | 0.4550 | 0.4069 | 0.4574 |
| `push sub and` | 3 | 19/423 | 0.0600 | 0.0300 | 0.9570 | 0.4220 | 0.2000 | 0.0310 |
| `call nop` | 2 | 72/430 | 0.2100 | 0.1300 | 0.8570 | 0.4290 | 0.1401 | 0.1705 |


**Top 15 by lift toward ransomware.** Not the selected set: greedy set cover optimises coverage at a precision floor, lift ranks purity. Because support is floored at 40 training documents, high lift here does *not* mean a rare n-gram -- the top row is present in 264 of the training ransomware. **15 of these 15 rules fire on the test split** (before the `_codes` fix it was 0 of 15, which is what the correction at the top of this file is about):

| rule | n | train R/G docs | lift | train P | train R | test P | test R |
|---|---|---|---|---|---|---|---|
| `xorps movlpd` | 2 | 264/4 | 74.0400 | 0.9850 | 0.3320 | 0.9851 | 0.3646 |
| `mov shr mov shl` | 4 | 334/12 | 33.7100 | 0.9650 | 0.4200 | 0.9223 | 0.2624 |
| `call add add lea` | 4 | 274/11 | 30.0700 | 0.9610 | 0.3440 | 0.7000 | 0.0387 |
| `nop push push` | 3 | 330/19 | 21.3500 | 0.9460 | 0.4150 | 0.8182 | 0.4972 |
| `mov rol mov` | 3 | 278/17 | 20.0500 | 0.9420 | 0.3490 | 0.9741 | 0.3122 |
| `xor sub sbb` | 3 | 374/26 | 17.8000 | 0.9350 | 0.4700 | 0.8986 | 0.1713 |
| `mov sbb mov mov` | 4 | 297/21 | 17.4300 | 0.9340 | 0.3730 | 0.9020 | 0.3812 |
| `rep movsb` | 2 | 351/26 | 16.7100 | 0.9310 | 0.4410 | 0.7960 | 0.4420 |
| `jmp pop mov pop` | 4 | 340/26 | 16.1900 | 0.9290 | 0.4270 | 0.8051 | 0.2624 |
| `adc mov mov mov` | 4 | 512/42 | 15.1900 | 0.9240 | 0.6430 | 0.9043 | 0.5746 |
| `mov mov mov adc` | 4 | 279/23 | 14.9800 | 0.9240 | 0.3510 | 0.9626 | 0.2845 |
| `shr or mov mov` | 4 | 276/24 | 14.2200 | 0.9200 | 0.3470 | 0.9083 | 0.3011 |
| `mov adc mov mov` | 4 | 454/46 | 12.3100 | 0.9080 | 0.5700 | 0.9641 | 0.5939 |
| `xor xor ret` | 3 | 274/29 | 11.7200 | 0.9040 | 0.3440 | 0.9595 | 0.1961 |
| `sbb mov mov mov` | 4 | 484/53 | 11.4100 | 0.9010 | 0.6080 | 0.9202 | 0.4779 |

### balanced

81,584 n-grams cleared support ({'2': 7429, '3': 26008, '4': 48147}); 3,449 kept as candidates; greedy set cover selected 12 ransomware-leaning and 29 goodware-leaning rules, covering 791/796 training ransomware.


**Selected ransomware rules**

| rule (mnemonic n-gram) | n | train R/G docs | lift | odds | train P | train R | test P | test R |
|---|---|---|---|---|---|---|---|---|
| `add push call` | 3 | 746/112 | 8.3500 | 117.0100 | 0.8690 | 0.9370 | 0.8784 | 0.7182 |
| `mov push call` | 3 | 759/133 | 7.1600 | 131.9100 | 0.8510 | 0.9540 | 0.8693 | 0.7901 |
| `push call add` | 3 | 757/132 | 7.1900 | 125.9900 | 0.8520 | 0.9510 | 0.8648 | 0.7597 |
| `call pop` | 2 | 763/146 | 6.5600 | 133.2500 | 0.8390 | 0.9590 | 0.8545 | 0.7790 |
| `sub push` | 2 | 763/147 | 6.5100 | 132.1900 | 0.8380 | 0.9590 | 0.8631 | 0.8011 |
| `lea push` | 2 | 767/153 | 6.2900 | 143.9800 | 0.8340 | 0.9640 | 0.8504 | 0.8011 |
| `jne push` | 2 | 753/167 | 5.6600 | 86.4000 | 0.8180 | 0.9460 | 0.8675 | 0.7956 |
| `adc mov` | 2 | 672/127 | 6.6400 | 37.0900 | 0.8410 | 0.8440 | 0.8831 | 0.7514 |
| `xor push` | 2 | 691/157 | 5.5300 | 35.1900 | 0.8150 | 0.8680 | 0.8683 | 0.8011 |
| `jb push` | 2 | 646/103 | 7.8600 | 37.3300 | 0.8620 | 0.8120 | 0.9034 | 0.7238 |
| `pop push` | 2 | 634/157 | 5.0700 | 20.9600 | 0.8020 | 0.7960 | 0.8243 | 0.5442 |
| `rep stosb` | 2 | 451/85 | 6.6500 | 14.0200 | 0.8410 | 0.5670 | 0.8736 | 0.4199 |


**Selected goodware rules**

| rule (mnemonic n-gram) | n | train R/G docs | lift | odds | train P | train R | test P | test R |
|---|---|---|---|---|---|---|---|---|
| `mov add pop jmp` | 4 | 10/776 | 0.0200 | 0.0000 | 0.9870 | 0.7740 | 0.7879 | 0.6142 |
| `movsxd mov` | 2 | 20/755 | 0.0300 | 0.0100 | 0.9740 | 0.7530 | 0.5221 | 0.5591 |
| `mov movabs` | 2 | 17/752 | 0.0300 | 0.0100 | 0.9780 | 0.7500 | 0.5000 | 0.5512 |
| `call nop` | 2 | 72/792 | 0.1200 | 0.0300 | 0.9170 | 0.7900 | 0.3662 | 0.6142 |
| `mov movsxd` | 2 | 20/748 | 0.0300 | 0.0100 | 0.9740 | 0.7470 | 0.5357 | 0.5906 |
| `lea mov call test` | 4 | 203/842 | 0.3000 | 0.0700 | 0.8060 | 0.8400 | 0.3421 | 0.7165 |
| `mov add pop ret` | 4 | 169/818 | 0.2600 | 0.0600 | 0.8290 | 0.8160 | 0.5959 | 0.6850 |
| `sub mov mov call` | 4 | 192/815 | 0.3000 | 0.0700 | 0.8090 | 0.8130 | 0.3080 | 0.6063 |
| `xor add pop` | 3 | 86/738 | 0.1500 | 0.0400 | 0.8960 | 0.7370 | 0.4211 | 0.5669 |
| `add pop jmp` | 3 | 184/787 | 0.2900 | 0.0800 | 0.8110 | 0.7850 | 0.6439 | 0.6693 |
| `lea call jmp` | 3 | 183/781 | 0.3000 | 0.0800 | 0.8100 | 0.7790 | 0.2202 | 0.6535 |
| `lea lea mov call` | 4 | 97/718 | 0.1700 | 0.0600 | 0.8810 | 0.7170 | 0.3709 | 0.6220 |
| `sub lea mov mov` | 4 | 139/743 | 0.2400 | 0.0700 | 0.8420 | 0.7420 | 0.2710 | 0.5591 |
| `mov call lea call` | 4 | 186/769 | 0.3100 | 0.0900 | 0.8050 | 0.7670 | 0.4187 | 0.6693 |
| `lea call test jne` | 4 | 189/768 | 0.3100 | 0.1000 | 0.8030 | 0.7660 | 0.2857 | 0.6299 |
| `push sub mov call` | 4 | 127/728 | 0.2200 | 0.0700 | 0.8510 | 0.7270 | 0.4220 | 0.5748 |
| `push sub lea` | 3 | 190/760 | 0.3200 | 0.1000 | 0.8000 | 0.7580 | 0.4221 | 0.6614 |
| `add pop ret mov` | 4 | 156/737 | 0.2700 | 0.0900 | 0.8250 | 0.7360 | 0.4055 | 0.6929 |
| `lea ret` | 2 | 131/719 | 0.2300 | 0.0800 | 0.8460 | 0.7180 | 0.2655 | 0.6063 |
| `call mov lea call` | 4 | 141/721 | 0.2500 | 0.0800 | 0.8360 | 0.7200 | 0.3187 | 0.6299 |
| `lea mov lea lea` | 4 | 149/724 | 0.2600 | 0.0900 | 0.8290 | 0.7230 | 0.3140 | 0.5984 |
| `test cmove` | 2 | 152/725 | 0.2600 | 0.0900 | 0.8270 | 0.7240 | 0.3739 | 0.6772 |
| `lock cmpxchg` | 2 | 53/657 | 0.1000 | 0.0400 | 0.9250 | 0.6560 | 0.2595 | 0.5354 |
| `int3 mov mov mov` | 4 | 158/711 | 0.2800 | 0.1000 | 0.8180 | 0.7100 | 0.3022 | 0.6614 |
| `mov call jmp lea` | 4 | 61/604 | 0.1300 | 0.0600 | 0.9080 | 0.6030 | 0.4641 | 0.5591 |
| `je call jmp` | 3 | 149/642 | 0.2900 | 0.1300 | 0.8120 | 0.6410 | 0.7900 | 0.6220 |
| `je add pop` | 3 | 81/485 | 0.2100 | 0.1200 | 0.8570 | 0.4840 | 0.6000 | 0.3307 |
| `lea shl add` | 3 | 31/426 | 0.0900 | 0.0600 | 0.9320 | 0.4250 | 0.1837 | 0.2835 |
| `jg lea` | 2 | 96/438 | 0.2800 | 0.1800 | 0.8200 | 0.4370 | 0.2249 | 0.3701 |


**Top 15 by lift toward ransomware.** Not the selected set: greedy set cover optimises coverage at a precision floor, lift ranks purity. Because support is floored at 40 training documents, high lift here does *not* mean a rare n-gram -- the top row is present in 374 of the training ransomware. **15 of these 15 rules fire on the test split** (before the `_codes` fix it was 0 of 15, which is what the correction at the top of this file is about):

| rule | n | train R/G docs | lift | train P | train R | test P | test R |
|---|---|---|---|---|---|---|---|
| `xor sub sbb` | 3 | 374/11 | 40.9800 | 0.9710 | 0.4700 | 0.9394 | 0.1713 |
| `shr rcr` | 2 | 327/13 | 30.5300 | 0.9620 | 0.4110 | 0.9111 | 0.2265 |
| `div mov mov div` | 4 | 282/12 | 28.4400 | 0.9590 | 0.3540 | 0.9111 | 0.2265 |
| `add lea push mov` | 4 | 327/14 | 28.4200 | 0.9590 | 0.4110 | 0.9346 | 0.2762 |
| `mov mul add jb` | 4 | 281/12 | 28.3400 | 0.9590 | 0.3530 | 0.9111 | 0.2265 |
| `jb cmp jbe dec` | 4 | 281/12 | 28.3400 | 0.9590 | 0.3530 | 0.9111 | 0.2265 |
| `div mov mul mov` | 4 | 281/12 | 28.3400 | 0.9590 | 0.3530 | 0.9111 | 0.2265 |
| `call add push lea` | 4 | 550/24 | 28.2800 | 0.9580 | 0.6910 | 0.8649 | 0.1768 |
| `sub mov lea push` | 4 | 360/16 | 27.5000 | 0.9570 | 0.4520 | 0.8519 | 0.1271 |
| `mov shr rcr` | 3 | 287/13 | 26.8000 | 0.9570 | 0.3610 | 0.9111 | 0.2265 |
| `mov mov shr rcr` | 4 | 286/13 | 26.7100 | 0.9570 | 0.3590 | 0.9111 | 0.2265 |
| `rcr shr` | 2 | 285/13 | 26.6100 | 0.9560 | 0.3580 | 0.9111 | 0.2265 |
| `shr rcr shr rcr` | 4 | 283/13 | 26.4300 | 0.9560 | 0.3560 | 0.9111 | 0.2265 |
| `rcr shr rcr` | 3 | 283/13 | 26.4300 | 0.9560 | 0.3560 | 0.9111 | 0.2265 |
| `shr rcr shr` | 3 | 283/13 | 26.4300 | 0.9560 | 0.3560 | 0.9111 | 0.2265 |


## 3. Behaviour signatures, per rule

### mendeley

| id | rule | kind | thr | fires on ransomware (test) | fires on goodware (test) | train P | train R | test P | test R | log-odds weight |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | aesni_any | presence | - | 0.0000 | 0.0078 | 0.9901 | 0.1256 | 0.0000 | 0.0000 | 4.5690 |
| R02 | pclmulqdq_any | presence | - | 0.0000 | 0.0000 | 1.0000 | 0.0013 | 0.0000 | 0.0000 | 1.3310 |
| R03 | sha_ni_any | presence | - | 0.0304 | 0.0000 | 1.0000 | 0.0477 | 1.0000 | 0.0304 | 4.6240 |
| R04 | crc32_any | presence | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.8690 |
| R05 | rdrand_any | presence | - | 0.0000 | 0.0000 | 0.9839 | 0.0766 | 0.0000 | 0.0000 | 4.0230 |
| R06 | crypto_const_any | presence | - | 0.3646 | 0.8062 | 0.7445 | 0.6407 | 0.5593 | 0.3646 | 2.1300 |
| R07 | loop_instr_any | presence | - | 0.4254 | 0.4806 | 0.5460 | 0.4849 | 0.7130 | 0.4254 | 0.6930 |
| R08 | cpuid_any | presence | - | 0.8039 | 0.1938 | 0.7476 | 0.4799 | 0.9209 | 0.8039 | 1.8300 |
| R09 | rdtsc_any | presence | - | 0.0276 | 0.0155 | 0.8182 | 0.2148 | 0.8333 | 0.0276 | 1.9280 |
| R10 | peb_teb_any | presence | - | 0.6657 | 0.0620 | 0.4925 | 0.4535 | 0.9679 | 0.6657 | 0.3420 |
| R11 | syscall_any | presence | - | 0.0193 | 0.0930 | 0.8824 | 0.0565 | 0.3684 | 0.0193 | 2.2290 |
| R12 | int_other_any | presence | - | 0.7127 | 0.2713 | 0.5178 | 0.5289 | 0.8805 | 0.7127 | 0.5590 |
| R13 | xor_dense | density | 11.0432 | 0.8619 | 0.9845 | 0.4189 | 0.9020 | 0.7107 | 0.8619 | -2.6750 |
| R14 | rotshift_dense | density | 5.7158 | 0.7956 | 0.7597 | 0.6990 | 0.8693 | 0.7461 | 0.7956 | 2.7510 |
| R15 | tight_crypto_loop | density | 0.3394 | 0.9282 | 1.0000 | 0.4389 | 0.9422 | 0.7226 | 0.9282 | -0.2900 |
| R16 | arx_window | density | 0.1500 | 0.8260 | 0.9767 | 0.5679 | 0.9083 | 0.7035 | 0.8260 | 2.0930 |
| R17 | rep_string_dense | density | 0.0000 | 1.0000 | 1.0000 | 0.4425 | 1.0000 | 0.7373 | 1.0000 | -0.2310 |
| R18 | scan_string_dense | density | 0.0000 | 1.0000 | 1.0000 | 0.4425 | 1.0000 | 0.7373 | 1.0000 | -0.2310 |
| R19 | int3_dense | density | 0.0000 | 1.0000 | 1.0000 | 0.4425 | 1.0000 | 0.7373 | 1.0000 | -0.2310 |
| R20 | call_dense | density | 27.0250 | 0.9227 | 1.0000 | 0.4213 | 0.9083 | 0.7214 | 0.9227 | -2.2630 |
| R21 | indirect_call_ratio | density | 0.0752 | 0.6298 | 0.6899 | 0.4213 | 0.9045 | 0.7192 | 0.6298 | -1.9800 |
| R22 | simd_dense | density | 0.1528 | 0.9669 | 0.3101 | 0.5435 | 0.7374 | 0.8974 | 0.9669 | 1.0650 |
| R23 | backedge_dense | density | 5.0157 | 0.8757 | 0.9845 | 0.4260 | 0.9146 | 0.7140 | 0.8757 | -1.4110 |


What each rule means:

- **R01 aesni_any** -- AES-NI round instruction (aesenc/aesdec/aeskeygenassist) present
- **R02 pclmulqdq_any** -- carry-less multiply present (AES-GCM / CRC / GF(2^n) math)
- **R03 sha_ni_any** -- SHA extension instruction (sha1rnds4/sha256rnds2/...) present
- **R04 crc32_any** -- SSE4.2 crc32 instruction present
- **R05 rdrand_any** -- hardware RNG (rdrand/rdseed) present -- key generation
- **R06 crypto_const_any** -- MD5/SHA-1/SHA-256 IV, SHA-1 K, CRC-32 polynomial, FNV or TEA delta appears as an immediate
- **R07 loop_instr_any** -- x86 `loop`/`loope`/`loopne` present (rare in modern compiler output)
- **R08 cpuid_any** -- cpuid present -- feature probe or VM/sandbox detection
- **R09 rdtsc_any** -- rdtsc/rdtscp present -- timing-based anti-analysis or seeding
- **R10 peb_teb_any** -- direct PEB/TEB access (fs:[0x30], fs:[0x18], gs:[0x60])
- **R11 syscall_any** -- syscall/sysenter in user code -- ntdll bypass
- **R12 int_other_any** -- software interrupt other than int3 (int 0x2d/0x2e anti-debug)
- **R13 xor_dense** -- xor instructions per 1k decoded instructions
- **R14 rotshift_dense** -- rol/ror/shl/shr/sar/shld/shrd per 1k -- bit-mixing chains
- **R15 tight_crypto_loop** -- backward branches spanning <=40 instructions over a logic/shift heavy body, per 1k -- the crypto inner-loop shape
- **R16 arx_window** -- 16-instruction windows that are both logic-heavy and arith- or rotate-heavy (ARX round shape), per 1k
- **R17 rep_string_dense** -- rep movs/stos per 1k -- bulk buffer copy/fill
- **R18 scan_string_dense** -- scas/cmps per 1k -- string scanning, path/extension enumeration
- **R19 int3_dense** -- int3 per 1k -- alignment padding density, a linker/packer artefact
- **R20 call_dense** -- call instructions per 1k
- **R21 indirect_call_ratio** -- fraction of calls with a non-immediate target (IAT/vtable/dynamic)
- **R22 simd_dense** -- SSE/AVX instructions per 1k
- **R23 backedge_dense** -- resolved backward branches per 1k -- loop density

### balanced

| id | rule | kind | thr | fires on ransomware (test) | fires on goodware (test) | train P | train R | test P | test R | log-odds weight |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | aesni_any | presence | - | 0.0000 | 0.0157 | 0.8547 | 0.1256 | 0.0000 | 0.0000 | 2.0950 |
| R02 | pclmulqdq_any | presence | - | 0.0000 | 0.0157 | 0.0500 | 0.0013 | 0.0000 | 0.0000 | -2.3530 |
| R03 | sha_ni_any | presence | - | 0.0304 | 0.0157 | 0.7600 | 0.0477 | 0.8462 | 0.0304 | 1.3920 |
| R04 | crc32_any | presence | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.8700 |
| R05 | rdrand_any | presence | - | 0.0000 | 0.0157 | 0.9385 | 0.0766 | 0.0000 | 0.0000 | 2.9210 |
| R06 | crypto_const_any | presence | - | 0.3646 | 0.1811 | 0.8644 | 0.6407 | 0.8516 | 0.3646 | 3.0160 |
| R07 | loop_instr_any | presence | - | 0.4254 | 0.2677 | 0.6176 | 0.4849 | 0.8191 | 0.4254 | 1.0990 |
| R08 | cpuid_any | presence | - | 0.8039 | 0.5039 | 0.4586 | 0.4799 | 0.8197 | 0.8039 | 0.1200 |
| R09 | rdtsc_any | presence | - | 0.0276 | 0.0157 | 0.9344 | 0.2148 | 0.8333 | 0.0276 | 3.0790 |
| R10 | peb_teb_any | presence | - | 0.6657 | 0.1890 | 0.7206 | 0.4535 | 0.9094 | 0.6657 | 1.6280 |
| R11 | syscall_any | presence | - | 0.0193 | 0.0079 | 0.7759 | 0.0565 | 0.8750 | 0.0193 | 1.4900 |
| R12 | int_other_any | presence | - | 0.7127 | 0.6220 | 0.4189 | 0.5289 | 0.7656 | 0.7127 | -0.2190 |
| R13 | xor_dense | density | 8.8981 | 0.8895 | 0.9685 | 0.4479 | 0.9611 | 0.7236 | 0.8895 | 0.4270 |
| R14 | rotshift_dense | density | 6.4125 | 0.7541 | 0.4488 | 0.6166 | 0.8405 | 0.8273 | 0.7541 | 2.0010 |
| R15 | tight_crypto_loop | density | 0.5704 | 0.9033 | 0.8346 | 0.4764 | 0.9146 | 0.7552 | 0.9033 | 0.9900 |
| R16 | arx_window | density | 0.0883 | 0.9254 | 0.6457 | 0.5460 | 0.9246 | 0.8034 | 0.9254 | 2.0490 |
| R17 | rep_string_dense | density | 0.6500 | 0.7597 | 0.5512 | 0.5440 | 0.8003 | 0.7971 | 0.7597 | 1.2540 |
| R18 | scan_string_dense | density | 0.0250 | 0.5773 | 0.2520 | 0.6826 | 0.5754 | 0.8672 | 0.5773 | 1.6110 |
| R19 | int3_dense | density | 0.0000 | 1.0000 | 1.0000 | 0.4427 | 1.0000 | 0.7403 | 1.0000 | -0.2300 |
| R20 | call_dense | density | 19.0282 | 0.9337 | 0.9291 | 0.4508 | 0.9673 | 0.7412 | 0.9337 | 0.6920 |
| R21 | indirect_call_ratio | density | 0.0208 | 0.8398 | 1.0000 | 0.4472 | 0.9736 | 0.7053 | 0.8398 | 0.5160 |
| R22 | simd_dense | density | 0.0000 | 1.0000 | 1.0000 | 0.4427 | 1.0000 | 0.7403 | 1.0000 | -0.2300 |
| R23 | backedge_dense | density | 4.3454 | 0.9227 | 0.9134 | 0.4537 | 0.9736 | 0.7422 | 0.9227 | 0.9880 |


What each rule means:

- **R01 aesni_any** -- AES-NI round instruction (aesenc/aesdec/aeskeygenassist) present
- **R02 pclmulqdq_any** -- carry-less multiply present (AES-GCM / CRC / GF(2^n) math)
- **R03 sha_ni_any** -- SHA extension instruction (sha1rnds4/sha256rnds2/...) present
- **R04 crc32_any** -- SSE4.2 crc32 instruction present
- **R05 rdrand_any** -- hardware RNG (rdrand/rdseed) present -- key generation
- **R06 crypto_const_any** -- MD5/SHA-1/SHA-256 IV, SHA-1 K, CRC-32 polynomial, FNV or TEA delta appears as an immediate
- **R07 loop_instr_any** -- x86 `loop`/`loope`/`loopne` present (rare in modern compiler output)
- **R08 cpuid_any** -- cpuid present -- feature probe or VM/sandbox detection
- **R09 rdtsc_any** -- rdtsc/rdtscp present -- timing-based anti-analysis or seeding
- **R10 peb_teb_any** -- direct PEB/TEB access (fs:[0x30], fs:[0x18], gs:[0x60])
- **R11 syscall_any** -- syscall/sysenter in user code -- ntdll bypass
- **R12 int_other_any** -- software interrupt other than int3 (int 0x2d/0x2e anti-debug)
- **R13 xor_dense** -- xor instructions per 1k decoded instructions
- **R14 rotshift_dense** -- rol/ror/shl/shr/sar/shld/shrd per 1k -- bit-mixing chains
- **R15 tight_crypto_loop** -- backward branches spanning <=40 instructions over a logic/shift heavy body, per 1k -- the crypto inner-loop shape
- **R16 arx_window** -- 16-instruction windows that are both logic-heavy and arith- or rotate-heavy (ARX round shape), per 1k
- **R17 rep_string_dense** -- rep movs/stos per 1k -- bulk buffer copy/fill
- **R18 scan_string_dense** -- scas/cmps per 1k -- string scanning, path/extension enumeration
- **R19 int3_dense** -- int3 per 1k -- alignment padding density, a linker/packer artefact
- **R20 call_dense** -- call instructions per 1k
- **R21 indirect_call_ratio** -- fraction of calls with a non-immediate target (IAT/vtable/dynamic)
- **R22 simd_dense** -- SSE/AVX instructions per 1k
- **R23 backedge_dense** -- resolved backward branches per 1k -- loop density


## 4. Which rules fire on the hard negatives

The `balanced` dataset replaces the Mendeley goodware with Goodware_Balanced -- archivers, encryption utilities, backup and sync clients, secure-delete tools. That is the bucket these signatures are supposed to survive. The delta column is how much more often each rule fires on benign software when the benign software is ransomware-adjacent.

| id | rule | goodware fire rate, Mendeley test | goodware fire rate, Goodware_Balanced test | delta | example false positives |
|---|---|---|---|---|---|
| R22 | simd_dense | 0.3101 | 1.0000 | 0.6899 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R12 | int_other_any | 0.2713 | 0.6220 | 0.3507 | sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe, Jpg_transform.dll |
| R21 | indirect_call_ratio | 0.6899 | 1.0000 | 0.3101 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R08 | cpuid_any | 0.1938 | 0.5039 | 0.3101 | sdelete.exe, javaw.exe, eraser.exe, Jpg_transform.dll, mozavutil.dll |
| R10 | peb_teb_any | 0.0620 | 0.1890 | 0.1270 | sdelete.exe, subst.exe, PostgreSQL 17_17.11-3_Machine_X64_exe_en-US.exe, ConEmuHk64.dll, Microsoft .NET Windows Desktop Runtime 9.0_9.0.19_Machine_X64_burn_en-US.exe |
| R03 | sha_ni_any | 0.0000 | 0.0157 | 0.0157 | Steam.exe, SteamService.exe |
| R02 | pclmulqdq_any | 0.0000 | 0.0157 | 0.0157 | Steam.exe, SteamService.exe |
| R05 | rdrand_any | 0.0000 | 0.0157 | 0.0157 | Steam.exe, SteamService.exe |
| R01 | aesni_any | 0.0078 | 0.0157 | 0.0079 | Steam.exe, SteamService.exe |
| R09 | rdtsc_any | 0.0155 | 0.0157 | 0.0002 | Steam.exe, SteamService.exe |
| R04 | crc32_any | 0.0000 | 0.0000 | 0.0000 | - |
| R19 | int3_dense | 1.0000 | 1.0000 | 0.0000 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R13 | xor_dense | 0.9845 | 0.9685 | -0.0160 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R20 | call_dense | 1.0000 | 0.9291 | -0.0709 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R23 | backedge_dense | 0.9845 | 0.9134 | -0.0711 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R11 | syscall_any | 0.0930 | 0.0079 | -0.0851 | Metadata.dll |
| R15 | tight_crypto_loop | 1.0000 | 0.8346 | -0.1654 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, javaw.exe, eraser.exe |
| R07 | loop_instr_any | 0.4806 | 0.2677 | -0.2129 | Bitwarden-Installer-2026.8.0.exe, Steam.exe, Stub_Plugin.exe, wscript_2447bbda.exe, Video.dll |
| R14 | rotshift_dense | 0.7597 | 0.4488 | -0.3109 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, eraser.exe, Jpg_transform.dll |
| R16 | arx_window | 0.9767 | 0.6457 | -0.3310 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, eraser.exe, Jpg_transform.dll, mozavutil.dll |
| R17 | rep_string_dense | 1.0000 | 0.5512 | -0.4488 | Bitwarden-Installer-2026.8.0.exe, sdelete.exe, CM_FP_bin.ctest.exe, eraser.exe, mozavutil.dll |
| R06 | crypto_const_any | 0.8062 | 0.1811 | -0.6251 | Bitwarden-Installer-2026.8.0.exe, eraser.exe, mozavutil.dll, Steam.exe, RegionCapture.dll |
| R18 | scan_string_dense | 1.0000 | 0.2520 | -0.7480 | sdelete.exe, eraser.exe, Steam.exe, Stub_Plugin.exe, wscript_2447bbda.exe |


## 5. Calibration baseline: is macro-F1 0.96 real?

The `calibration_baseline` rows in section 1 are the highest scores anywhere in this project -- higher than the tokenization pipeline's best (0.926, `results/summary.md`), higher than graph2vec (0.895, `results/graph2vec/summary.md`) and far higher than CNN-ViT (0.628 seed-mean, `results/cnn_vit/summary.md`). A result that far outside the field is a claim about the corpus until it is audited, so this section audits it. Source: `rules_pipeline/baseline_audit.py` -> `results/rules/baseline_audit.json`.


### 5.1 Is the fit clean?

| dataset | vocab + IDF from train rows only | `transform(test)` left vocabulary unchanged | `transform(test)` left IDF unchanged | features | features test would have added | non-mnemonic tokens in vocabulary |
|---|---|---|---|---|---|---|
| mendeley | **PASS** | **PASS** | **PASS** | 74,738 | 25,956 | 0 |
| balanced | **PASS** | **PASS** | **PASS** | 86,677 | 25,639 | 0 |


Read the last two columns together. Fitting the vectoriser on train+test instead of train would have produced tens of thousands of extra n-gram columns, so the check is not vacuous -- there really is test-only vocabulary, and the production path really does exclude it. Every surviving feature is a 1-, 2- or 3-gram of mnemonics from the corpus vocabulary: no file length, no file size, no architecture flag, no family, no filename, no path. `X` handed to the classifier is the TF-IDF matrix and nothing is concatenated to it.


### 5.2 Are the rows disjoint, and are they the same rows?

| dataset | train (good/ransom) | test (good/ransom) | matches results/expC/sample_counts.json | sha256 shared train/test | duplicate sha256 anywhere | family overlap | group overlap | train / test families |
|---|---|---|---|---|---|---|---|---|
| mendeley | 2018 (1114/904) | 491 (129/362) | **PASS** | 0 | 0 | 0 | 0 | 24 / 14 |
| balanced | 2017 (1113/904) | 489 (127/362) | n/a | 0 | 0 | 0 | 0 | 24 / 14 |


The cohort rows are the same rows every other pipeline uses -- both harnesses call `cnn_vit_pipeline.cohort.load_split`, and the counts reconcile against the tokenization pipeline's own `results/expC/sample_counts.json` (1,114 goodware / 904 ransomware train, 129 / 362 test; this pipeline's `train` figure splits further into a fit and a val fold). No sha256 is on both sides, no ransomware family is on both sides, no group straddles the boundary.

The `balanced` row **is** the expB-style split, not a separate experiment: `cohort._balanced_frame` keeps the Mendeley ransomware split byte-for-byte and reads its goodware membership verbatim from the tokenization pipeline's own `results/expB/splits.csv` (snapshot at `results/cnn_vit/expB_splits_snapshot.csv`), mapped to sha256 through `LLM_Features_Balanced/opcode_manifest.csv` and intersected with `cohort_balanced`. So `mendeley` here corresponds to expA/expC and `balanced` to expB/expD; there is no third split to run.


Architecture composition, which matters for §5.6:

| dataset | split/class | x86 | x64 |
|---|---|---|---|
| mendeley | train/goodware | 632 | 482 |
| mendeley | train/ransomware | 862 | 42 |
| mendeley | test/goodware | 117 | 12 |
| mendeley | test/ransomware | 290 | 72 |
| balanced | train/goodware | 200 | 913 |
| balanced | train/ransomware | 862 | 42 |
| balanced | test/goodware | 46 | 81 |
| balanced | test/ransomware | 290 | 72 |


### 5.3 Exact-duplicate audit

Two content keys, independent of the cohort sha256: the sha256 of the `mn/<sha>.txt` file's raw bytes, and the sha256 of the capped token stream the vectoriser actually reads (first 30,000 mnemonics). A test row whose content hash appears in train is, for this model, a training row with a different name.

| dataset | content key | unique streams / files | goodware test leaked | ransomware test leaked | streams under both labels |
|---|---|---|---|---|---|
| mendeley | raw_mn_file_bytes | 1,655 / 2,509 | 62/129 (48.1%) | 0/362 (0.0%) | 2 |
| mendeley | capped_token_stream_30000 | 1,606 / 2,509 | 62/129 (48.1%) | 0/362 (0.0%) | 2 |
| balanced | raw_mn_file_bytes | 1,848 / 2,506 | 3/127 (2.4%) | 0/362 (0.0%) | 0 |
| balanced | capped_token_stream_30000 | 1,797 / 2,506 | 5/127 (3.9%) | 0/362 (0.0%) | 1 |


Largest duplicate groups in `mendeley` (by capped token stream):

| group size | in train | in test | labels | example filenames |
|---|---|---|---|---|
| 60 | 44 | 16 | 0 | GetSudokuPortable_1.0_Rev_2.paf.exe, PingusPortable_0.7.6.paf.exe, BigSolitairesPortable_1.4_Rev_3_English.paf.exe |
| 47 | 31 | 16 | 0 | IceBreakerPortable_2.2.1_English.paf.exe, SWI-PrologPortable_9.0.4_English.paf.exe, FrhedPortable_2017.11.paf.exe |
| 45 | 45 | 0 | 1 | 00ce72bb6fb1d2c1d32aa4c4a147e1b9b390cf9d3ae8b5c0cab2718118db4430, 038e577d25d5b9237fbbef6080f53f462b01e75f83449bf0020ef0b14f371ac6, 085105e613ad37808a8db9a3c2ba5561d5d38d5c5c43b469c93d15f0d64af0c1 |
| 39 | 39 | 0 | 1 | 03bb4d5c0179fdceacc5df7645e6e7fa93e931ac9beb1968c7bbe40ba3499194, 03d7a1c01a60f9718304c719a5734516e5e3f43ea0547f48e4462a063d986cfb, 0b3f4c353eef59e7c2cf0e211de11fe646d9fd20289f2bea7005b35aa6247a4a |
| 35 | 31 | 4 | 0 | PortableApps.comLauncher_2.2.3.paf.exe, ArmagetronAdvancedPortable_0.2.9.1.0.paf.exe, LBreakout2Portable_2.6.5_English.paf.exe |
| 34 | 0 | 34 | 1 | 002e70b8fde758f88adc506f3c71df1eb32ce1a2b5bec45f3e8e43dba923709f, 05f6252ae8441e6198a97e7bbae93c03fff0850e082406d6d784274d5ae07122, 0b1f19ba8740b10ed017671aab023228756a6864fb008bf23f3c606189bdcd98 |


Largest duplicate groups in `balanced` (by capped token stream):

| group size | in train | in test | labels | example filenames |
|---|---|---|---|---|
| 45 | 45 | 0 | 1 | 00ce72bb6fb1d2c1d32aa4c4a147e1b9b390cf9d3ae8b5c0cab2718118db4430, 038e577d25d5b9237fbbef6080f53f462b01e75f83449bf0020ef0b14f371ac6, 085105e613ad37808a8db9a3c2ba5561d5d38d5c5c43b469c93d15f0d64af0c1 |
| 39 | 39 | 0 | 1 | 03bb4d5c0179fdceacc5df7645e6e7fa93e931ac9beb1968c7bbe40ba3499194, 03d7a1c01a60f9718304c719a5734516e5e3f43ea0547f48e4462a063d986cfb, 0b3f4c353eef59e7c2cf0e211de11fe646d9fd20289f2bea7005b35aa6247a4a |
| 34 | 0 | 34 | 1 | 002e70b8fde758f88adc506f3c71df1eb32ce1a2b5bec45f3e8e43dba923709f, 05f6252ae8441e6198a97e7bbae93c03fff0850e082406d6d784274d5ae07122, 0b1f19ba8740b10ed017671aab023228756a6864fb008bf23f3c606189bdcd98 |
| 34 | 34 | 0 | 1 | 09f01f2256663969229c2c954f7b29751b41a0c2b36cdb9b67c9491b76c04898, 0ff4058f709d278ed662719b9627618c48e7a656c59f6bfecda9081c7cbd742b, 1228d0f04f0ba82569fc1c0609f9fd6c377a91b9ea44c1e7f9f84b2b90552da2 |
| 33 | 33 | 0 | 1 | 00ad914476509f84b40f2dbe804dc7c37a1a24ef3472674574d3367079bf0a2a, 04f65270c92dda82c759c1eee49cf8f4c98a2ed0071272e49132331fda482dba, 082f91d85c437f415cea44b36afb4198da07b78593c836a398cd96365166e7d8 |
| 25 | 0 | 25 | 1 | 05ae137c49d99b41296d91667f040082fc33d6f28acbccdc28cfbedf59f6f75b, 06173ef5e0646e104caad18a0f849975dfcddf6c292edfa4c2980b8947502ac8, 1670e8bb8065d23e1b93ed8173f079f338abef880047da21af95dd4db57e20bd |


### 5.4 What survives deduplication

| dataset | condition | test n | LogReg macro-F1 | LinearSVC macro-F1 | recall goodware | recall ransomware |
|---|---|---|---|---|---|---|
| mendeley | as reported (all test rows) | 491 | 0.9680 | 0.9603 | 0.9302 | 0.9917 |
| mendeley | same model, test rows not duplicated in train | 429 | 0.9547 | 0.9422 | 0.8955 | 0.9917 |
| mendeley | retrained on deduplicated train (1390 rows), clean test | 429 | 0.9262 | - | 0.8955 | 0.9724 |
| balanced | as reported (all test rows) | 489 | 0.7956 | 0.8023 | 0.8898 | 0.7983 |
| balanced | same model, test rows not duplicated in train | 484 | 0.8020 | 0.8089 | 0.9180 | 0.7983 |
| balanced | retrained on deduplicated train (1510 rows), clean test | 484 | 0.8089 | - | 0.9262 | 0.8039 |


### 5.5 Permutation and ablation

| dataset | condition | n features | macro-F1 | recall ransomware | FPR |
|---|---|---|---|---|---|
| mendeley | full model (LogReg) | 74738 | 0.9680 | 0.9917 | 0.0698 |
| mendeley | (a) train labels shuffled, 3 runs (mean) | 74738 | 0.3864 | - | - |
| mendeley | (b) drop top-50 n-grams | 74688 | 0.9734 | 0.9917 | 0.0543 |
| mendeley | (b') drop top-500 | 74238 | 0.9433 | 0.9586 | 0.0543 |
| mendeley | (c) unigram TF-IDF only | 503 | 0.9378 | 0.9834 | 0.1318 |
| mendeley | (d) length-only control (2 scalars, RF) | 2 | 0.5092 | 0.3895 | 0.1473 |
| balanced | full model (LogReg) | 86677 | 0.7956 | 0.7983 | 0.1102 |
| balanced | (a) train labels shuffled, 3 runs (mean) | 86677 | 0.3653 | - | - |
| balanced | (b) drop top-50 n-grams | 86627 | 0.7583 | 0.7459 | 0.1102 |
| balanced | (b') drop top-500 | 86177 | 0.7969 | 0.8039 | 0.1181 |
| balanced | (c) unigram TF-IDF only | 651 | 0.6809 | 0.6547 | 0.1575 |
| balanced | (d) length-only control (2 scalars, RF) | 2 | 0.4269 | 0.2680 | 0.1181 |


(a) is the test that matters. With the training labels shuffled the same pipeline collapses to roughly the always-predict-ransomware floor (macro-F1 0.424), so the 0.96 is not an artefact of the vectoriser, the grid search, or the metric -- it is coming from the label-feature association. (b) says the score is not carried by a handful of magic n-grams: the signal is spread over the whole vocabulary. (d) says it is not file length in disguise.


### 5.6 Why it works


**mendeley -- top n-grams by logistic-regression coefficient.**

| toward ransomware | coef + | toward goodware | coef - |
|---|---|---|---|
| `int3 push mov` | 1.2940 | `mov test js` | -1.4140 |
| `ret sub call` | 1.2330 | `imul cmp jae` | -1.3460 |
| `jmp jmp jmp` | 1.2320 | `pop call push` | -1.3270 |
| `mov jae sub` | 1.2240 | `int3 int3 jmp` | -1.3080 |
| `jae sub dec` | 1.1810 | `popal je and` | -1.2610 |
| `adc mov` | 1.1520 | `jmp sub sub` | -1.2260 |
| `shr xor mov` | 1.0830 | `je sub je` | -1.2240 |
| `movsx shl or` | 1.0730 | `dec shl or` | -1.2210 |
| `shl or movsx` | 1.0610 | `sub je sub` | -1.2030 |
| `call lock` | 1.0550 | `add add inc` | -1.1740 |
| `or movsx` | 1.0270 | `push push jmp` | -1.1710 |
| `movsx shl` | 1.0260 | `imul cmp` | -1.1660 |
| `pop call leave` | 1.0240 | `push pop call` | -1.1630 |
| `mov jmp jmp` | 1.0220 | `popal je` | -1.1550 |
| `xor mov shr` | 1.0140 | `pop jne push` | -1.1550 |


**mendeley -- per-architecture.** The whole-corpus model, scored inside each architecture:

| arch | n | support_goodware | support_ransomware | recall_goodware | recall_ransomware | accuracy | macro_f1 |
|---|---|---|---|---|---|---|---|
| x64 | 84 | 12 | 72 | 1.0000 | 0.9583 | 0.9643 | 0.9338 |
| x86 | 407 | 117 | 290 | 0.9231 | 1.0000 | 0.9779 | 0.9724 |


And a model **trained and tested inside one architecture only** -- no cross-architecture shortcut available:

| arch | train n (good/ransom) | test n | macro-F1 | recall goodware | recall ransomware | note |
|---|---|---|---|---|---|---|
| x64 | 524 (482/42) | 84 | 0.7619 | 1.0000 | 0.8056 |  |
| x86 | 1494 (632/862) | 407 | 0.9724 | 0.9231 | 1.0000 |  |


**mendeley -- per-family test recall** (all 14 families are unseen in training):

| family | n | recall | detected |
|---|---|---|---|
| blackbasta | 30 | 0.9000 | 27 |
| avoslocker | 50 | 1.0000 | 50 |
| bianlian | 11 | 1.0000 | 11 |
| blackbyte | 7 | 1.0000 | 7 |
| blackcat | 50 | 1.0000 | 50 |
| bluesky | 34 | 1.0000 | 34 |
| clop | 45 | 1.0000 | 45 |
| hive | 50 | 1.0000 | 50 |
| holyghost | 4 | 1.0000 | 4 |
| karma | 13 | 1.0000 | 13 |
| lorenz | 16 | 1.0000 | 16 |
| maui | 3 | 1.0000 | 3 |
| playcrypt | 43 | 1.0000 | 43 |
| quantum | 6 | 1.0000 | 6 |


**balanced -- top n-grams by logistic-regression coefficient.**

| toward ransomware | coef + | toward goodware | coef - |
|---|---|---|---|
| `adc mov` | 1.4430 | `je sub je` | -1.5430 |
| `add push lea` | 1.3980 | `int3 int3 jmp` | -1.5080 |
| `sbb mov mov` | 1.3960 | `jmp` | -1.4620 |
| `add adc mov` | 1.3170 | `sub je sub` | -1.4080 |
| `jb mov push` | 1.3140 | `lea call nop` | -1.3940 |
| `add push push` | 1.2950 | `int3 int3 int3` | -1.3850 |
| `rol` | 1.2880 | `call nop` | -1.3600 |
| `pop mov pop` | 1.2630 | `leave ret mov` | -1.2960 |
| `pop mov` | 1.2620 | `leave ret` | -1.2580 |
| `call push lea` | 1.2220 | `sub je` | -1.1840 |
| `pop pop mov` | 1.2220 | `pop call leave` | -1.1810 |
| `int3 push mov` | 1.1630 | `int3 int3` | -1.1360 |
| `xor call call` | 1.1570 | `call leave ret` | -1.1340 |
| `movabs nop movsxd` | 1.1470 | `jb and xor` | -1.1200 |
| `idiv` | 1.1300 | `pop leave ret` | -1.1080 |


**balanced -- per-architecture.** The whole-corpus model, scored inside each architecture:

| arch | n | support_goodware | support_ransomware | recall_goodware | recall_ransomware | accuracy | macro_f1 |
|---|---|---|---|---|---|---|---|
| x64 | 153 | 81 | 72 | 1.0000 | 0.7778 | 0.8954 | 0.8926 |
| x86 | 336 | 46 | 290 | 0.6957 | 0.8034 | 0.7887 | 0.6709 |


And a model **trained and tested inside one architecture only** -- no cross-architecture shortcut available:

| arch | train n (good/ransom) | test n | macro-F1 | recall goodware | recall ransomware | note |
|---|---|---|---|---|---|---|
| x64 | 955 (913/42) | 153 | 0.7506 | 1.0000 | 0.5139 |  |
| x86 | 1062 (200/862) | 336 | 0.8695 | 0.6522 | 0.9931 |  |


**balanced -- per-family test recall** (all 14 families are unseen in training):

| family | n | recall | detected |
|---|---|---|---|
| blackcat | 50 | 0.0000 | 0 |
| hive | 50 | 0.6400 | 32 |
| blackbyte | 7 | 0.7143 | 5 |
| blackbasta | 30 | 0.9000 | 27 |
| avoslocker | 50 | 1.0000 | 50 |
| bianlian | 11 | 1.0000 | 11 |
| bluesky | 34 | 1.0000 | 34 |
| clop | 45 | 1.0000 | 45 |
| holyghost | 4 | 1.0000 | 4 |
| karma | 13 | 1.0000 | 13 |
| lorenz | 16 | 1.0000 | 16 |
| maui | 3 | 1.0000 | 3 |
| playcrypt | 43 | 1.0000 | 43 |
| quantum | 6 | 1.0000 | 6 |


**Reading §5.6.** Three things, in order of how much they matter.

1. **The ransomware side generalises, and duplication cannot explain it.** On `mendeley`, 13 of the 14 held-out families are detected at recall 1.00 and the worst is `blackbasta` at 0.90. These are families with **no representative in training at all**, and the ransomware test-leak rate in §5.3 is 0.0% -- not one ransomware test file is a verbatim copy of a training file. Whatever the goodware half of this result is worth, the ransomware half is genuine cross-family generalisation.

2. **It is not the architecture shortcut.** Mendeley training has only 42 x64 ransomware against 482 x64 goodware, so 'x64 implies goodware' is available -- and would be punished by a test set that is 72 of 84 x64 rows ransomware. The model scores macro-F1 0.934 inside x64 and 0.972 inside x86, and a model trained **only** on x64 rows still reaches 0.762 from 42 positive training examples. The shortcut is available and is not being taken.

3. **The n-grams are compiler idiom, not semantics.** The positive weights are dominated by wide-integer arithmetic (`adc mov`, `sbb mov mov`, `add adc mov`, `movsx shl or`, `rol`, `idiv`) -- the shape of bignum and block-cipher inner loops compiled without SIMD -- and the negative weights by MSVC CRT padding and epilogue idiom (`int3 int3 int3`, `call leave ret`, `pop leave ret`). That is a real and explainable difference, but it is a difference between *toolchains and code styles*, not between 'encrypts your files' and 'does not'. It is exactly the kind of feature a different goodware corpus can erase, and that is what `balanced` does: macro-F1 falls to 0.796, x86 goodware recall falls to 0.70, and **blackcat** is missed entirely (50 files, recall 0.00).


## 6. Verdict on the calibration baseline

| dataset | as reported | goodware test leak rate | ransomware test leak rate | on non-duplicated test rows | retrained dedup train, clean test | shuffled train labels |
|---|---|---|---|---|---|---|
| mendeley | 0.9680 | 48.1% | 0.0% | 0.9547 | 0.9262 | 0.3864 |
| balanced | 0.7956 | 3.9% | 0.0% | 0.8020 | 0.8089 | 0.3653 |


The mechanical checks all pass: vocabulary and IDF are fitted on train rows only, the test rows are disjoint from train by cohort sha256, the 14 test families are disjoint from the 24 training families, no group straddles the split, the rows reconcile against `results/expC/sample_counts.json`, the feature matrix is mnemonic n-grams and nothing else, and the score collapses to the always-predict-ransomware floor when the training labels are shuffled. **The number is not produced by a coding error in the harness.**

What it *is* partly produced by is the corpus. The Mendeley goodware half of the test set contains verbatim duplicates of training files (the leak-rate column above; cause in `docs/tokenization_audit.md` §2.1 -- `extract.py` disassembles installer stubs, so many PortableApps launchers collapse onto a handful of identical opcode streams that land on both sides of the split). The ransomware half does not have this problem, because it is family-disjoint by construction. So the goodware column of the Mendeley result is inflated and the ransomware column is not, and the honest headline is the deduplicated row, not the as-reported one.

The `balanced` dataset is the control for exactly this. Its goodware test-leak rate is a twelfth of Mendeley's, and there the deduplicated numbers go *up*, not down -- the duplication is not doing any work. The same model scores materially lower on that dataset for a different reason (§5.6 point 3).

**Bottom line.** The mnemonic 1-3-gram TF-IDF baseline is clean enough to quote and is the strongest result in this project. Quote it as **0.95 (Mendeley, duplicate test rows removed) / 0.80 (Goodware_Balanced)**, not as 0.968, and always with the pair -- the gap between the two datasets is the most informative thing about it. Two caveats travel with the number: it is one fixed family split, so the per-family table is a sample of size 14, not a distribution; and what it has learned is compiler and code-style idiom, which is cheap to evade deliberately even though it transfers across 14 unseen families that were not trying to evade it.
