"""Step 11, final classifier: `head.make_head` is a 2-layer MLP over [binary vector ; bin_feat], bin_feat being the log1p
counts of the full binary (`head.bin_feat`). With `stack_tfidf` the standardised out-of-fold TF-IDF logit of the binary
(`baselines.py --stack-logits`, batch key `stack_logit`) is one more input, and with `imports` the unreferenced-import bag
(32 dimensions) and its two log counts (step 10; batch keys `imp_idx`, `imp_w`, `imp_offsets`, `imp_feat`). Out: one logit.
RunConfig: dropout, stack_tfidf, imports."""
