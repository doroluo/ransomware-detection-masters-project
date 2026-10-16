"""Step 11, final classifier: `head.make_head` is a 2-layer MLP over [binary vector ; bin_feat], bin_feat being the log1p
counts of the full binary (`head.bin_feat`). With `stack_tfidf` the standardised out-of-fold TF-IDF logit of the binary
(`baselines.py --stack-logits`, batch key `stack_logit`) is one more input. Out: one logit.
RunConfig: dropout, stack_tfidf."""
