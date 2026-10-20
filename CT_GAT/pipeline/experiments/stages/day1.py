"""Stage day1.

Day 1 reads seed 0 of the kfold x86 arms (R0 TF-IDF baseline, R2 tfidf / mean, R4 shuffle, R5 untyped, R7 typed GNN):
a pooled table, the preliminary gate-A verdict of PLAN_GNN.md section 5, and the stats.py paired differences. An arm
with missing folds is reported as missing with the folds that are present; nothing is dropped silently.
"""
from s04_evaluation.stage import Stage

ARMS = (('R0 TF-IDF', 'tfidf'), ('R2 tfidf', 'kfold_x86_gnn-noedges'), ('R2 mean', 'kfold_x86_gnn-noedges-enc-mean'),
        ('R4 shuffle', 'kfold_x86_gnn-shuffle'), ('R5 untyped', 'kfold_x86_gnn-untyped'), ('R7 typed', 'kfold_x86_gnn'))
PAIRS = (('R2 tfidf', 'R4 shuffle'), ('R2 tfidf', 'R7 typed'), ('R4 shuffle', 'R7 typed'), ('R5 untyped', 'R7 typed'),
         ('R0 TF-IDF', 'R2 tfidf'))

GATE_A = ('R2 `--encoder tfidf` against R0 TF-IDF (PLAN_GNN.md section 5): pooled AUC >= R0 - 0.01 and macro-F1 at '
          'the validation threshold >= R0 - 0.02. The AUC-checkpoint line is the one that applies the validation '
          'threshold; the F1-checkpoint line scores at 0.5. This is one seed and cannot pass the pre-registered '
          'rule.')

STAGE = Stage('day1', layout='day1', intro=GATE_A, arms=ARMS, pairs=PAIRS,
              options={'gate_a': ('R0 TF-IDF', 'R2 tfidf'), 'margins': (0.01, 0.02)})
