"""Stage day1b.

Day 1b reads the extra arms of step 5b (direction, augmentation, masking, depth, train fraction): the pooled table of
every kfold x86 GNN group on disk, the AUC-selection paired differences of each new arm against R7 and R2 tfidf, and
the learning curve (train fraction 0.25 / 0.5 / 1.0 for R7 and TF-IDF).
"""
from s04_evaluation.stage import Stage

NEW_ARMS = tuple(f'kfold_x86_gnn-{v}' for v in ('mtok15', 'dfn5-dlib40', 'undir', 'fwd', 'dfn15', 'tf50', 'tf25',
                                                'mblk15', 'dedge20', 'flip20', 'mtok50', 'bwd', 'L8'))
REFERENCES = (('R7 typed', 'kfold_x86_gnn'), ('R2 tfidf', 'kfold_x86_gnn-noedges'))
CURVE = ((0.25, 'tf25'), (0.5, 'tf50'), (1.0, ''))

STAGE = Stage('day1b', layout='day1b', arms=REFERENCES,
              options={'new_arms': NEW_ARMS, 'curve': CURVE,
                       'curve_models': (('R7 typed', 'kfold_x86_gnn'), ('TF-IDF', 'tfidf'))})
