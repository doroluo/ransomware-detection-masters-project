"""Constants shared by more than one gate stage."""

PATHS_X86 = dict(protocol='kfold', arch='x86', mask_blocks=0.15)
REF_VEC = 'ours_cap0_ref'
PATHS_VEC = dict(**PATHS_X86, block_input='vec', block_vec=REF_VEC, val_seed=0)

CB_VEC = 'ours_cap0_codebert'
CB_VEC_ARMS = dict(**PATHS_X86, block_input='vec', block_vec=CB_VEC)
