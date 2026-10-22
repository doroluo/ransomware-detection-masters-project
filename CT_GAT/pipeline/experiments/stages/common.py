"""Constants shared by more than one gate stage."""
from common.config import RunConfig
from s04_evaluation.stage import beats, variant_of

PATHS_X86 = dict(protocol='kfold', arch='x86', mask_blocks=0.15)

CB_VEC = 'ours_cap0_codebert'
CB_VEC_ARMS = dict(**PATHS_X86, block_input='vec', block_vec=CB_VEC)

# ---- the team arms on a dataset other than kfold x86 (stages team_mendeley_all, team_v3_all, team_v3_matched) --------------
# Groups come from RunConfig, so they cannot drift from the run names `experiments/team_pipelines/common.team_config` and
# `experiments.train` write. The vocabulary clash: `sig500v3` is the s02-fix *build* version, `kfold_v3` the fold *protocol*.

SEEDS = (0, 1, 2)
BUILDS = ('sig500v3', 'sig250', 'callwalk500', 'callwalk250')      # signal 500 (the fixed build), signal 250, callwalk 500 and 250
INPUTS = (('feat', 'feat (Model 2)'), ('emb', 'emb (Model 3)'), ('e2e', 'e2e (Model 3, trained encoder)'))
WHOLE_INPUTS = (('sigwhole', 'sigwhole'), ('sigwhole-feat', 'sigwhole feat (Model 2)'), ('sigwhole-emb', 'sigwhole emb (Model 3)'))
MODEL1_LENGTHS = (('L128', None, ''), ('L512', 512, ''), ('L512 whole', 512, 'whole'))      # label, --max-instructions, --tag
AB = ' ab'      # the label suffix of an arm trained with --arch-balance


def team_group(arm, protocol, arch, arch_balance=False):
    """RunConfig.group of a team arm: `kfold_all_gnn_team-<arm>`, `kfold_v3_matched_gnn-ab_team-<arm>`."""
    return RunConfig(protocol=protocol, fold=0, arch=arch, model='gnn', tag=f'team-{arm}', arch_balance=arch_balance).group


def model1_group(protocol, arch, instructions=None, tag='', arch_balance=False):
    """RunConfig.group of Model 1 (`--model fivechannel`): `kfold_all_fivechannel-L512-ab`, `kfold_all_fivechannel-L512_whole`."""
    kw = dict(max_instructions=instructions) if instructions else {}
    return RunConfig(protocol=protocol, fold=0, arch=arch, model='fivechannel', tag=tag, arch_balance=arch_balance, **kw).group


def team_label(build, name, arch_balance=False):
    return f'team {build} {name}' + (AB if arch_balance else '')


def team_rows(protocol, arch, builds=BUILDS, arch_balance=False):
    """(label, group) of the three inputs of each build: `team sig500v3 feat (Model 2)` -> `kfold_all_gnn_team-sig500v3-feat`."""
    return tuple((team_label(build, name, arch_balance), team_group(f'{build}-{key}', protocol, arch, arch_balance))
                 for build in builds for key, name in INPUTS)


def whole_rows(protocol, arch):
    """The three arms on the uncapped build (kfold only: the combined dataset has no uncapped build)."""
    return tuple((f'team {name}', team_group(arm, protocol, arch)) for arm, name in WHOLE_INPUTS)


def model1_rows(protocol, arch, lengths=MODEL1_LENGTHS, arch_balance=False):
    """Model 1 at the given lengths: (`Model 1 five-channel L512`, group)."""
    return tuple((f'Model 1 five-channel {label}' + (AB if arch_balance else ''),
                  model1_group(protocol, arch, instructions, tag, arch_balance)) for label, instructions, tag in lengths)


def input_pairs(builds=BUILDS, arch_balance=False):
    """feat -> emb (what the stored block embedding adds to the block features) and emb -> e2e (what training the encoder adds)."""
    pairs = []
    for build in builds:
        feat, emb, e2e = (team_label(build, name, arch_balance) for _, name in INPUTS)
        pairs += [(feat, emb), (emb, e2e)]
    return tuple(pairs)


def sampler_pairs(arch_balance=False):
    """signal -> callwalk at the same budget and input: the sampler."""
    return tuple((team_label(signal, name, arch_balance), team_label(walk, name, arch_balance))
                 for signal, walk in (('sig500v3', 'callwalk500'), ('sig250', 'callwalk250')) for _, name in INPUTS)


def budget_pairs(arch_balance=False):
    """500 -> 250 blocks at the same sampler and input: the budget."""
    return tuple((team_label(big, name, arch_balance), team_label(small, name, arch_balance))
                 for big, small in (('sig500v3', 'sig250'), ('callwalk500', 'callwalk250')) for _, name in INPUTS)


def published_v3_rows(arch):
    """The published kfold_v3 rows of docs/history/gates_v3.md: TF-IDF, and the GNN arms R2 / R4 / R7 (block masking 15%,
    cell-balanced checkpoint selection), plain and arch-balanced (`ab`). The plain GNN arms are not on disk."""
    gnn = (('R2 no-edges', dict(control='no_edges')), ('R4 shuffle', dict(control='shuffle')), ('R7 typed', {}))
    group = lambda **kw: f"kfold_v3_{arch}_{variant_of(mask_blocks=0.15, select_on='cell', **kw)}"      # noqa: E731
    return ((('R0 TF-IDF', 'tfidf'),) + tuple((name, group(**kw)) for name, kw in gnn)
            + (('R0 TF-IDF' + AB, 'tfidf-ab'),) + tuple((name + AB, group(arch_balance=True, **kw)) for name, kw in gnn))


def seed_rule(first, second, result, info):
    """The rule column: the pre-registered minimum effect (`beats`, 3 seeds); with fewer seeds in both arms (TF-IDF and the
    published GNN arms have seed 0 only) the word is a direction, not the test."""
    word = 'beats' if beats(result) else 'does not beat'
    return word if len(info['seeds']) == len(SEEDS) else f"{word} (seeds {info['seeds']} only: a direction, not the test)"


def team_v3_parts(arch):
    """(arms, pairs, match-rate builds) of the stages team_v3_all and team_v3_matched on `arch`: the published rows, the team arms
    of the 500- and 250-block builds of the signal and callwalk samplers (the 250s when they are built), Model 1 at 128 and 512
    instructions, and the `-ab` twins of the signal-500 groups. Each team arm is paired against R7 ab, R2 ab and TF-IDF ab, each
    twin against its plain arm, with the pairs of inputs, sampler and budget."""
    protocol = 'kfold_v3'
    published = published_v3_rows(arch)
    team = team_rows(protocol, arch)
    model1 = model1_rows(protocol, arch, MODEL1_LENGTHS[:2])
    twins = team_rows(protocol, arch, ('sig500v3',), arch_balance=True) + model1_rows(protocol, arch, MODEL1_LENGTHS[:2], arch_balance=True)
    references = [label for label, _ in published if label in ('R7 typed' + AB, 'R2 no-edges' + AB, 'R0 TF-IDF' + AB)]
    pairs = (tuple((ref, label) for label, _ in team + model1 + twins for ref in references)
             + tuple((label.removesuffix(AB), label) for label, _ in twins)
             + input_pairs() + input_pairs(('sig500v3',), arch_balance=True) + sampler_pairs() + budget_pairs()
             + ((model1[0][0], model1[1][0]),))
    builds = tuple((f'b_{family}_combined_v3', f'team/b_{family}_combined_v3') for family in ('sig500', 'callwalk500'))
    return published + team + model1 + twins, pairs, builds
