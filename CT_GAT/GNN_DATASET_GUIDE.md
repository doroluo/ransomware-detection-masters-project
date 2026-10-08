# Mendeley CFG dataset: start here

This package contains static IDA CFG exports for **2,670 unique Mendeley binaries**
(2,675 original file occurrences). Labels: **0 = goodware, 1 = ransomware**.
It includes a Python reader for GNN and graph-transformer work. No sample
executables or IDA databases are included. Assembly listings are a separate export;
you do not need them to read the CFGs or instruction tokens in this package.

## First five minutes

1. Extract `Mendeley_GNN_v1.zip` once with Windows Extract All, 7-Zip, or your usual ZIP tool.
2. Open `samples.csv` to browse labels, architecture, family, graph size, and availability.
3. Open either `examples/<SHA256>.json` in a text editor to see an ordinary, formatted
   JSON example containing one real function graph and its parent sample metadata.
4. Open a terminal **inside the extracted `Mendeley_GNN_v1` directory**:

```console
python dataset.py verify
python dataset.py inspect
python dataset.py inspect --split test --architecture x64 --class-name ransomware
```

Python 3.8+ is sufficient for the reader. Inspecting and loading graphs uses only
the standard library. `verify` checks all packaged files against SHA256SUMS.txt;
checksums detect corruption, not authenticity of an untrusted download.

The `.jsonl.gz` extension means **gzip-compressed JSON Lines**, not RAR or an
executable format. Windows may show a WinRAR icon because of its file association.
The reader handles decompression automatically. Each JSON line is one record, so
large samples can be streamed instead of loading the entire dataset into memory.

## Folder map

```text
Mendeley_GNN_v1/
  START_HERE.md                  this guide
  dataset.py                     portable loader and inspection commands
  samples.csv                    one row per SHA-256, including excluded samples
  summary.json                   exact counts and feature names
  SHA256SUMS.txt                 integrity checks for all other files
  splits/
    train.csv                    recommended training binaries
    test.csv                     recommended test binaries
    excluded.csv                 other binaries and explicit exclusion reasons
  cfg/
    goodware/train/<SHA256>.jsonl.gz
    goodware/test/<SHA256>.jsonl.gz
    ransomware/train/<family>/<SHA256>.jsonl.gz
    ransomware/test/<family>/<SHA256>.jsonl.gz
  examples/<SHA256>.json         one small training function per class
  provenance/
    original_occurrences.csv     all original 2,675 paths and cohort decisions
    identity_aliases.csv         identity mapping from reconciliation
    extraction_policy.json      extraction configuration and source hashes
```

The directories preserve original collection groupings. **Use `splits/*.csv` for
model membership**, not folder enumeration. Reconciliation can exclude a sample
or change its effective split. Duplicate occurrences share a canonical file;
`original_occurrences.csv` retains every original grouping and path. Its source
and `feature_file` columns refer to the original extraction environment, while
`samples.csv.cfg_path` is the portable path to use in this package.

## Which samples should I train on?

| Default split | Goodware | Ransomware | Total |
|---|---:|---:|---:|
| Train | 1,054 | 896 | 1,950 |
| Test | 127 | 357 | 484 |
| Total | 1,181 | 1,253 | 2,434 |

The default lists require a recovered native x86/x64 CFG, a ready export, and
accepted, selected cohort identity. They contain no repeated identity groups
within or across splits. **236 unique binaries remain in the inventory but are
outside these defaults.** Some have recovered CFGs but fail cohort/identity
eligibility; others lack a usable graph. Across the full inventory, 2,546 have
recovered CFGs. An extraction status of `ok` alone does not mean usable CFG data.

These defaults preserve the reconciled experiment split, not a newly randomized
split. They are a starting policy, not proof of complete disassembly or of an
unbiased experiment. Read `exclusion_reason` before changing membership.

Important `samples.csv` fields:

| Field | Meaning |
|---|---|
| `sample_id` | SHA-256 of the input representation, also the data filename |
| `label`, `class_name` | Binary classification target; never node features |
| `original_split` | Original folder split for the canonical occurrence |
| `recommended_split` | Reconciled train/test membership; blank means excluded |
| `architecture` | `x86`, `x64`, or another recorded input architecture |
| `family` | Source family/group metadata; do not feed this into the classifier |
| `identity_group` | Keep all equivalent/packed/unpacked representations together |
| `representation`, `cohort_tag` | Representation and original cohort mapping details |
| `input_kind`, `cfg_status` | Input type and graph recovery outcome |
| `eligible_for_gnn`, `exclusion_reason` | Default policy decision and rationale |
| `cfg_path` | Relative path from the extracted package root |
| `num_functions`, `num_blocks`, `num_edges`, `num_instructions` | Export graph sizes, useful for memory planning |
| `original_file_occurrences` | Number of source occurrences represented by this SHA |

For example, filter both recommended splits to x64 before an x64-only experiment.
To create a different split, use the full inventory plus original occurrence
metadata, keep `identity_group` disjoint, save the chosen manifests, and record
your architecture, family, representation, and eligibility rules. Derive any
validation split from training identities; do not tune on the test set. Identity
reconciliation removes known exact/representation duplicates, not every possible
near-duplicate. Family or temporal holdouts require their own documented protocol.

## Load a graph without parsing the export format

Run this from the extracted package directory:

```python
from dataset import MendeleyDataset

train = MendeleyDataset('.', split='train', architecture='x64')
sample = train[0]                       # one binary and its metadata
for graph in train.functions(sample):  # stream its function CFGs
    print(graph['num_nodes'], graph['edge_index'])
    print(graph['node_tokens'][0] if graph['num_nodes'] else [])
```

A **node is a basic block** containing an ordered sequence of instructions.
An **edge is a directed control-flow connection between blocks in one function**.
These exports contain both blocks and edges: they are CFGs.

```text
edge_index = [[0, 0, 1],     source nodes
              [1, 2, 2]]    destination nodes

Meaning: 0 -> 1, 0 -> 2, 1 -> 2
```

The reader remaps IDA's function-local block IDs to consecutive node indices.
It retains isolated blocks and external placeholders. It does not silently drop
nodes, impose block/instruction caps, reverse edges, or add self-loops.

| Returned field | Shape / purpose |
|---|---|
| `edge_index` | `[2, E]`, directed integer node indices |
| `x` | `[N, 6]`, numeric structural baseline features |
| `node_tokens` | `N` lists of ordered instruction mnemonics, variable length |
| `node_operand_categories` | Operand categories aligned with each instruction |
| `node_api_references` | Resolved DLL/symbol references aligned with instructions |
| `node_is_external` | Flags for placeholder blocks that may have no instructions |
| `node_start_rva`, `function_name`, `function_rva` | Inspection metadata, not default model inputs |

The six numeric features are `log1p(instruction count)`, `log1p(call count)`,
`log1p(import-reference count)`, external-placeholder flag, `log1p(in-degree)`,
and `log1p(out-degree)`, in that order. They are a usable structural baseline,
**not a trained opcode embedding or a complete ransomware detector**.

For a graph transformer, encode each block's `node_tokens` (optionally operands
and API references), then combine these block embeddings with CFG connectivity.
Fit vocabularies, normalization, embeddings, and feature selection only on the
training partition. Keep tokens for unknown instructions/APIs. The loader does
not pad, truncate, choose an attention mask, or train that encoder for you.

## PyTorch Geometric adapter

Install PyTorch and PyTorch Geometric for your own CPU/CUDA environment first.
The optional adapter follows the official
[PyG Data convention](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html):
floating point `x[N,F]`, integer `edge_index[2,E]`, and a binary-level target.

```python
from dataset import MendeleyDataset

train = MendeleyDataset('.', split='train')
sample = train[0]
graph = train.sample_graph(sample)  # union of all function CFGs in this binary
data = train.to_pyg(graph, sample)
print(data.x.shape, data.edge_index.shape, data.y)
```

`sample_graph` offsets node indices and combines functions into a **disjoint
union**. It adds no call edges between functions and is not an interprocedural
CFG. A graph-level classifier can pool over the binary. It materializes one
whole binary, which can be large; use `functions()` for bounded per-function
processing and a binary-level aggregation design when needed. Functions also
vary in size, so streaming alone is not a strict memory bound.

`to_pyg` uses the six structural features. It does not automatically encode or
attach the token lists; use the returned Python graph to build custom node
embeddings and replace `data.x`. If training per function, the inherited label
is the **parent binary's weak label**, not a claim that every function is malicious.
Keep functions from one identity in one split and evaluate predictions per binary.

## Inspect full details or open plain JSON

The two example JSON files are deliberately small, one-function previews, not
the full dataset. To export all function graphs of a chosen binary to readable
JSON, replace `SHA256` below with its sample ID:

```console
python dataset.py export-json --split all --sample-id SHA256 --output SHA256.json
```

Open the result in Notepad, VS Code, or another text editor. Large binaries may
produce very large JSON files; there is no silent size cap. The command refuses
to overwrite an existing output.

For research beyond the lightweight view, the original compressed export is
preserved byte-for-byte. Read it with `gzip.open(path, 'rt', encoding='utf-8')`
and apply `json.loads` to each line. Schema `ida-cfg-research/2.1` has a header,
function records, optional orphan/string records, and a completion footer.
The original records retain the extractor's operands, addresses, symbols,
references and other metadata. The convenience loader exposes a subset;
it does not remove the original data. Orphan instructions are not fabricated
into CFG nodes. Import references do not by themselves prove runtime behavior.

The function iterator checks identity/schema immediately and checks the completion
footer when exhausted. Stopping early does not validate the remaining stream.
Use `verify` after transfer to validate the entire packaged snapshot.

## Modelling limitations and reproducibility

- Recovered static CFGs may be incomplete for packed/obfuscated inputs and indirect
  control flow. Keep representation and recovery metadata with results.
- Managed .NET, unsupported architectures, and samples with no recovered function
  instructions need separate analysis; do not turn them into empty training graphs.
- Addresses, symbols, family names, paths, and split metadata can create shortcuts.
  Preserve them for inspection and controlled ablations, not as implicit features.
- Goodware and ransomware architecture proportions differ. Report architecture and
  family results as well as overall metrics; compare models on identical manifests.
- `summary.json`, original provenance, and checksums identify this snapshot. Save
  any custom split, tokenization policy, graph sampling, caps, and model settings
  with the experiment. Report exclusions and binary-level sample counts.

## Rebuild from the desktop raw export

From the repository root (choose a new destination for each release):

```console
python CT_GAT/preprocessing/build_gnn_bundle.py --source reports/ida_cfg_quality_20261006/full --destination reports/google_drive_upload/2026-10-07/Mendeley_GNN_v1 --guide CT_GAT/GNN_DATASET_GUIDE.md --zip
```

The builder checks each original artifact's recorded SHA-256, copies derived
exports into the grouped layout, checks copied hashes, verifies identity split
separation, writes manifests and examples, and checks ZIP CRCs. It never changes
the raw exports or copies sample executables. Existing destinations are refused.
