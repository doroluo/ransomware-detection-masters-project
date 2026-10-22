# IDA operand-role and header-import readers

`preprocessing/ida_feature_readers.py` provides both readers without modifying
Yanping's vocabulary, training code, IDA exports, or main graph caches. It uses
Python's standard library and the existing `token_mapping.py` dictionaries.

```powershell
python CT_GAT/preprocessing/ida_feature_readers.py C:/path/sample.jsonl.gz --out-dir reports/ida_features
```

Supply several inputs positionally for a batch. Each complete input creates a new
`<out-dir>/<sample_id>/` containing:

- `operand_roles.jsonl.gz`: header, instruction rows and complete footer.
- `imports.json`: full header import entries and symbol presence/reference sets.
- `manifest.json`: source/code/vocabulary hashes, counts and coverage.

Publication uses a temporary `.pending-*` directory and atomic rename. Existing
output is refused, not overwritten. Failed runs can leave pending diagnostic files,
but never a successful final directory. The source hash is rechecked at the end.

## Operand categories for Yanping

```python
from ida_feature_readers import iter_operand_roles

for row in iter_operand_roles("sample.jsonl.gz"):
    token_triplet = row["tokens"]  # [opcode_or_api, operand1, operand2]
```

Each instruction has the existing three-token format and vocabulary ranges
(`opcode/API < 256`, operand `< 76`). Function order and address-sorted block order
match the graph-cache builder. `fn_index`, `block_index`, `instruction_index`, original
function/block IDs and instruction-within-block are **alignment metadata**, not model
features. Empty functions advance the function index. Orphans are omitted by default;
`--include-orphans` / `include_orphans=True` adds them with null function/block indices.

| Reader role | Existing token ID | Evidence |
|---|---:|---|
| PADDING | 0 | Absent operand in the two-operand triplet |
| REG_DATA | 70 | Explicit non-stack register, without exporting its name |
| REG_STACK | 71 | Explicit SP/BP register family |
| MEM_STACK_REF | 72 | Only when explicit `address_registers` metadata proves SP/BP use |
| MEM_GLOBAL_REF | 73 | IDA direct `memory` operand category |
| IMM_VALUE | 74 | Immediate operand; numeric value is discarded |
| CODE_TARGET | 74 | Code target category, consistent with legacy numeric targets |
| MEM_UNKNOWN | 73 | Phrase/displacement without explicit base/index register metadata |
| UNKNOWN_OPERAND | 73 | Legacy fallback for unsupported/missing operand classification |

**Roles here mean operand categories, not read/write access.** Current 2.1 exports
do not provide operand access flags. They also lack explicit memory base/index names
and address-size mode. The reader does not reverse-engineer `phrase`/`processor_flags`
into confident stack/global labels. Token 73 was already the legacy memory/fallback
bucket; a separate `MEM_UNKNOWN` label and `role_basis` preserve uncertainty without
changing the model's embedding size. `MEM_STACK_REF` will usually be absent on the
current exports, even though the original assembly contains stack memory references.
The optional `address_registers` field is supported for augmented exports; this reader
does not add it or change the IDA extractor.

Rows preserve all operand roles in `operand_roles`/`operand_role_ids`; the legacy
triplet uses only the first two, with `extra_operands` recording the remainder.
No register identity, operand width, immediate value, displacement, address, family
or label is emitted as an input feature. Missing opcode entries use UNKNOWN_OPCODE.
Recognized imported CALL names map by exact normalized API membership; no substring
match. Unknown imported calls retain CALL, as in the existing encoder. Vocabulary
fitting or expansion, if desired later, must still use training folds only.

## Header imports and the unreferenced-import feature

```python
from ida_feature_readers import read_header_imports, read_import_features

header = read_header_imports("sample.jsonl.gz")  # fast; header only
features = read_import_features("sample.jsonl.gz")  # complete-export check
all_symbols = features["header_imports"]
unreferenced_symbols = features["unreferenced_imports"]
```

`read_header_imports` returns every normal/delay entry in header order, including
duplicates and ordinal imports. It explicitly reports `export_complete_verified=False`:
reading a header cannot validate the remainder of the file.

`read_import_features` scans **both function and orphan instructions**, validates
import indices and complete-footer counts, and derives:

- `header_imports`: distinct lowercased `dll!name` / `dll!#ordinal` identities.
- `referenced_imports`: identities with an observed instruction `import_ids` reference.
- `unreferenced_imports`: header identity set minus observed-reference identity set.
- `unreferenced_entry_indices`: the corresponding entry-level complement, which can
  differ from the symbol-level feature set when duplicate entries refer to one API.

Per-entry flags separate any reference from a call-site reference. Taking an import's
address counts as a reference. `delayed` remains available in the raw entry metadata,
and symbol presence combines normal/delay entries without changing API identity.

“Unused” is a convenient feature name, **not runtime evidence**: unresolved indirect
calls, dynamically resolved APIs and incomplete recovery can leave real usage unseen.
An unreferenced feature is therefore named `unreferenced_imports`. Do not silently
count only imports encountered while encoding function instructions, since that would
lose precisely the header-only imports this experiment needs.

Fit feature dictionaries and IDF on the training partition, then transform val/test.
Compare header-only, referenced-only and unreferenced-only imports on the same folds;
report import/reference coverage by architecture and corpus. This reader does not
train models or add its diagnostic coverage counts as model features.

## Checks and real smoke

```powershell
python -m pytest CT_GAT/preprocessing/test_ida_feature_readers.py -q
```

The real `instnm.exe` smoke produced 788 instruction rows, 541 data-register roles,
84 stack-register roles and 143 ambiguous memory roles. All 37 distinct header imports
were statically referenced, so this particular binary has zero unreferenced symbols.
Fixtures cover nonzero unreferenced imports, duplicate identities, delay imports,
ordinal imports, orphan-only references, extra operands and incomplete exports.

A second real smoke on Mendeley sample
`002dd9b7cbf8ca2a09434f8c4abd85631efe922ab8daa1219d86d83a0228aeda`
produced 36,769 instruction rows and 88 distinct header imports. All 88 had an
observed static reference. Both input hashes were unchanged. Outputs and per-sample
coverage manifests are under `reports/ida_reader_smoke/`; these are smoke artifacts,
not a corpus-wide feature build or a trained experiment.
