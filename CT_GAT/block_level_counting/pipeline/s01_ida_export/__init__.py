"""Step 1, IDA export: one `<sha256>.jsonl.gz` per binary (schema ida-cfg-research/2.1: header, one record per function
with its blocks, instructions and CFG edges, footer). Runs on the VM, not in this repo; see README.md.
Output feeds step 2. No RunConfig field."""
