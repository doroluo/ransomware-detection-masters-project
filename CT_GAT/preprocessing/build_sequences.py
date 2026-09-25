"""Batch-parse ASM files from metadata.csv into transformer-ready sequences.

Skips empty ASM files. Saves one .pt per sample:
  {file_id, sequence: LongTensor[N, 4], label, asm_path}
  where each row is [opcode_or_api, operand1, operand2, behavior].
"""

from pathlib import Path
import pandas as pd
import torch

from token_mapping import parse_asm_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "sequences"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METADATA_PATH)

    successful = 0
    skipped_empty = 0
    failed = 0

    for _, row in df.iterrows():
        file_id = str(row["file_id"])
        label = int(row["label"])
        asm_path = Path(str(row["asm_path"]))

        if not asm_path.is_absolute():
            asm_path = PROJECT_ROOT / asm_path

        try:
            if not asm_path.is_file():
                print(f"Missing asm: {asm_path}")
                failed += 1
                continue

            instructions = parse_asm_file(asm_path)

            if len(instructions) == 0:
                print(f"Skipped empty file: {asm_path.name}")
                skipped_empty += 1
                continue

            sequence = torch.tensor(instructions, dtype=torch.long)
            output_path = OUTPUT_DIR / f"{file_id}.pt"

            torch.save(
                {
                    "file_id": file_id,
                    "sequence": sequence,
                    "label": label,
                    "asm_path": str(asm_path),
                },
                output_path,
            )

            print(
                f"Saved {file_id}: "
                f"{sequence.shape[0]} instructions, label={label}"
            )
            successful += 1

        except Exception as error:
            print(f"Failed: {asm_path}")
            print("Error:", error)
            failed += 1

    print("\nFinished.")
    print("Successful:", successful)
    print("Skipped empty:", skipped_empty)
    print("Failed:", failed)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
