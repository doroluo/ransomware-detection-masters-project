import os
import pefile
from capstone import *
import pandas as pd

'''
extract.py
This script extracts the opcode + operand pairs from the executable sections of the files in the directories.
The output is saved in the BASE_OUTPUT_DIR directory.
The output is saved in the following format:
<family_name>_<filename>.txt
The family_name is the name of the family of the file.
The filename is the name of the file.
The opcode + operand pairs are saved in the following format:
<opcode> <operand>\n<opcode> <operand>\n...

Please use
find . -type f \( -name "*.exe" -o -name "*.dll" \) -exec upx -d {} +
before using this script
'''

# Change directories based on dataset
DIRECTORIES = {
    'good_train': './Goodware_Training/goodware',
    'good_test':  './Goodware_Test/goodware_test',
    'mal_train':  './Ransomware_Training/rans',
    'mal_test':   './Ransomware_Test/rans_test'
}

BASE_OUTPUT_DIR = './Features_Extraction'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def is_dotnet(pe):
    """Checks if the PE file is a .NET assembly."""
    try:
        com_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR']]
        if com_dir.VirtualAddress != 0 and com_dir.Size > 0:
            return True
    except (IndexError, AttributeError, KeyError):
        pass
    return False

def extract_instructions(file_path):
    """Extracts opcode + operand pairs from executable sections."""
    instructions = []
    try:
        pe = pefile.PE(file_path, fast_load=True)
        
        if is_dotnet(pe):
            return "Error: .NET Binary (Managed Code)"

        for section in pe.sections:
            if b'UPX' in section.Name:
                return "Error: UPX Compressed (Needs Unpacking)"
        
        if pe.FILE_HEADER.Machine == 0x014c:  # x86
            md = Cs(CS_ARCH_X86, CS_MODE_32)
        elif pe.FILE_HEADER.Machine == 0x8664:  # x64
            md = Cs(CS_ARCH_X86, CS_MODE_64)
        else:
            return f"Error: Unsupported Architecture ({hex(pe.FILE_HEADER.Machine)})"

        found_code = False
        for section in pe.sections:
            if section.Characteristics & 0x20000000: # IMAGE_SCN_MEM_EXECUTE
                found_code = True
                code_data = section.get_data()
                for i in md.disasm(code_data, section.VirtualAddress):
                    full_instruction = f"{i.mnemonic} {i.op_str}".strip()
                    instructions.append(full_instruction)
        
        if not found_code:
            return "Error: No sections marked as Executable"
        
        return "\n".join(instructions) if instructions else "Error: Section found but disassembly returned empty"
    
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN EXECUTION ---
for category, input_folder in DIRECTORIES.items():
    print(f"\n--- Processing Category: {category} ---")
    
    category_output_path = os.path.join(BASE_OUTPUT_DIR, category)
    os.makedirs(category_output_path, exist_ok=True)

    summary_data = []
    skipped_data = []

    if not os.path.exists(input_folder):
        print(f"Directory {input_folder} not found. Skipping...")
        continue

    for root, dirs, files in os.walk(input_folder):
        # Get the relative path to use as a family name
        rel_path = os.path.relpath(root, input_folder)
        # If it's the root folder, use 'root' or empty, else replace slashes for filenames
        family_prefix = "root" if rel_path == "." else rel_path.replace(os.sep, "_")

        for filename in files:
            file_path = os.path.join(root, filename)
            
            if filename.startswith('.'):
                continue

            try:
                with open(file_path, 'rb') as f:
                    header = f.read(2)
                if header != b'MZ':
                    continue # Not a windows executable
            except Exception:
                continue

            print(f"Analyzing: {family_prefix} -> {filename}")
            
            instr_sequence = extract_instructions(file_path)

            if not instr_sequence.startswith("Error"):
                # NEW: Prefix the family name to the filename
                txt_filename = f"{family_prefix}_{filename}.txt"
                txt_file_path = os.path.join(category_output_path, txt_filename)
                
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write(instr_sequence)
                
                summary_data.append({
                    'original_filename': filename,
                    'family_label': family_prefix,
                    'opcode_file': txt_filename,
                    'label': 1 if 'mal' in category else 0
                })
            else:
                skipped_data.append({
                    'filename': filename,
                    'family_label': family_prefix,
                    'reason': instr_sequence.replace("Error: ", "")
                })
                print(f"  [!] Skipped {filename}: {instr_sequence}")

    # Save outputs
    if summary_data:
        pd.DataFrame(summary_data).to_csv(
            os.path.join(BASE_OUTPUT_DIR, f"{category}_index.csv"), index=False
        )
    
    if skipped_data:
        skipped_file = os.path.join(BASE_OUTPUT_DIR, f"{category}_skipped_log.csv")
        pd.DataFrame(skipped_data).to_csv(skipped_file, index=False)
        print(f"Skipped log saved to: {skipped_file}")

print("\nAll extractions complete with family prefixing.")
