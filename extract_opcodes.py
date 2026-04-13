#!/usr/bin/env python3
import os
import sys
import subprocess

def prepare_hash_list(source_dir):
    """
    Automatically prepares the hash list by looking inside the specified source directory.
    """
    print(f"[*] Scanning {source_dir} for binaries...")
    
    if not os.path.exists(source_dir):
        print(f"[!] Error: Directory '{source_dir}' not found.")
        sys.exit(1)

    # List files in the source directory, excluding common metadata/scripts
    cmd = f"ls {source_dir} | grep -vE '\\.zip|\\.py|\\.txt|\\.md' > hashes.txt"
    subprocess.run(cmd, shell=True)
    
    if not os.path.exists("hashes.txt") or os.path.getsize("hashes.txt") == 0:
        print(f"[!] Error: No files found in '{source_dir}'.")
        sys.exit(1)

def run_extraction(source_dir, label):
    # Set output directory
    output_dir = f"./opcode_data/{label}"
    os.makedirs(output_dir, exist_ok=True)

    with open("hashes.txt", "r") as f:
        hashes = [line.strip() for line in f if line.strip()]

    print(f"[*] Found {len(hashes)} files in {source_dir}. Starting extraction...")

    for h in hashes:
        # Construct the full path to the binary inside its source folder
        file_path = os.path.join(source_dir, h)
        
        if os.path.exists(file_path):
            # Using the mingw32 disassembler objdump
            out_path = f"{output_dir}/{h}.txt"
            cmd = f"x86_64-w64-mingw32-objdump -d {file_path} | grep -Po '^\\s+[0-9a-f]+:\\s+[0-9a-f ]+\\s+\\K[a-z]+' > {out_path}"
            
            result = os.system(cmd)
            
            if result == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"  [+] Success: {h}")
            else:
                if os.path.exists(out_path):
                    os.remove(out_path)
                print(f"  [-] Failed: {h} (Likely packed or invalid PE format)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_opcodes.py [0|1]")
        print("  0: Extract from ./goodware_files/")
        print("  1: Extract from ./ransomware_files/")
        sys.exit(1)

    choice = sys.argv[1]

    if choice == "0":
        prepare_hash_list("goodware_files")
        run_extraction("goodware_files", "goodware")
    elif choice == "1":
        prepare_hash_list("ransomware_files")
        run_extraction("ransomware_files", "ransomware")
    else:
        print("[!] Invalid input. Use 0 or 1.")
