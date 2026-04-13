## 📄 README: Static Opcode Extraction Pipeline

### Project Overview
This pipeline automates the extraction of mnemonic opcode sequences from Windows-based ransomware and goodware binaries. It converts dangerous ransomware samples into safe, non-executable text files for use in Machine Learning models.

### 1. Prerequisites
Ensure your Ubuntu VM has the necessary tools for Windows binary disassembly and zip handling:
```bash
sudo apt update
sudo apt install -y binutils-mingw-w64 p7zip-full python3-pip
```

### 2. Environment Setup
**Isolation:** Ensure your VM has no network access to your host machine.

### 3. Downloading & Unzipping VirusShare Samples
**Searching for Ransomware Samples:**
```
ransomware detgte:30 filetype:PE32+ after:"2025-01-01"
```

**Searching for Goodware Samples:**
```
Unable to successfully query any goodware files. Instead use this promising dataset: https://data.mendeley.com/datasets/yzhcvn7sj5/1
```

**Unzipping Sample Files on the VM:**
1. You must use the password provided by VirusShare to unzip your files.
2. Make sure the unzipped files are in the same directory as the extract_opcodes.py file

### 4. Running the Extraction
1. **Run Permissions (Only need to run this once):**
   ```bash
    chmod +x extract_opcodes.py
   ```
2. **Execute:**
   The execution prompt will ask if you want to extract a goodware (0) or ransomware (1) hash(es).
   ```bash
   ./extract_opcode.py [0|1]
   ```
3. **Output:** The extracted sequences will be stored in the `./opcode_data/` folder as `.txt` files.

### 5. Cleaning Up (Safety)
Once the `.txt` files are generated, the original binaries are no longer needed. To prevent accidental execution:
```bash
rm 75cac1fd162460052ba8de72e3c571ff944cae93a91fb5cbadffee623374c1f6...
```
