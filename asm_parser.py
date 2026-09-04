import os
import re
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1. FIXED SQUARE DESIGN CONFIGURATION
SQUARE_RESOLUTION = 256  
TOTAL_TOKEN_CAPACITY = SQUARE_RESOLUTION * SQUARE_RESOLUTION  
VIT_PATCH_SIZE = 16  # Matches typical Vision Transformer projection dimensions

# Binary dataset: goodware vs ransomware
FAMILY_NAMES = {
    '0': 'Goodware',
    '1': 'Ransomware',
}

# ==============================================================================
# 2. STRUCTURAL TOKEN-ID MAP & PARSING REGEX
# ==============================================================================

TOKEN_MAP = {
    'PADDING': 0, 'UNKNOWN_OPCODE': 1, 'NOP_SLED': 2,
    
    # Control Flow / Jumps
    'JMP': 10, 'JZ': 11, 'JNZ': 12, 'JE': 13, 'JNE': 14, 
    'JS': 15, 'JNS': 16, 'JG': 17, 'JL': 18, 'JGE': 19, 'JLE': 20,
    'CALL': 21, 'RET': 22, 'RETN': 23,
    
    # Data Movement
    'MOV': 25, 'PUSH': 26, 'POP': 27, 'LEA': 28, 'XCHG': 29,
    
    # Arithmetic Processing
    'ADD': 30, 'SUB': 31, 'INC': 32, 'DEC': 33, 'CMP': 34, 'TEST': 35,
    'IMUL': 36, 'MUL': 37, 'IDIV': 38, 'DIV': 39,
    
    # Logical / Bitwise Algebra
    'XOR': 40, 'OR': 41, 'AND': 42, 'NOT': 43, 
    'SHL': 44, 'SHR': 45, 'ROL': 46, 'ROR': 47, 'SAR': 48, 'SAL': 49,
    'NEG': 50,
    
    # Floating Point & Vectorized/Cryptographic Mixing
    'FSTP': 55, 'FLD': 56, 'PADD': 57, 'PXOR': 58, 'MOVDQA': 59,
    
    # System Level Interruption & Profiling
    'INT': 60, 'SYSCALL': 61, 'SYSENTER': 62,    
    'CPUID': 63, 'RDTSC': 64,
    
    # Operand Structural Tokens
    'REG_DATA': 70,       # Data registers (EAX, EBX, etc.)
    'REG_STACK': 71,      # Stack pointers (EBP, ESP)     
    'MEM_STACK_REF': 72,  # Memory references via EBP/ESP
    'MEM_GLOBAL_REF': 73, # Pure pointer brackets [0x401000]
    'IMM_VALUE': 74,       # Constants/Immediate values
    'UNKNOWN_API': 75     # Fallback for unlisted APIs
}

API_MAP = {
    'amsiscanbuffer': 76, 'etweventwrite': 77, 'nttraceevent': 78,
    'impersonatedloggedonuser': 79, 'duplicatetokenex': 80, 'adjusttokenprivileges': 81,
    'openprocesstoken': 82, 'deviceiocontrol': 83, 'ntdeviceiocontrolfile': 84,
    'cocreateinstance': 85, 'coinitalizeex': 86, 'clsidfromstring': 87,
    'ntopenprocesstoken': 88, 'openmutexa': 89, 'createmutexa': 90,
    'createprocesswithlogonw': 91, 'setthreadplaceholder': 92, 'switchtothread': 93,
    'registerwindowmessagea': 94, 'tpallocwork': 95, 'tppostwork': 96,
    'lookupprivilegevaluea': 97, 'ntadjustprivilegestoken': 98,  'ntcreatesection': 99,
    'virtualalloc': 100, 'virtualallocex': 101, 'writeprocessmemory': 102, 
    'createthread': 103, 'createremotethread': 104, 'ntmapviewofsection': 105,
    'virtualprotect': 106, 'virtualprotectex': 107, 'openprocess': 108,
    'ntallocatevirtualmemory': 109, 'ntwritevirtualmemory': 110, 'ntprotectvirtualmemory': 111,
    'ntcreatethreadex': 112, 'queueuserapc': 113, 'rtlcreateremotethread': 114,
    'setthreadcontext': 115, 'getthreadcontext': 116, 'ntopenprocess': 117,
    'ntsetcontextthread': 118, 'ntgetcontextthread': 119,
    'getprocaddress': 120, 'loadlibrarya': 121, 'loadlibraryw': 122, 
    'loadlibraryex': 123, 'ldrloaddll': 124, 'getmodulehandlea': 125, 'getmodulehandlew': 126,
    'getmodulefilenamea': 127, 'getmodulefilenamew': 128, 'ldrgetprocedureaddress': 129,
    'freelibrary': 130, 'exitprocess': 131, 'terminateprocess': 132,
    'ntterminateprocess': 133, 'ntqueryinformationprocess': 134, 'rtlgetversion': 135,
    'iswow64process': 136, 'getstartuptime': 137, 'getcommandlinea': 138, 'getcommandlinew': 139,
    'createfilea': 140, 'createfilew': 141, 'writefile': 142, 'readfile': 143, 
    'copyfilea': 144, 'copyfilew': 145, 'deletefilea': 146, 'deletefilew': 147,
    'getsystemdirectorya': 148, 'getsystemdirectoryw': 149, 'getwindowsdirectorya': 150,
    'getwindowsdirectoryw': 151, 'movefileexw': 152, 'movefileexa': 153,
    'getfilesize': 154, 'setfilepointer': 155, 'findfirstfilea': 156,
    'findnextfilea': 157, 'findfirstfilew': 158, 'findnextfilew': 159,
    'regsetvalueex': 160, 'regopenkeyex': 161, 'regcreatekeyex': 162, 
    'regqueryvalueex': 163, 'regclosekey': 164, 'regdeletevaluea': 165,
    'regdeletevaluew': 166, 'regenumkeyexw': 167, 'regenumkeyexa': 168,
    'ntopenkey': 169, 'ntsetvaluekey': 170, 'ntqueryvaluekey': 171,
    'shreggetvaluew': 172, 'shreggetvaluea': 173, 'regsavekeya': 174,
    'internetopena': 175, 'internetopenw': 176, 'internetconnecta': 177,
    'internetconnectw': 178, 'httpsendrequesta': 179, 'httpsendrequestw': 180,
    'internetreadfile': 181, 'internetwritefile': 182, 'wsastartup': 183,
    'socket': 184, 'connect': 185, 'send': 186, 'recv': 187,
    'wsasend': 188, 'wsarecv': 189, 'getaddrinfo': 190, 'gethostbyname': 191,
    'closesocket': 192, 'internetopenurla': 193, 'internetopenurlw': 194,
    'isdebuggerpresent': 195, 'checkremotedebuggerpresent': 196, 'gettickcount': 197,
    'gettickcount64': 198, 'sleep': 199, 'ntdelayexecution': 200,
    'outputdebugstringa': 201, 'outputdebugstringw': 202, 'ntquerysysteminformation': 203,
    'getsystemtime': 204, 'getlocaltime': 205, 'getasynckeystate': 206,
    'setwindowshookexa': 207, 'setwindowshookexw': 208, 'callnexthookex': 209,
    'process32first': 210, 'process32next': 211, 'createtoolhelp32snapshot': 212,
    'getcomputernamea': 213, 'getcomputernamew': 214, 'getusernamea': 215,
    'getusernamew': 216, 'getnativeid': 217, 'enumprocesses': 218,
    'enumprocessmodules': 219, 'getcurrentprocessid': 220, 'getcurrentthreadid': 221,
    'getlogicaldrives': 222, 'getdriveclass': 223, 'getdiskfreespaceexa': 224,
    'cryptacquirecontexta': 225, 'cryptacquirecontextw': 226, 'cryptcreatehash': 227, 
    'crypthashdata': 228, 'cryptderivekey': 229, 'cryptencrypt': 230,
    'cryptdecrypt': 231, 'cryptdestroykey': 232, 'cryptdestroyhash': 233,
    'cryptreleasecontext': 234, 'bcreptopenalgorithmprovider': 235, 'bcryptexecute': 236,
    'cryptgenrandom': 237, 'rtlcomputecrc32': 238, 'cryptstringtobinarya': 239,
    'shellexecutea': 240, 'shellexecutew': 241, 'createprocessa': 242, 
    'createprocessw': 243, 'resumethread': 244, 'suspendthread': 245,
    'unmapviewofsection': 246, 'findresourcea': 247, 'findresourcew': 248, 
    'loadresource': 249, 'lockresource': 250, 'gettemppatha': 251, 
    'gettemppathw': 252, 'winexec': 253, 'ntcreateuserprocess': 254, 'createprocessasuserw': 255
}

REG_DATA_REGEX = re.compile(
    r'\b(rax|rbx|rcx|rdx|rsi|rdi|r8|r9|r10|r11|r12|r13|r14|r15|'
    r'eax|ebx|ecx|edx|esi|edi|ax|bx|cx|dx|al|bl|cl|dl)\b',
    re.IGNORECASE,
)
REG_STACK_REGEX = re.compile(r'\b(rbp|rsp|ebp|esp|bp|sp)\b', re.IGNORECASE)
MEM_STACK_REGEX = re.compile(r'\[\s*(rbp|rsp|ebp|esp|bp|sp)\s*[\+\-].*\]', re.IGNORECASE)
MEM_REGEX = re.compile(r'\[.*\]')
IMM_REGEX = re.compile(r'\b(0x[0-9a-fA-F]+|[0-9]+h?)\b')

def categorize_operand(op_str):
    if not op_str: return TOKEN_MAP['PADDING']
    op_str = op_str.strip()
    if MEM_REGEX.search(op_str):
        if MEM_STACK_REGEX.search(op_str): return TOKEN_MAP['MEM_STACK_REF']
        return TOKEN_MAP['MEM_GLOBAL_REF']
    if REG_DATA_REGEX.search(op_str): return TOKEN_MAP['REG_DATA']
    if REG_STACK_REGEX.search(op_str): return TOKEN_MAP['REG_STACK']
    if IMM_REGEX.search(op_str): return TOKEN_MAP['IMM_VALUE']
    return TOKEN_MAP['MEM_GLOBAL_REF']

def parse_asm_line(line):
    line = line.strip()
    if not line or line.startswith(';') or ';' in line.split(): return None  
    if ';' in line: line = line.split(';')[0].strip()  
    tokens = line.split()
    if len(tokens) < 2 or ':' not in tokens[0]: return None
    if any(keyword in tokens[1].lower() for keyword in ['public', 'assume', 'proc', 'endp', 'segment', 'ends', 'unicode', 'extrn']): return None
    if any(directive in tokens for directive in ['db', 'dw', 'dd', 'align']): return None

    code_tokens = []
    for token in tokens[1:]:
        if re.match(r'^[0-9A-Fa-f]{2}$', token) or re.match(r'^[0-9A-Fa-f]{4}$', token): continue
        code_tokens.append(token)
        
    if not code_tokens: return None
    opcode = code_tokens[0].upper()
    operand_blob = "".join(code_tokens[1:])
    raw_operands = [o.strip() for o in operand_blob.split(',') if o.strip()]
    
    op_id = TOKEN_MAP['PADDING']
    if opcode == 'CALL':
        op_id = TOKEN_MAP['CALL']
        if raw_operands:
            target = raw_operands[0].lower()
            for api_name, api_id in API_MAP.items():
                if api_name in target:
                    op_id = api_id
                    break
            if op_id == TOKEN_MAP['CALL']: op_id = TOKEN_MAP['UNKNOWN_API']
    elif opcode == 'NOP': op_id = TOKEN_MAP['NOP_SLED']
    else: op_id = TOKEN_MAP.get(opcode, TOKEN_MAP['UNKNOWN_OPCODE'])
        
    op1_id = categorize_operand(raw_operands[0]) if len(raw_operands) > 0 else TOKEN_MAP['PADDING']
    op2_id = categorize_operand(raw_operands[1]) if len(raw_operands) > 1 else TOKEN_MAP['PADDING']
    
    return [op_id, op1_id, op2_id]

def load_labels(csv_path):
    label_lookup = {}
    if not os.path.exists(csv_path):
        print(f" Warning: Labels file not found at {csv_path}. Sorting will map to 'Unclassified'.")
        return label_lookup
        
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            if len(row) >= 2:
                file_id = row[0].strip().replace('"', '')  
                class_id = row[1].strip() 
                label_lookup[file_id] = class_id
    return label_lookup


def build_labels_from_source_dirs(asm_dir, goodware_dir, ransomware_dir, csv_path):
    """
    Label each .asm file by matching its stem against goodware / ransomware
    source executables, then write trainLabels.csv (Id,Class).
    Class 0 = Goodware, Class 1 = Ransomware.
    """
    goodware_stems = {
        p.stem for p in Path(goodware_dir).rglob('*') if p.is_file()
    }
    ransomware_stems = {
        p.stem for p in Path(ransomware_dir).rglob('*') if p.is_file()
    }

    labels = {}
    unknown = []
    for asm_path in Path(asm_dir).glob('*.asm'):
        stem = asm_path.stem
        if stem in goodware_stems:
            labels[stem] = '0'
        elif stem in ransomware_stems:
            labels[stem] = '1'
        else:
            unknown.append(stem)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Class'])
        for file_id, class_id in sorted(labels.items()):
            writer.writerow([file_id, class_id])

    print(
        f"Labels written to {csv_path}: "
        f"{sum(v == '0' for v in labels.values())} goodware, "
        f"{sum(v == '1' for v in labels.values())} ransomware, "
        f"{len(unknown)} unlabeled"
    )
    if unknown:
        print(f"  Unlabeled examples: {unknown[:5]}")
    return labels

def process_single_file(asm_path, output_dir, file_class_id):
    """
    RESEARCH CORE UPGRADE: Fixed Canvas Context-Aware Texture Masking.
    Protects Self-Attention blocks from background leakage, avoids edge-gradients,
    and scales cleanly to track adversarial line additions.
    """
    flattened_token_stream = []
    
    with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            triplet = parse_asm_line(line)
            if triplet:
                flattened_token_stream.extend(triplet)
                
    total_tokens = len(flattened_token_stream)
    if total_tokens == 0: 
        return "Empty Stream"
        
    token_array_1d = np.array(flattened_token_stream, dtype=np.uint8)

    # PIPELINE REFACTOR: Contextual Optimization Boundary Management
    if total_tokens > TOTAL_TOKEN_CAPACITY:
        # Strict truncation to fit the uniform 256x256 grid
        final_array = token_array_1d[:TOTAL_TOKEN_CAPACITY]
    elif total_tokens < TOTAL_TOKEN_CAPACITY:
        # 1. Extrapolate authentic internal statistical profile
        mean_val = np.mean(token_array_1d)
        std_val = np.std(token_array_1d) if np.std(token_array_1d) > 0 else 1.0
        
        # 2. Fabricate low-variance texture padding block matching token profile
        padding_needed = TOTAL_TOKEN_CAPACITY - total_tokens
        noise_padding = np.random.normal(loc=mean_val, scale=std_val * 0.15, size=padding_needed)
        noise_padding = np.clip(noise_padding, 0, 255).astype(np.uint8)
        
        # 3. Concatenate real stream seamlessly with texturized matching noise background
        final_array = np.concatenate([token_array_1d, noise_padding])
    else:
        final_array = token_array_1d

    # 4. Generate visual representation image
    img_matrix = final_array.reshape((SQUARE_RESOLUTION, SQUARE_RESOLUTION))
    
    folder_suffix = FAMILY_NAMES.get(file_class_id, f"Class_{file_class_id}" if file_class_id else "Unclassified")
    class_directory = os.path.join(output_dir, f"Class_{file_class_id}_{folder_suffix}")
    os.makedirs(class_directory, exist_ok=True)
        
    base_name = os.path.splitext(os.path.basename(asm_path))[0]
    final_output_path = os.path.join(class_directory, f"{base_name}.png")
    
    img = Image.fromarray(img_matrix, mode='L')
    img.save(final_output_path)

    # 5. DYNAMIC HIERARCHICAL ATTENTION MASK CALCULATOR (ViT Track)
    # Generates a binary coordinate array matching standard 16x16 patch sizes
    token_mask_1d = np.zeros(TOTAL_TOKEN_CAPACITY, dtype=np.uint8)
    token_mask_1d[:min(total_tokens, TOTAL_TOKEN_CAPACITY)] = 1
    mask_matrix = token_mask_1d.reshape((SQUARE_RESOLUTION, SQUARE_RESOLUTION))
    
    patches_per_side = SQUARE_RESOLUTION // VIT_PATCH_SIZE
    patch_attention_mask = np.zeros((patches_per_side, patches_per_side), dtype=np.uint8)
    
    for r in range(patches_per_side):
        for c in range(patches_per_side):
            # Sample localized square window block segment
            patch_block = mask_matrix[
                r * VIT_PATCH_SIZE : (r + 1) * VIT_PATCH_SIZE, 
                c * VIT_PATCH_SIZE : (c + 1) * VIT_PATCH_SIZE
            ]
            # If the patch contains even 1 authentic code instruction byte, mark as active
            if np.any(patch_block == 1):
                patch_attention_mask[r, c] = 1
                
    # Save the companion array directly adjacent to the matrix PNG image
    mask_output_path = os.path.join(class_directory, f"{base_name}_vit_mask.npy")
    np.save(mask_output_path, patch_attention_mask)
    
    return folder_suffix

def batch_process_and_sort_directory(source_dir, destination_dir, labels_csv):
    labels_dictionary = load_labels(labels_csv)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        
    asm_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.asm')]
    total_files = len(asm_files)
    
    max_cores = os.cpu_count()
    print(f"Initialization Complete. Found {total_files} .asm files in queue.")
    print(f"🚀 MULTI-CORE ENGAGED: Spawning Worker Pool over {max_cores} CPU cores...")
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = {}
        for filename in asm_files:
            source_file_path = os.path.join(source_dir, filename)
            base_name, _ = os.path.splitext(filename)
            target_class = labels_dictionary.get(base_name, None)
            
            future = executor.submit(process_single_file, source_file_path, destination_dir, target_class)
            futures[future] = base_name
            
        for idx, future in enumerate(as_completed(futures), 1):
            base_name = futures[future]
            try:
                family_info = future.result()
                print(f"[{idx}/{total_files}] Transformation Pipeline Success: {base_name}.png + _vit_mask.npy ➔ [{family_info}]")
            except Exception as e:
                print(f"[-] [{idx}/{total_files}] Critical Processing Failure on {base_name}: {str(e)}")

# ==============================================================================
# 4. RUNTIME SYSTEM EXECUTION WINDOW
# ==============================================================================
if __name__ == "__main__":
    # Capstone .asm files produced by asm_parse.py
    INPUT_ASM_DIRECTORY = "/home/yl/quarantine/extract/dataset_asm"
    OUTPUT_SORTED_IMAGES = "/home/yl/quarantine/extract/training_dataset_sorted"
    LABELS_CSV_PATH = "/home/yl/quarantine/extract/trainLabels.csv"

    # Source PE trees used to assign Class 0 / Class 1
    GOODWARE_DIR = "/home/yl/quarantine/extract/goodware"
    RANSOMWARE_DIR = "/home/yl/quarantine/extract/ransomware"

    if not os.path.exists(INPUT_ASM_DIRECTORY):
        raise SystemExit(f"ASM directory not found: {INPUT_ASM_DIRECTORY}")

    build_labels_from_source_dirs(
        INPUT_ASM_DIRECTORY,
        GOODWARE_DIR,
        RANSOMWARE_DIR,
        LABELS_CSV_PATH,
    )
    batch_process_and_sort_directory(
        INPUT_ASM_DIRECTORY,
        OUTPUT_SORTED_IMAGES,
        LABELS_CSV_PATH,
    )
