import os
import re
import pandas as pd
import pickle
from collections import Counter
from nltk.util import bigrams
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split

# --- Part 1: Normalization and Loading ---
# https://github.com/huggingface/tokenizers/issues/376

def normalize_instruction(line):
    line = line.strip().lower()
    if not line:
        return None
    
    # Standardize numeric values first
    line = re.sub(r'0x[0-9a-f]+', '<HEX>', line)
    line = re.sub(r'\b\d+\b', '<OFFSET>', line)
    
    # Normalize internal bracket spacing []
    # This ensures [ebp - <HEX>] and [ebp+<HEX>] become consistent
    line = re.sub(r'\[\s*(.*?)\s*\]', r'[\1]', line) # Removes outer spaces: [ ebp ] -> [ebp]
    line = re.sub(r'\s*([-+])\s*', r'\1', line)      # Removes spaces around operators: ebp + <HEX> -> ebp+<HEX>
    
    # Standardize overall spacing
    token = re.sub(r'[\s,]+', ' ', line).strip()
    return token

def load_raw_with_operands(base_path, split='train'):
    """
    Reads raw .txt malware/benign files and organizes them into a labeled DataFrame.
    """
    if split == 'train':
        data_categories = {'mal_train': 1, 'good_train': 0}
    else:
        data_categories = {'mal_test': 1, 'good_test': 0}

    all_data = []
    print(f"Reading raw files from {base_path}({split})...")
    
    for folder, label in data_categories.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    lines = f.readlines()
                    # Normalize each instruction
                    tokens = [normalize_instruction(l) for l in lines if normalize_instruction(l)]
                    
                    # Limit sequence length for project consistency
                    tokens = tokens[:5000] 
                    
                    # Join with spaced separator for pre-tokenizer safety
                    opcodes_text = ' <SEP> '.join(tokens)
                    all_data.append({'Opcodes': opcodes_text, 'Malware': label})

    return pd.DataFrame(all_data)

# --- Part 2: Tokenizer Training Logic ---

def batch_iterator(data_series, batch_size=100):
    """
    Yields batches from a Pandas Series to keep memory usage low.
    """
    for i in range(0, len(data_series), batch_size):
        yield data_series[i : i + batch_size]

def prepare_tokenizer_trainer(alg, v_size, unk_token, spl_tokens):
    """
    Initializes the tokenizer and adds special tokens to prevent fragmentation.
    """
    protected_tokens = [AddedToken(t, single_word=True, normalized=False) for t in spl_tokens]
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=protected_tokens, vocab_size=v_size)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=protected_tokens, vocab_size=v_size)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=unk_token, special_tokens=protected_tokens, vocab_size=v_size)

    # Updated regex to capture <TAGS>, [MEM_REFS], and standard words
    # example of MEM_REFS: [ebp-0x4]
    tag_regex = r"<[^>]+>|\[[^\]]+\]|\w+"
    tokenizer.pre_tokenizer = Sequence([Split(pattern=tag_regex, behavior="isolated"), Whitespace()])
    tokenizer.add_special_tokens(protected_tokens)
    
    return tokenizer, trainer

def train_tokenizer(alg, train_data, v_size, unk_token, spl_tokens, batch_size):
    """
    Trains the requested tokenizer algorithm on the opcode series[cite: 3].
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg, v_size, unk_token, spl_tokens)
    tokenizer.train_from_iterator(batch_iterator(train_data['Opcodes'], batch_size), trainer=trainer)
    return tokenizer

def encode_and_save(tokenizer, df, name, output_dir='../New/TokenizedData'):
    """
    Encodes the dataframe and saves it as a pickle file[cite: 3].
    """
    encoded = tokenizer.encode_batch(df['Opcodes'])
    df_copy = df.copy()
    df_copy['Opcodes'] = [e.tokens for e in encoded]
    
    output_path = os.path.join(output_dir, f"{name}.pkl")
    df_copy.to_pickle(output_path)
    print(f"Saved {name} to {output_path}")

# --- Part 3: Main Execution ---

def main():
    # 1. Setup Directories
    for folder in ['../New/TokenizedData', '../New/Tokenizers']:
        os.makedirs(folder, exist_ok=True)

    # 2. Load Data
    train = load_raw_with_operands('../dataset/', split='train')
    test = load_raw_with_operands('../dataset/', split='test')

    if train.empty:
        print("Error: No training data found.")
        return

    # 3. Parameters
    batch_size = 1000
    unk_token = "<UNK>"
    # Include your research-specific semantic tags
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<HEX>", "<OFFSET>"]
    v_size = 2000

    # 4. Train and Run BPE
    print("Starting BPE Training...")
    bpe_tokenizer = train_tokenizer('BPE', train, v_size, unk_token, spl_tokens, batch_size)
    bpe_tokenizer.save("../New/Tokenizers/BPE-trained.json")
    
    encode_and_save(bpe_tokenizer, train, "BPE_train")
    encode_and_save(bpe_tokenizer, test, "BPE_test")

    # 5. Train and Run WordPiece
    print("Starting WordPiece Training...")
    wpc_tokenizer = train_tokenizer('WPC', train, v_size, unk_token, spl_tokens, batch_size)
    wpc_tokenizer.save("../New/Tokenizers/WPC-trained.json")
    
    encode_and_save(wpc_tokenizer, train, "WPC_train")
    encode_and_save(wpc_tokenizer, test, "WPC_test")

if __name__ == "__main__":
    main()