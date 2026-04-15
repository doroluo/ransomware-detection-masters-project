import os
import lief
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def extract_modern_pe_features(file_path):
    """
    Extracts a fixed-size numerical feature vector from a .exe file using modern LIEF.
    """
    try:
        binary = lief.PE.parse(file_path)
        if not binary:
            return None

        features = []
        
        # --- 1. Header & General Features ---
        features.append(binary.optional_header.sizeof_code)
        features.append(binary.optional_header.sizeof_image)
        features.append(binary.optional_header.sizeof_headers)
        features.append(binary.optional_header.sizeof_uninitialized_data)
        features.append(1 if binary.has_debug else 0)
        features.append(1 if binary.has_relocations else 0)
        features.append(1 if binary.has_signatures else 0)
        features.append(1 if binary.has_tls else 0)
        
        # --- 2. Section Features (Crucial for Ransomware/Packers) ---
        features.append(len(binary.sections))
        entropies = [section.entropy for section in binary.sections]
        features.append(float(np.mean(entropies)) if entropies else 0.0)
        features.append(float(np.max(entropies)) if entropies else 0.0)
        features.append(float(np.min(entropies)) if entropies else 0.0)
        
        # --- 3. Import/Export Features ---
        features.append(1 if binary.has_imports else 0)
        features.append(len(binary.imports) if binary.has_imports else 0)
        
        # Count total imported functions from DLLs
        num_imported_funcs = 0
        if binary.has_imports:
            for lib in binary.imports:
                num_imported_funcs += len(lib.entries)
        features.append(num_imported_funcs)
        
        features.append(1 if binary.has_exports else 0)

        return np.array(features, dtype=np.float32)

    except Exception as e:
        # Silently catch malformed binaries without crashing the script
        return None

def extract_features_from_folder(folder_path, label):
    X = []
    y = []
    
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return X, y

    all_filepaths = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            all_filepaths.append(os.path.join(root, filename))
            
    print(f"Processing {len(all_filepaths)} files in {folder_path}...")
    
    for file_path in all_filepaths:
        try:
            features = extract_modern_pe_features(file_path)
            
            if features is not None:
                X.append(features)
                y.append(label)
        except Exception:
            # Silently skip files that cause hard crashes in the parser
            continue
            
    return X, y

if __name__ == "__main__":
    lief.logging.disable()
    print("Starting Feature Extraction...")
    
    X_goodware, y_goodware = extract_features_from_folder("./goodware", 0)
    X_ransomware, y_ransomware = extract_features_from_folder("./ransomware/rans", 1)
    
    X = X_goodware + X_ransomware
    y = y_goodware + y_ransomware
    
    if len(X) == 0:
        raise ValueError("Error: No valid .exe files successfully parsed. Check your directories.")
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nExtraction Complete: {len(X_goodware)} Goodware | {len(X_ransomware)} Ransomware")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining LightGBM Model...")
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("\nEvaluating Model...")
    y_pred = clf.predict(X_test)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Goodware', 'Ransomware']))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")