import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from extractor import extract_features_from_folder, prune_brittle_features

def train_lightgbm(X, y):
    """Trains a LightGBM classifier on extracted PE features."""
    dataset = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    return lgb.train(params, dataset, num_boost_round=100)

def simulate_targeted_mimicry(X_ransomware, X_goodware_pool, alpha=0.4):
    """
    Simulates realistic evasion by blending ONLY malleable feature spaces 
    (byte/entropy histograms, string metadata, and import padding) with 
    randomly paired benign samples. Header invariants remain intact.
    """
    X_adv = X_ransomware.copy()
    n_ransom = len(X_ransomware)
    n_good = len(X_goodware_pool)

    # Randomly pair each ransomware sample with a benign target
    random_indices = np.random.choice(n_good, size=n_ransom, replace=True)
    targets = X_goodware_pool[random_indices]

    # 1. Blend Malleable Byte & Entropy Histograms (Indices 0:512)
    X_adv[:, 0:512] = (1.0 - alpha) * X_adv[:, 0:512] + alpha * targets[:, 0:512]

    # 2. Simulate Import Table Padding (Indices 526-527: imports and DLL counts)
    # Attackers add benign imports to match higher import counts
    X_adv[:, 526] = np.maximum(X_adv[:, 526], targets[:, 526])
    X_adv[:, 527] = np.maximum(X_adv[:, 527], targets[:, 527])

    # 3. Blend Malleable String Statistics (Indices -4:)
    X_adv[:, -4:] = (1.0 - alpha) * X_adv[:, -4:] + alpha * targets[:, -4:]

    return X_adv

def evaluate_model(model, X_clean_test, y_clean_test, X_adv_test):
    """Calculates clean precision, clean recall, and adversarial evasion rate."""
    preds_clean = (model.predict(X_clean_test) >= 0.5).astype(int)
    precision = precision_score(y_clean_test, preds_clean, zero_division=0)
    recall = recall_score(y_clean_test, preds_clean, zero_division=0)

    if len(X_adv_test) > 0:
        preds_adv = (model.predict(X_adv_test) >= 0.5).astype(int)
        evasion_rate = np.mean(preds_adv == 0)
    else:
        evasion_rate = 0.0

    return precision, recall, evasion_rate

def main():
    if os.path.exists("X_clean.npy") and os.path.exists("y_clean.npy"):
        print("[*] Loading cached feature matrices from disk...")
        X_clean = np.load("X_clean.npy")
        y_clean = np.load("y_clean.npy")
    else:
        print("=== Phase 1: Extracting Native PE Features ===")
        X_good, y_good = extract_features_from_folder("./goodware", label=0, sanitize=False)
        X_ransom, y_ransom = extract_features_from_folder("./ransomware", label=1, sanitize=False)

        if len(X_good) == 0 or len(X_ransom) == 0:
            print("[!] Verification failed. Check `./goodware` and `./ransomware` paths.")
            return

        X_clean = np.vstack([X_good, X_ransom])
        y_clean = np.hstack([y_good, y_ransom])

        np.save("X_clean.npy", X_clean)
        np.save("y_clean.npy", y_clean)
        print("[+] Feature matrices cached to `X_clean.npy` and `y_clean.npy`.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    # Separate Goodware training pool for mimicry targets
    goodware_train_pool = X_train[y_train == 0]

    # Test set: Heavy Mimicry (alpha=0.6) on isolated test ransomware
    ransom_test_mask = (y_test == 1)
    X_ransom_test = X_test[ransom_test_mask]
    X_adv_test = simulate_targeted_mimicry(X_ransom_test, goodware_train_pool, alpha=0.6)

    print(f"[*] Dataset Split: {len(X_train)} Train, {len(X_test)} Test samples.")
    print(f"[*] Generated targeted mimicry attacks on {len(X_ransom_test)} test ransomware samples.")

    print("\n=== Phase 2: Benchmarking Model Defenses ===")

    # 1. Baseline Model
    print("[*] Training Baseline LightGBM Model...")
    baseline_model = train_lightgbm(X_train, y_train)
    b_prec, b_rec, b_evas = evaluate_model(baseline_model, X_test, y_test, X_adv_test)

    # 2. Defense Strategy: Feature Pruning
    print("[*] Evaluating Defense 1: Feature Pruning...")
    X_tr_pruned = prune_brittle_features(X_train)
    X_te_pruned = prune_brittle_features(X_test)
    X_adv_pruned = prune_brittle_features(X_adv_test)

    pruned_model = train_lightgbm(X_tr_pruned, y_train)
    p_prec, p_rec, p_evas = evaluate_model(pruned_model, X_te_pruned, y_test, X_adv_pruned)

    # 3. Defense Strategy: Adversarial Retraining (Augment with Moderate Mimicry alpha=0.3)
    print("[*] Evaluating Defense 2: Adversarial Retraining (Trained on alpha=0.3, Tested on alpha=0.6)...")
    ransom_train_mask = (y_train == 1)
    X_adv_train = simulate_targeted_mimicry(X_train[ransom_train_mask], goodware_train_pool, alpha=0.3)
    y_adv_train = np.ones(len(X_adv_train), dtype=int)

    X_aug_train = np.vstack([X_train, X_adv_train])
    y_aug_train = np.hstack([y_train, y_adv_train])

    retrained_model = train_lightgbm(X_aug_train, y_aug_train)
    r_prec, r_rec, r_evas = evaluate_model(retrained_model, X_test, y_test, X_adv_test)

    print("\n" + "="*70)
    print(f"{'Strategy / Model Architecture':<30} | {'Clean Prec':<10} | {'Clean Rec':<10} | {'Evasion Rate':<12}")
    print("="*70)
    print(f"{'Baseline LightGBM':<30} | {b_prec:<10.4f} | {b_rec:<10.4f} | {b_evas*100:<11.2f}%")
    print(f"{'Feature Pruning Defense':<30} | {p_prec:<10.4f} | {p_rec:<10.4f} | {p_evas*100:<11.2f}%")
    print(f"{'Adversarial Retraining':<30} | {r_prec:<10.4f} | {r_rec:<10.4f} | {r_evas*100:<11.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
    