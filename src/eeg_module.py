import numpy as np
import os
import json
import hashlib
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TEMPLATE_DIR = "templates"
VAULT_DIR = "vault"
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(VAULT_DIR, exist_ok=True)

#enrolment phase
def simulate_eeg_data(task, length=256, channels=8, noise_level=0.05):
    if task.lower() == "reading":
        base = np.sin(np.linspace(0, 10, length)).reshape(-1, 1) * np.ones((1, channels))
    elif task.lower() == "imagination":
        base = np.cos(np.linspace(0, 10, length)).reshape(-1, 1) * np.ones((1, channels))
    elif task.lower() == "writing":
        base = np.tan(np.linspace(0, 1, length)).reshape(-1, 1) * np.ones((1, channels))
        base = np.clip(base, -1, 1)
    else:
        base = np.zeros((length, channels))
    noise = noise_level * np.random.randn(length, channels)
    return base + noise

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def extract_features(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max_val = np.max(data, axis=0)
    min_val = np.min(data, axis=0)
    ptp = np.ptp(data, axis=0)
    var = np.var(data, axis=0)
    return np.concatenate([mean, std, max_val, min_val, ptp, var])



def enroll_user(user_id, task, samples=10):
    features = []
    for _ in range(samples):
        eeg = simulate_eeg_data(task)
        filtered = normalize_data(eeg)
        feats = extract_features(filtered)
        features.append(feats)

    template = {
        "user_id": user_id,
        "task": task,
        "features": np.array(features).tolist()
    }

    file_path = os.path.join(TEMPLATE_DIR, f"{user_id}_{task}.json")
    with open(file_path, "w") as f:
        json.dump(template, f)

    print(f"✅ Enrollment complete for '{user_id}' task: '{task}'")
    return True  # <-- 🔧 Add this line


def load_templates(user_id):
    X, y = [], []
    for fname in os.listdir(TEMPLATE_DIR):
        if fname.startswith(user_id):
            with open(os.path.join(TEMPLATE_DIR, fname), "r") as f:
                data = json.load(f)
                feats = np.array(data["features"])
                X.append(feats)
                y.extend([data["task"]] * feats.shape[0])
    X = np.vstack(X)
    return X, np.array(y)


def train_svm_classifier(X, y):
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(X_reduced, y)
    return clf, pca

# Authentication phase
def authenticate_user(user_id, claimed_task):
    eeg = simulate_eeg_data(claimed_task)
    filtered = normalize_data(eeg)
    test_features = extract_features(filtered).reshape(1, -1)

    X_train, y_train = load_templates(user_id)
    clf, pca = train_svm_classifier(X_train, y_train)

    test_reduced = pca.transform(test_features)
    predicted_task = clf.predict(test_reduced)[0]
    confidence = clf.predict_proba(test_reduced).max()

    print(f"🔎 Predicted: {predicted_task} | Confidence: {confidence:.2f}")
    return predicted_task.lower() == claimed_task.lower() and confidence > 0.80, test_features, predicted_task, confidence

#Key Generation Phase
def select_top_k_features(features, k=16):
    variances = np.var(features, axis=0)
    # Only keep features with some variance
    valid_indices = np.where(variances > 1e-6)[0]

    if len(valid_indices) < k:
        top_k_indices = np.argsort(variances)[-k:]
    else:
        top_k_variances = np.argsort(variances[valid_indices])[-k:]
        top_k_indices = valid_indices[top_k_variances]

    selected = features[:, top_k_indices]
    return selected

def binarize_features(features):
    # Add small noise
    noisy = features + np.random.normal(0, 0.01, features.shape)

    # Normalize (z-score)
    mean = np.mean(noisy, axis=0)
    std = np.std(noisy, axis=0)
    std[std == 0] = 1e-6
    normalized = (noisy - mean) / std

    # Binarize using 50th percentile threshold
    thresholds = np.percentile(normalized, 50, axis=0)
    binary = (normalized > thresholds).astype(int)

    return binary







def hash_binary_key(binary_vector):
    bitstring = ''.join(map(str, binary_vector.flatten()))
    return hashlib.sha256(bitstring.encode()).hexdigest()

def generate_key(user_id, task):
    file_path = os.path.join(TEMPLATE_DIR, f"{user_id}_{task}.json")
    if not os.path.exists(file_path):
        return None, None, None

    with open(file_path, 'r') as f:
        data = json.load(f)

    features = np.array(data["features"])  # shape: (1, N)

    selected = select_top_k_features(features, k=16)
    binary = binarize_features(selected)
    hashed_key = hash_binary_key(binary)

    return selected, binary, hashed_key


# Secret Vault Encryption ---
def encrypt_message(message, key):
    return hashlib.sha256((message + key).encode()).hexdigest()

def decrypt_message(hash_value, key):
    return f"[Decrypted content cannot be reversed. Hash: {hash_value}]"

def store_secret(key, secret):
    encrypted = encrypt_message(secret, key)
    with open(f"{VAULT_DIR}/{key}_secret.txt", "w") as f:
        f.write(encrypted)
    return True

def retrieve_secret(key):
    try:
        with open(f"{VAULT_DIR}/{key}_secret.txt", "r") as f:
            encrypted = f.read()
        return decrypt_message(encrypted, key)
    except Exception as e:
        return f"❌ Error: {str(e)}"
