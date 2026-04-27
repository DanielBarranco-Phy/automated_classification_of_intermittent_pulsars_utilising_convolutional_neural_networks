# Step 1 - Data Exploration

import warnings
warnings.filterwarnings("ignore") 

import numpy as np

# ==========================================
# AGGRESSIVE NUMPY FIX FOR OLD LIBRARIES
# ==========================================
if not hasattr(np, 'str'): np.str = str 
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

import matplotlib.pyplot as plt
from pypulse.archive import Archive
import os
from scipy import signal

# ==========================================
# 1. SETUP PATHS & NEW TEXT FILES
# ==========================================
PATHS_1910 = [
    "/data/physics_group/pulsar/users/tleeming/PSRJ1910+0517/jodrell2/PSRCHIVE_scr/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1910+0517/jodrell2/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1910+0517/" 
]

PATHS_1929 = [
    "/data/physics_group/pulsar/users/tleeming/PSRJ1929+1357/jodrell2/PSRCHIVE_scr/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1929+1357/jodrell2/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1929+1357/"
]

# Using Tibby's finalized, deduplicated text files
lbls_1910 = ['J1910_ID.txt', 'ID_J1910_new.txt'] 
lbls_1929 = ['ID_J1929_updated.txt', 'ID_J1929_new.txt'] 

TARGET_BINS = 512

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_labels(file_list):
    """Parses text files to get a dictionary of exactly which files are 1 or 0."""
    labels = {}
    for filepath in file_list:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].split('.')[0] # e.g. J111129_192723
                        labels[key] = int(parts[-1])
    return labels

def find_file(filename, search_dirs):
    """Hunts down the physical file across the directories."""
    base = filename.split('.')[0]
    for d in search_dirs:
        for ext in ['.FTp', '.ft', '.rf', '.ar']:
            path = os.path.join(d, base + ext)
            if os.path.exists(path): return path
    return None

def get_n_examples_with_proof(label_dict, search_dirs, target_label, n=3):
    """Finds physical paths and keeps the original filename and label for proof."""
    found_data = []
    for filename, label in label_dict.items():
        if label == target_label:
            path = find_file(filename, search_dirs)
            if path:
                # Store a tuple containing: (File Path, File Name, Label Value)
                found_data.append((path, filename, label))
                if len(found_data) >= n: break
    return found_data

def process_file_1d(filepath):
    """Loads and returns the standardized 1D profile."""
    try:
        ar = Archive(filepath)
        if ar.getNpol() > 1: ar.pscrunch()
        if ar.getNchan() > 1: ar.fscrunch()
        if ar.getNsubint() > 1: ar.tscrunch()
        
        data = ar.getData(squeeze=True)
        
        # Resample to 512
        if len(data) != TARGET_BINS:
            data = signal.resample(data, TARGET_BINS)
            
        # Normalize (0 to 1) for visual comparison
        dmin, dmax = np.min(data), np.max(data)
        if dmax - dmin != 0:
            data = (data - dmin) / (dmax - dmin)
        else:
            data = np.zeros(TARGET_BINS)
        return data
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return np.zeros(TARGET_BINS)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

print("Scanning for J1910 and J1929 examples...")
dict_1910 = get_labels(lbls_1910)
dict_1929 = get_labels(lbls_1929)

# Get 3 exact paths for each category INCLUDING the label proof
ex_1910_on  = get_n_examples_with_proof(dict_1910, PATHS_1910, 1, 3)
ex_1910_off = get_n_examples_with_proof(dict_1910, PATHS_1910, 0, 3)
ex_1929_on  = get_n_examples_with_proof(dict_1929, PATHS_1929, 1, 3)
ex_1929_off = get_n_examples_with_proof(dict_1929, PATHS_1929, 0, 3)

# --- VISUAL GRID (4x3) ---
fig, axes = plt.subplots(4, 3, figsize=(14, 11)) # Slightly wider for the text
fig.suptitle("Step 1 Visual Verification (with Ground Truth Check)", fontsize=16)

rows = [ex_1910_on, ex_1910_off, ex_1929_on, ex_1929_off]
row_names = ["J1910 ON\n(Train)", "J1910 OFF\n(Train)", "J1929 ON\n(Test)", "J1929 OFF\n(Test)"]

for row_idx, (file_data, name) in enumerate(zip(rows, row_names)):
    # Label the row
    axes[row_idx, 0].set_ylabel(name, fontsize=12, fontweight='bold')
    
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        if col_idx < len(file_data):
            # Unpack our proof data
            filepath, filename, label_val = file_data[col_idx]
            
            data = process_file_1d(filepath)
            ax.plot(data, color='black', lw=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Title styling with the PROOF included
            if label_val == 1:
                ax.set_title(f"Pulse | {filename} | Lbl: {label_val}", fontsize=10, color='green', fontweight='bold')
            else:
                ax.set_title(f"Noise | {filename} | Lbl: {label_val}", fontsize=10, color='gray')
        else:
            ax.axis('off') # Hide empty plots if <3 files found

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Step 2 - Supervised Dataset Construction
import warnings
warnings.filterwarnings("ignore") 

import numpy as np

# ==========================================
# AGGRESSIVE NUMPY FIX FOR OLD LIBRARIES
# ==========================================
if not hasattr(np, 'str'): np.str = str 
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

from pypulse.archive import Archive
from scipy import signal
import os

# ==========================================
# 1. SETUP PATHS & COMBINED LABEL FILES
# ==========================================
PATHS_1910 = [
    "/data/physics_group/pulsar/users/tleeming/PSRJ1910+0517/jodrell2/PSRCHIVE_scr/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1910+0517/jodrell2/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1910+0517/" 
]

PATHS_1929 = [
    "/data/physics_group/pulsar/users/tleeming/PSRJ1929+1357/jodrell2/PSRCHIVE_scr/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1929+1357/jodrell2/",
    "/data/physics_group/pulsar/users/tleeming/PSRJ1929+1357/"
]

# Combined Master Lists
lbls_1910 = ['J1910_ID.txt', 'ID_J1910_new.txt'] 
lbls_1929 = ['updated_ID_J1929.txt', 'ID_J1929_updated.txt', 'ID_J1929_new.txt'] 

TARGET_BINS = 512

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_labels(file_list):
    """Parses text files and returns a unified dictionary of all labels."""
    labels = {}
    for filepath in file_list:
        if os.path.exists(filepath):
            print(f"📖 Loaded labels from: {filepath}")
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].split('.')[0]
                        labels[key] = int(parts[-1])
    return labels

def find_file(filename, search_dirs):
    """Hunts down the physical file across the directories."""
    base = filename.split('.')[0]
    for d in search_dirs:
        for ext in ['.FTp', '.ft', '.rf', '.ar']:
            path = os.path.join(d, base + ext)
            if os.path.exists(path): return path
    return None

def create_master_dataset(label_files, search_dirs, save_name):
    """
    Finds the files, processes them to 1D, normalizes to [0,1], and saves as .npy.
    """
    # 1. Load Combined Labels
    label_dict = get_labels(label_files)
    if not label_dict:
        print(f"❌ No labels found for {save_name}.")
        return

    print(f"\n--- Processing {save_name} ({len(label_dict)} combined IDs detected) ---")
    
    data_list = []
    labels_list = []
    matched = 0
    
    for idx, (filename, label) in enumerate(label_dict.items()):
        path = find_file(filename, search_dirs)
        
        if path:
            try:
                # --- A. LOAD ARCHIVE ---
                ar = Archive(path)
                if ar.getNpol() > 1: ar.pscrunch()
                if ar.getNchan() > 1: ar.fscrunch()
                if ar.getNsubint() > 1: ar.tscrunch()
                
                profile = ar.getData(squeeze=True)
                
                # --- B. STANDARDIZE (512 Bins) ---
                if len(profile) != TARGET_BINS:
                    profile = signal.resample(profile, TARGET_BINS)
                
                # --- C. NORMALIZE (0 to 1) ---
                p_min, p_max = np.min(profile), np.max(profile)
                if p_max - p_min != 0:
                    profile = (profile - p_min) / (p_max - p_min)
                else:
                    profile = np.zeros(TARGET_BINS)
                
                data_list.append(profile)
                labels_list.append(label)
                matched += 1
                
                # Update console to show progress without spamming
                if matched % 100 == 0:
                    print(f"   Processed {matched} files...", end='\r')
                    
            except Exception:
                pass # Skip corrupted files
                
    # --- SAVE TENSORS ---
    X = np.array(data_list)
    y = np.array(labels_list)
    
    print(f"\n✅ Finished {save_name}")
    print(f"   Final Tensor Shape: {X.shape}")
    print(f"   Class Distribution: {np.sum(y==1)} ON (Pulse) | {np.sum(y==0)} OFF (Noise)")
    
    np.save(f'{save_name}_data_1D.npy', X)
    np.save(f'{save_name}_labels_1D.npy', y)
    print(f"💾 Saved {save_name}_data_1D.npy & {save_name}_labels_1D.npy\n")

# ==========================================
# 3. EXECUTE PIPELINE
# ==========================================
print("🚀 Constructing Master Tensors for AI Training...")

# Generate Training Set (J1910)
create_master_dataset(lbls_1910, PATHS_1910, 'J1910_Master')

# Generate Testing Set (J1929)
create_master_dataset(lbls_1929, PATHS_1929, 'J1929_Master')

#Step 3 - Model Architecture and Training

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ==========================================
# 1. LOAD DATA
# ==========================================
print("LOADING DATA...")
# We load the newly created MASTER datasets
X = np.load('J1910_Master_data_1D.npy')
y = np.load('J1910_Master_labels_1D.npy')

# Reshape for CNN: (Samples, 512) -> (Samples, 512, 1)
# CNNs expect a 3D input: Batch x Steps x Channels
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"Data Loaded. Shape: {X.shape}")

# ==========================================
# 2. SPLIT TRAINING / VALIDATION
# ==========================================
# We keep 20% of J1910 aside to check progress during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)} ({np.sum(y_train==1)} Pulses)")
print(f"Validation samples: {len(X_val)} ({np.sum(y_val==1)} Pulses)")

# ==========================================
# 3. COMPUTE CLASS WEIGHTS
# ==========================================
# Since we have fewer Pulses (1) than Noise (0), we weight Pulses higher.
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print(f"\nClass Weights: {class_weights}") 
print("(This tells the AI to pay much more attention to Pulses during training)")

# ==========================================
# 4. BUILD THE 1D CNN MODEL
# ==========================================
model = Sequential([
    # Input Layer
    Input(shape=(512, 1)),
    
    # Feature Extraction (The "Eye")
    Conv1D(filters=16, kernel_size=16, activation='relu'),
    MaxPooling1D(pool_size=4),
    
    Conv1D(filters=32, kernel_size=8, activation='relu'),
    MaxPooling1D(pool_size=4),
    
    # Classification (The "Brain")
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), # Prevents memorization of the 295 files
    Dense(1, activation='sigmoid') # Binary Output (0 or 1)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

print("\n=== MODEL SUMMARY ===")
model.summary()

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================
print("\n🚀 STARTING TRAINING...")
history = model.fit(
    X_train, y_train,
    epochs=50,               # How many times to loop through data
    batch_size=16,           # Small batch size for small data
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    verbose=1
)

# ==========================================
# 6. PLOT TRAINING HISTORY
# ==========================================
plt.figure(figsize=(12, 4))

# Plot Loss (We want this to go DOWN)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss (Lower is Better)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

# Plot Recall (We want this to go UP - Finding Pulses)
plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Train Recall', color='green')
plt.plot(history.history['val_recall'], label='Val Recall', color='red')
plt.title('Recall / Sensitivity (Higher is Better)')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

#Step 4 - Generalization Test

# ==========================================
# STEP 4: FINAL TESTING ON J1929 MASTER
# ==========================================
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Test Data (J1929 Master)
print("Loading Test Data (J1929 Master)...")
X_test = np.load('J1929_Master_data_1D.npy')
y_test = np.load('J1929_Master_labels_1D.npy')

# Reshape for CNN
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 2. Ask the AI to Predict
print("Predicting...")
predictions_prob = model.predict(X_test)

# Convert probabilities (0.1, 0.9) to labels (0, 1) using 50% threshold
predictions = (predictions_prob > 0.5).astype(int)

# 3. The Scorecard (Comparison with Tibby's List)
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, predictions)

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Noise', 'Predicted Pulse'],
            yticklabels=['Actual Noise', 'Actual Pulse'])
plt.title('J1929 Master Test Results')
plt.xlabel('AI Prediction')
plt.ylabel('Ground Truth (Tibby)')
plt.show()

# 4. Detailed Metrics
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, predictions, target_names=['Noise', 'Pulse']))

# 5. Calculate Specific Metrics for the Report
tn, fp, fn, tp = cm.ravel()
print(f"Correctly Found Pulses (True Positives): {tp}")
print(f"Missed Pulses (False Negatives):       {fn}")
print(f"False Alarms (False Positives):        {fp}")
print(f"Correctly Ignored Noise (True Negatives): {tn}")

#Step 5 - Scenario 2 (Reversed)

import warnings
warnings.filterwarnings("ignore") 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# AGGRESSIVE NUMPY FIX FOR OLD LIBRARIES
# ==========================================
if not hasattr(np, 'str'): np.str = str 
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

# ==========================================
# 1. LOAD DATA (TRAINING ON J1929)
# ==========================================
print("LOADING TRAINING DATA (J1929 MASTER)...")
X_train_full = np.load('J1929_Master_data_1D.npy')
y_train_full = np.load('J1929_Master_labels_1D.npy')

# Reshape for CNN: (Samples, 512, 1)
X_train_full = X_train_full.reshape((X_train_full.shape[0], X_train_full.shape[1], 1))
print(f"J1929 Data Loaded. Shape: {X_train_full.shape}")

# ==========================================
# 2. SPLIT TRAINING / VALIDATION
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Training samples: {len(X_train)} ({np.sum(y_train==1)} Pulses)")
print(f"Validation samples: {len(X_val)} ({np.sum(y_val==1)} Pulses)")

# ==========================================
# 3. COMPUTE DYNAMIC CLASS WEIGHTS
# ==========================================
print("\nCOMPUTING NEW PENALIZATION WEIGHTS...")
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print(f"Calculated Class Weights for J1929: {class_weights}") 

# ==========================================
# 4. BUILD THE CONVOLUTIONAL NEURAL NETWORK
# ==========================================
model = Sequential([
    Input(shape=(512, 1)),
    
    Conv1D(filters=16, kernel_size=16, activation='relu'),
    MaxPooling1D(pool_size=4),
    
    Conv1D(filters=32, kernel_size=8, activation='relu'),
    MaxPooling1D(pool_size=4),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================
print("\nSTARTING TRAINING PHASE...")
history = model.fit(
    X_train, y_train,
    epochs=50,               
    batch_size=16,           
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    verbose=1
)

# ==========================================
# 6. PLOT TRAINING HISTORY
# ==========================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss (Lower is Better)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Train Recall', color='green')
plt.plot(history.history['val_recall'], label='Val Recall', color='red')
plt.title('Recall / Sensitivity (Higher is Better)')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 7. FINAL TESTING ON J1910 MASTER
# ==========================================
print("\nLOADING TEST DATA (J1910 MASTER)...")
X_test = np.load('J1910_Master_data_1D.npy')
y_test = np.load('J1910_Master_labels_1D.npy')

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("PREDICTING ON UNSEEN J1910 ARCHIVE...")
predictions_prob = model.predict(X_test)
predictions = (predictions_prob > 0.5).astype(int)

# ==========================================
# 8. METRICS AND CONFUSION MATRIX
# ==========================================
print("\nCONFUSION MATRIX")
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Noise', 'Predicted Pulse'],
            yticklabels=['Actual Noise', 'Actual Pulse'])
plt.title('J1910 Master Test Results (Reverse Generalization)')
plt.xlabel('AI Prediction')
plt.ylabel('Ground Truth')
plt.show()

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions, target_names=['Noise', 'Pulse']))

tn, fp, fn, tp = cm.ravel()
print(f"Correctly Found Pulses (True Positives): {tp}")
print(f"Missed Pulses (False Negatives):       {fn}")
print(f"False Alarms (False Positives):        {fp}")
print(f"Correctly Ignored Noise (True Negatives): {tn}")

#Step 6 - Scenario 3 (Unified)

import warnings
warnings.filterwarnings("ignore") 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# AGGRESSIVE NUMPY FIX FOR OLD LIBRARIES
# ==========================================
if not hasattr(np, 'str'): np.str = str 
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

# ==========================================
# 1. LOAD AND MERGE DATA (J1910 + J1929)
# ==========================================
print("LOADING AND MERGING DATASETS...")
X_1910 = np.load('J1910_Master_data_1D.npy')
y_1910 = np.load('J1910_Master_labels_1D.npy')

X_1929 = np.load('J1929_Master_data_1D.npy')
y_1929 = np.load('J1929_Master_labels_1D.npy')

# Concatenate arrays along the first axis (samples)
X_unified = np.concatenate((X_1910, X_1929), axis=0)
y_unified = np.concatenate((y_1910, y_1929), axis=0)

# Reshape for CNN: (Samples, 512, 1)
X_unified = X_unified.reshape((X_unified.shape[0], X_unified.shape[1], 1))

print(f"Unified Data Loaded. Total Shape: {X_unified.shape}")
print(f"Total Pulses: {np.sum(y_unified==1)} | Total Noise: {np.sum(y_unified==0)}")

# ==========================================
# 2. SPLIT TRAINING / VALIDATION (80/20)
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(
    X_unified, y_unified, test_size=0.2, random_state=42, stratify=y_unified
)

print(f"\nTraining samples (80%): {len(X_train)} ({np.sum(y_train==1)} Pulses)")
print(f"Testing samples (20%): {len(X_val)} ({np.sum(y_val==1)} Pulses)")

# ==========================================
# 3. COMPUTE DYNAMIC CLASS WEIGHTS
# ==========================================
print("\nCOMPUTING UNIFIED PENALIZATION WEIGHTS...")
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print(f"Calculated Class Weights for Unified Set: {class_weights}") 

# ==========================================
# 4. BUILD THE CONVOLUTIONAL NEURAL NETWORK
# ==========================================
model = Sequential([
    Input(shape=(512, 1)),
    
    Conv1D(filters=16, kernel_size=16, activation='relu'),
    MaxPooling1D(pool_size=4),
    
    Conv1D(filters=32, kernel_size=8, activation='relu'),
    MaxPooling1D(pool_size=4),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================
print("\nSTARTING TRAINING PHASE...")
history = model.fit(
    X_train, y_train,
    epochs=50,               
    batch_size=16,           
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    verbose=1
)

# ==========================================
# 6. PLOT TRAINING HISTORY
# ==========================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss (Lower is Better)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Train Recall', color='green')
plt.plot(history.history['val_recall'], label='Val Recall', color='red')
plt.title('Recall / Sensitivity (Higher is Better)')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 7. FINAL TESTING ON THE 20% HOLDOUT
# ==========================================
print("\nPREDICTING ON 20% UNSEEN HOLDOUT SET...")
predictions_prob = model.predict(X_val)
predictions = (predictions_prob > 0.5).astype(int)

# ==========================================
# 8. METRICS AND CONFUSION MATRIX
# ==========================================
print("\nCONFUSION MATRIX")
cm = confusion_matrix(y_val, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Noise', 'Predicted Pulse'],
            yticklabels=['Actual Noise', 'Actual Pulse'])
plt.title('Unified Dataset Test Results (20% Holdout)')
plt.xlabel('AI Prediction')
plt.ylabel('Ground Truth')
plt.show()

print("\nCLASSIFICATION REPORT")
print(classification_report(y_val, predictions, target_names=['Noise', 'Pulse']))

tn, fp, fn, tp = cm.ravel()
print(f"Correctly Found Pulses (True Positives): {tp}")
print(f"Missed Pulses (False Negatives):       {fn}")
print(f"False Alarms (False Positives):        {fp}")
print(f"Correctly Ignored Noise (True Negatives): {tn}")
