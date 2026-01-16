import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Preprocessing Logic (Adapted from pre_proc.py)
# ==========================================

def load_data_from_file(file_path):
    """Loads .pkl or .npy files into a Pandas DataFrame [5]."""
    print(f"Loading data from {file_path}...")
    try:
        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        elif file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Please use .pkl or .npy")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_labels(df):
    """Fixes nested list labels (e.g. [['Loc']]) and encodes them [5]."""
    print("Cleaning and encoding labels...")
    
    # Extract string from nested lists
    df['failureType_clean'] = df['failureType'].apply(
        lambda x: x[0][0] if (isinstance(x, (list, np.ndarray)) and len(x)>0 and isinstance(x[0], (list, np.ndarray))) 
        else (x[0] if isinstance(x, (list, np.ndarray)) and len(x)>0 else x)
    )

    # Encode to integers
    le = LabelEncoder()
    df['failureType_encoded'] = le.fit_transform(df['failureType_clean'].astype(str))
    
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {mapping}")
    
    return df, mapping

def preprocess_wafer_map(wafer_map, target_size):
    """Resizes and projects wafer map to RGB channels [5]."""
    # Resize using Nearest Neighbor to preserve discrete values (0, 1, 2)
    resized_map = cv2.resize(wafer_map, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Create empty RGB image
    rgb_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Project channels: Red=Fail(2), Green=Pass(1), Blue=Background(0)
    rgb_image[:, :, 0] = np.where(resized_map == 2, 255, 0)
    rgb_image[:, :, 1] = np.where(resized_map == 1, 255, 0)
    rgb_image[:, :, 2] = np.where(resized_map == 0, 255, 0)
    
    return rgb_image

def process_pipeline(df, target_size):
    """Applies resizing and RGB projection to the dataframe [5]."""
    print(f"Processing wafer maps to shape {target_size} and converting to RGB...")
    
    # List comprehension for speed
    processed_images = [preprocess_wafer_map(x, target_size) for x in df['waferMap']]
    
    X_data = np.array(processed_images)
    y_data = df['failureType_encoded'].values
    
    return X_data, y_data

# ==========================================
# 2. PyTorch Dataset Class (Adapted from train.py)
# ==========================================

class WaferDataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        Args:
            X_data: Numpy array of shape (N, H, W, 3) -> Range [0, 255]
            y_data: Numpy array of labels
        """
        self.X = X_data
        self.y = y_data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Get image and label [6]
        img = self.X[idx] 
        label = self.y[idx]
        
        # Prepare for PyTorch: (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Normalize to [0, 1] and convert to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

# ==========================================
# 3. Main Interface Function
# ==========================================

def get_data_wafer(file_path, batch_size=32, target_size=(224, 224), val_split=0.1, test_split=0.1):
    """
    Main function to load data, process it, and return DataLoaders.
    
    Args:
        file_path (str): Path to .pkl or .npy file.
        batch_size (int): Batch size for DataLoaders.
        target_size (tuple): (Height, Width) for resizing.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        
    Returns:
        train_loader, val_loader, test_loader, label_mapping
    """
    # 1. Load and Clean
    df = load_data_from_file(file_path)
    if df is None:
        return None, None, None, None
        
    df, label_map = clean_labels(df)
    
    # 2. Preprocess Images
    X, y = process_pipeline(df, target_size=target_size)
    
    # 3. Split Data (Train / Temp)
    # Total test+val size
    test_val_size = val_split + test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, random_state=42, stratify=y
    )
    
    # Split Temp into Val / Test
    # Adjust ratio because X_temp is already a fraction of the total
    relative_test_size = test_split / test_val_size
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=42, stratify=y_temp
    )
    
    print(f"Data Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Create Datasets
    train_dataset = WaferDataset(X_train, y_train)
    val_dataset = WaferDataset(X_val, y_val)
    test_dataset = WaferDataset(X_test, y_test)
    
    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, label_map
