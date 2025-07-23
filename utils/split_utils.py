# split_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_labels(csv_path, test_size=0.15, val_size=0.15, seed=42):
    df = pd.read_csv(csv_path)
    df["image_num"] = df["image_num"].astype(int)  # 👈 ADĂUGĂ LINIA ASTA
    # Calculează scorul mediu
    vote_columns = [f"vote_{i}" for i in range(1, 11)]
    df["mean_score"] = sum((i * df[f"vote_{i}"]) for i in range(1, 11))

    # Bin scorurile pentru stratificare
    df["score_bin"] = pd.qcut(df["mean_score"], q=5, labels=False)

    # Train/Test/Val split
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["score_bin"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val["score_bin"], random_state=seed)

    # Elimină coloana auxiliară
    for d in [train, val, test]:
        d.drop(columns=["score_bin"], inplace=True)

    return train, val, test
