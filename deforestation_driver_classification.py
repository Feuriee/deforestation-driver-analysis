"""Klasifikasi sintetis penggerak deforestasi dengan fitur Sentinel-1/2."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "s1_vv_backscatter",
    "s1_vh_backscatter",
    "s2_ndvi",
    "s2_ndwi",
    "patch_size_ha",
    "dist_to_road_km",
]
CLASS_NAMES = [
    "Smallholder Ag",
    "Large Ag",
    "Road Dev",
    "Selective Logging",
    "Mining",
    "Wildfire",
    "Flooding",
]
CLASS_PROBABILITIES = [0.35, 0.25, 0.10, 0.10, 0.05, 0.10, 0.05]


def generate_synthetic_satellite_data(n_samples=1000, random_state=42):
    """Buat data sintetis Sentinel-1/2 dengan distribusi kelas tidak seimbang."""
    if n_samples <= 0:
        raise ValueError("n_samples harus lebih besar dari 0.")

    rng = np.random.default_rng(random_state)
    drivers = rng.choice(
        len(CLASS_NAMES), size=n_samples, p=CLASS_PROBABILITIES
    )
    features = np.empty((n_samples, len(FEATURE_COLUMNS)), dtype=float)

    for index, driver in enumerate(drivers):
        if driver == 0:  # Smallholder agriculture
            features[index] = [
                rng.normal(-12, 1.5),
                rng.normal(-18, 1.5),
                rng.normal(0.3, 0.1),
                rng.normal(-0.2, 0.1),
                rng.uniform(0.5, 3.0),
                rng.uniform(0.1, 5.0),
            ]
        elif driver == 1:  # Large-scale agriculture
            features[index] = [
                rng.normal(-14, 1.0),
                rng.normal(-20, 1.0),
                rng.normal(0.2, 0.08),
                rng.normal(-0.3, 0.1),
                rng.uniform(10.0, 50.0),
                rng.uniform(0.5, 10.0),
            ]
        elif driver == 2:  # Road development
            features[index] = [
                rng.normal(-8, 2.0),
                rng.normal(-14, 2.0),
                rng.normal(0.25, 0.1),
                rng.normal(-0.1, 0.1),
                rng.uniform(1.0, 5.0),
                rng.uniform(0.0, 0.2),
            ]
        elif driver == 3:  # Selective logging
            features[index] = [
                rng.normal(-10, 1.2),
                rng.normal(-15, 1.2),
                rng.normal(0.5, 0.1),
                rng.normal(-0.15, 0.1),
                rng.uniform(0.2, 2.0),
                rng.uniform(0.1, 3.0),
            ]
        elif driver == 4:  # Mining
            features[index] = [
                rng.normal(-5, 2.5),
                rng.normal(-11, 2.5),
                rng.normal(0.1, 0.05),
                rng.normal(0.1, 0.15),
                rng.uniform(2.0, 15.0),
                rng.uniform(1.0, 8.0),
            ]
        elif driver == 5:  # Wildfire
            features[index] = [
                rng.normal(-13, 2.0),
                rng.normal(-19, 2.0),
                rng.normal(0.15, 0.1),
                rng.normal(-0.4, 0.1),
                rng.uniform(5.0, 40.0),
                rng.uniform(2.0, 15.0),
            ]
        else:  # Flooding
            features[index] = [
                rng.normal(-18, 1.5),
                rng.normal(-24, 1.5),
                rng.normal(0.2, 0.1),
                rng.normal(0.5, 0.1),
                rng.uniform(3.0, 20.0),
                rng.uniform(1.0, 10.0),
            ]

    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df["driver_label"] = drivers
    df["driver_name"] = [CLASS_NAMES[driver] for driver in drivers]
    return df, CLASS_NAMES.copy()


def evaluate_model(y_true, y_pred):
    """Hitung metrik evaluasi klasifikasi multikelas."""
    return {
        "Overall Accuracy": accuracy_score(y_true, y_pred),
        "Micro F1-Score": f1_score(y_true, y_pred, average="micro"),
        "Macro F1-Score": f1_score(y_true, y_pred, average="macro"),
    }


def main():
    print("=" * 65)
    print("  SIMULASI KLASIFIKASI PENGGERAK DEFORESTASI")
    print("  Fitur sintetis Sentinel-1 dan Sentinel-2")
    print("=" * 65)

    print("\n[1] Membuat dataset sintetis...")
    df, class_names = generate_synthetic_satellite_data(n_samples=1000)
    df.to_csv("deforestation_data.csv", index=False)
    print("    -> Dataset disimpan sebagai 'deforestation_data.csv'")
    print(f"    -> Total sampel: {len(df)}")
    print("\nDistribusi kelas:")
    print(df["driver_name"].value_counts().reindex(class_names, fill_value=0))

    X = df[FEATURE_COLUMNS]
    y = df["driver_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[2] Melatih Standard Random Forest Classifier...")
    rf_standard = RandomForestClassifier(random_state=42)
    rf_standard.fit(X_train, y_train)
    y_pred_standard = rf_standard.predict(X_test)

    print("\n[3] Melatih Balanced Random Forest Classifier...")
    rf_balanced = RandomForestClassifier(
        class_weight="balanced", random_state=42
    )
    rf_balanced.fit(X_train, y_train)
    y_pred_balanced = rf_balanced.predict(X_test)

    standard_metrics = evaluate_model(y_test, y_pred_standard)
    balanced_metrics = evaluate_model(y_test, y_pred_balanced)
    metrics_df = pd.DataFrame(
        {"Standard RF": standard_metrics, "Balanced RF": balanced_metrics}
    )

    print("\n" + "=" * 65)
    print("                  HASIL EVALUASI MODEL")
    print("=" * 65)
    print(metrics_df.round(4))

    print("\n--- Laporan Klasifikasi Rinci (Balanced RF) ---")
    print(
        classification_report(
            y_test,
            y_pred_balanced,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("\n[4] Membuat confusion matrix...")
    cm = confusion_matrix(
        y_test, y_pred_balanced, labels=range(len(class_names))
    )
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Balanced RF")
    plt.xlabel("Prediksi Model")
    plt.ylabel("Kondisi Aktual (Ground Truth)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("    -> Plot disimpan sebagai 'confusion_matrix.png'")
    print("\nProses selesai.")


if __name__ == "__main__":
    main()
