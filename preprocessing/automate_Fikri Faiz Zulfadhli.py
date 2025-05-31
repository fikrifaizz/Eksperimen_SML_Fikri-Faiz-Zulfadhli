"""
Automated Preprocessing Pipeline for AI Developer Productivity Dataset
Author: Fikri Faiz Zulfadhli
Sesuai dengan notebook eksperimen manual
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_ai_dev_productivity(input_file='ai_dev_productivity.csv'):
    """
    Fungsi untuk melakukan preprocessing otomatis pada dataset AI Developer Productivity
    sesuai dengan eksperimen manual di notebook

    Args:
        input_file (str): Path ke file dataset input

    Returns:
        pd.DataFrame: Data yang sudah diproses dan siap untuk training
    """

    # 1. Memuat Dataset
    print("Memuat dataset...")
    data = pd.read_csv(input_file)
    print(f"Dataset dimuat dengan shape: {data.shape}")

    # 2. Outlier Removal menggunakan IQR method
    print("Menghapus outliers menggunakan metode IQR...")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    print(f"Shape setelah menghapus outliers: {data.shape}")

    # 3. Standardisasi Fitur
    print("Melakukan standardisasi fitur...")
    features_to_scale = [
        'hours_coding',
        'coffee_intake_mg',
        'commits',
        'cognitive_load',
        'ai_usage_hours',
        'sleep_hours'
    ]

    # Pisahkan fitur dan target
    X = data.drop('task_success', axis=1)
    y = data['task_success']

    # Proses standardisasi
    X_processed = X.copy()
    scaler = StandardScaler()
    X_processed[features_to_scale] = scaler.fit_transform(X_processed[features_to_scale])

    # Gabungkan kembali dengan target
    data_preprocessed = X_processed.copy()
    data_preprocessed['task_success'] = y

    print("Standardisasi selesai")

    # 4. Simpan hasil preprocessing
    output_file = 'ai_dev_productivity_processed.csv'
    data_preprocessed.to_csv(output_file, index=False)
    print(f"Data yang sudah diproses telah disimpan sebagai '{output_file}'")

    # 5. Simpan scaler model
    scaler_file = 'scaler_model.pkl'
    joblib.dump(scaler, scaler_file)
    print(f"StandardScaler telah disimpan sebagai '{scaler_file}'")

    return data_preprocessed


def main():
    """Fungsi utama untuk menjalankan preprocessing otomatis"""
    try:
        # Jalankan preprocessing
        processed_data = preprocess_ai_dev_productivity()

        print("\n" + "=" * 50)
        print("PREPROCESSING SELESAI!")
        print("=" * 50)
        print(f"Shape data final: {processed_data.shape}")
        print("\nFile yang dihasilkan:")
        print("1. ai_dev_productivity_processed.csv")
        print("2. scaler_model.pkl")

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()