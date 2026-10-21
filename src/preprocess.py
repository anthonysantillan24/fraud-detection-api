import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_synthetic_data(output_path: str = "data/raw/creditcard.csv"):
    """Genera un dataset sintético de transacciones si no existe."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Generando dataset sintético de prueba...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Creamos variables numéricas simulando componentes de transacciones
    data = {
        f"V{i}": np.random.randn(n_samples) for i in range(1, 6)
    }
    data["Amount"] = np.random.exponential(scale=50.0, size=n_samples)
    
    # Simulamos la etiqueta 'Class' (0: Legítima, 1: Fraude)
    data["Class"] = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset generado exitosamente en: {output_path}")

def load_data(file_path: str) -> pd.DataFrame:
    """Carga los datos desde un archivo CSV."""
    if not os.path.exists(file_path):
        generate_synthetic_data(file_path)
    return pd.read_csv(file_path)

def preprocess_data(file_path: str = "data/raw/creditcard.csv", target_col: str = "Class"):
    """Limpia los datos, escala las variables y divide en train/test."""
    df = load_data(file_path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar el scaler en la carpeta 'models/' para usarlo después en la API
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("¡Preprocesamiento completado con éxito!")
    print(f"Forma de X_train: {X_train.shape}, Forma de X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()