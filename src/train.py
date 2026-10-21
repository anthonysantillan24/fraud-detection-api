import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import preprocess_data

def train_model():
    """Entrena un modelo de regresión logística para detección de fraude."""
    print("Iniciando carga y preprocesamiento de datos para el entrenamiento...")
    X_train, X_test, y_train, y_test = preprocess_data()

    print("Entrenando el modelo de Machine Learning...")
    # Usamos class_weight='balanced' porque el fraude suele ser una clase minoritaria
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluamos el modelo
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Modelo entrenado con éxito. ROC-AUC Score: {roc_auc:.4f}")

    # Guardar el modelo entrenado
    os.makedirs("models", exist_ok=True)
    model_path = "models/fraud_model.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    train_model()