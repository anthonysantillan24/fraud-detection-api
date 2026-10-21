import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="API para detección de transacciones fraudulentas en tiempo real.",
    version="1.0.0"
)

# Cargar el modelo y el escalador guardados
try:
    model = joblib.load("models/fraud_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
except Exception as e:
    model = None
    scaler = None

# Definir exactamente las 6 características que el modelo espera
class Transaction(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float

@app.get("/")
def home():
    return {"message": "API Detección de Fraudes activa y lista para recibir predicciones."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="El modelo o el escalador no se encuentran cargados.")

    try:
        # Convertir los datos a DataFrame
        input_data = pd.DataFrame([transaction.model_dump()])

        # Pasar los 6 valores como matriz numérica al modelo
        X_array = input_data.values

        prediction = model.predict(X_array)[0]
        probability = model.predict_proba(X_array)[0][1]

        is_fraud = bool(prediction == 1)

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(float(probability), 4),
            "status": "Aprobada" if not is_fraud else "Alerta: Transacción sospechosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")