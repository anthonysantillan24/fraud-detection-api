#  Real-Time Fraud Detection MLOps API

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

Microservicio de detección de transacciones fraudulentas en tiempo real construido con **FastAPI** y **Scikit-Learn**, siguiendo las mejores prácticas de **MLOps** e ingeniería de software.

---

##  Demo y Capturas

### 1. Documentación e Interfaz Interactiva (Swagger UI)
Evaluación en tiempo real enviando un vector de características y recibiendo la probabilidad de fraude junto con la respuesta `200 OK`.

![Swagger UI Demo](assets/api_response.png)

### 2. Estructura del Proyecto
Organización modular de código separando la capa de servicio (`api/`), artefactos de modelos (`models/`), datos (`data/`) y lógica de procesamiento/entrenamiento (`src/`).

![Project Structure](assets/project_structure.png)

### 3. Entorno de Ejecución y Servidor Uvicorn
Servidor ASGI corriendo localmente y procesando peticiones HTTP POST en tiempo real.

![VS Code Terminal](assets/vscode_workspace.png)

---

##  Arquitectura del Proyecto

```text
fraud-detection-api/
│
├── api/
│   ├── __init__.py
│   └── main.py              # Endpoint principal de FastAPI (/predict)
├── data/
│   ├── raw/                 # Datos en bruto
│   └── processed/           # Datos procesados
├── models/
│   ├── fraud_model.joblib   # Modelo de Clasificación
│   └── scaler.joblib        # Escalador StandardScaler
├── src/
│   ├── preprocess.py        # Limpieza y escalamiento
│   ├── train.py             # Entrenamiento y evaluación
│   └── predict.py           # Inferencia
├── Dockerfile               # Contenerización para despliegue
├── requirements.txt         # Dependencias del proyecto
└── README.md
- Instalación y Ejecución Local
1. Clonar el repositorio
Bash
git clone [https://github.com/anthonysantillan24/fraud-detection-api.git](https://github.com/anthonysantillan24/fraud-detection-api.git)
cd fraud-detection-api
2. Crear y activar el entorno virtual
Bash
python -m venv venv
# En Windows (PowerShell):
.\venv\Scripts\Activate
3. Instalar dependencias
Bash
pip install -r requirements.txt
4. Iniciar la API con Uvicorn
Bash
uvicorn api.main:app --reload
Accede a la documentación interactiva en tu navegador:
- http://127.0.0.1:8000/docs

Ejemplo de Petición HTTP POST (/predict)
Body (JSON):

JSON
{
  "feature_1": 0.0,
  "feature_2": 0.0,
  "feature_3": 0.0,
  "feature_4": 0.0,
  "feature_5": 0.0,
  "feature_6": 0.0
}
Respuesta (JSON):

JSON
{
  "is_fraud": false,
  "fraud_probability": 0.4892,
  "status": "Aprobada"
}
