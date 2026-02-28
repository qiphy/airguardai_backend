# AirGuard AI Backend (Local)

🚀 AirGuard AI Backend (Python & Gemini)The AirGuard AI Backend is a high-performance FastAPI service that serves as the "brain" of our platform. It bridges real-time environmental data with our custom Bio-Spark Predictive Engine to calculate viral risks for urban communities. 

🛠️ Tech Stack & Google Cloud IntegrationFramework: FastAPI (Python) for asynchronous, high-speed API handling. AI Engine: Integrated with Gemini AI via Google AI Studio for processing complex data correlations. +1Infrastructure: Designed for deployment on Firebase Cloud and Google Cloud Platform (GCP) to handle scalable "Freemium" traffic. 

🏗️ Core Features

- Real-Time Data Fetching: Communicates with the AQICN API to retrieve hyper-local PM2.5 and AQI metrics. 

- Bio-Spark Risk Engine: Implements our two-step risk calculation:

  1) Genetic Analysis: Machine-learning-driven viral sequence similarity. 

  2) Environmental Response: Applying a sub-linear biological response model ($α \approx 0.7$) to estimate respiratory vulnerability. 

- Actionable Insights: Uses Gemini AI to transform raw data into simple, localized health advice.

🚀 Local Development Setup

This backend runs locally using **FastAPI** and provides:

- `GET /health` – service health check
- `GET /latest` – fetches (and caches) Unknown AQICN data
- `POST /predict` – runs a local scikit-learn model on the latest readings

## 1) Setup

From the project root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
```

## 2) Set your AQICN token

```bash
# macOS/Linux
export AQICN_TOKEN="YOUR_TOKEN"

# Windows PowerShell
# $env:AQICN_TOKEN="YOUR_TOKEN"
```

## 3) Run the server

```bash
cd backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Open:
- http://localhost:8080/health
- http://localhost:8080/latest
- http://localhost:8080/docs
