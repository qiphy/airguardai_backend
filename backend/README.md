# AirGuard AI Backend (Local)

This backend runs locally using **FastAPI** and provides:

- `GET /health` – service health check
- `GET /latest` – fetches (and caches) Kuala Lumpur AQICN data
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
