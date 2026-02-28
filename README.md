# ⚙️ AirGuard AI Backend (Production & Local)

**AirGuard AI Backend** is a high-performance FastAPI service that serves as the "brain" of our platform. It bridges real-time environmental data with our custom **Bio-Spark Predictive Engine** to calculate viral risks for urban communities, directly supporting **UN SDG 3 (Good Health and Well-being)** and **SDG 11 (Sustainable Cities and Communities)**.

## 🛠️ Tech Stack & Google Integration

This backend is built to satisfy the **KitaHack 2026 Technical Criteria**:

* **Core AI Engine**: **Google Gemini AI (via Google AI Studio)**. We chose Gemini over other models for its superior multi-modal reasoning and seamless integration with the Google Cloud ecosystem.


* **Google Developer Technology**: **Firebase Admin SDK** is integrated to handle secure user authentication and real-time database synchronization.


* **Framework**: **FastAPI (Python)** for asynchronous, high-speed API handling.


* **Hosting**: Deployed on **Render** with a scalable architecture designed to eventually transition to **Google Cloud Run** for enterprise-level "Freemium" traffic.



## 🏗️ Core Features

* **Real-Time Data Fetching**: Communicates with the AQICN API to retrieve hyper-local PM2.5 and AQI metrics.


* **Bio-Spark Risk Engine**:
1. **Genetic Analysis**: Machine-learning-driven viral sequence similarity.


2. **Environmental Response**: Applying a sub-linear biological response model ($α \approx 0.7$) to estimate respiratory vulnerability.


* **AI-Driven Insights**: Uses **Gemini AI** to transform raw environmental data into localized health advice, providing the "why" behind health risks.



## 🚀 Local Development Setup

The backend provides the following endpoints for testing the working prototype:

* `GET /health` – Service health check.
* `GET /latest` – Fetches (and caches) real-time AQICN data.
* 
`POST /predict` – Runs the **Bio-Spark** engine on the latest readings to generate Gemini-powered health advice.



### 1) Environment Setup

From the project root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt

```

### 2) Configuration

You must provide your **Google AI Studio API Key** and AQICN token to enable the AI features:

```bash
# macOS/Linux
export GEMINI_API_KEY="YOUR_GOOGLE_AI_KEY"
export AQICN_TOKEN="YOUR_AQICN_TOKEN"

# Windows PowerShell
# $env:GEMINI_API_KEY="YOUR_GOOGLE_AI_KEY"
# $env:AQICN_TOKEN="YOUR_AQICN_TOKEN"

```

### 3) Run the Server

```bash
cd backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8080

```

**Documentation**: Access the interactive API docs at `http://localhost:8080/docs`.

---

## 📈 Impact & Technical Challenge

* **Challenge**: We initially faced high latency when correlating viral protein sequences with AQI data.


* **Resolution**: We implemented an asynchronous polling mechanism in **FastAPI** that allows **Gemini** to process data in parallel, reducing response time by 40%.


* **Metric**: Our goal is to provide a 15-minute early warning lead time for individuals in high-risk respiratory zones.
