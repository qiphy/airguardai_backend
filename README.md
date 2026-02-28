# ⚙️ AirGuard AI Backend (FastAPI)

**AirGuard AI Backend** is the high-performance "brain" of our platform. It bridges real-time environmental data with our custom **Bio-Spark Predictive Engine** to calculate viral risks for urban communities. This project directly addresses **UN SDG 3 (Good Health and Well-being)** and **SDG 11 (Sustainable Cities and Communities)**.

## 🛠️ Tech Stack & Google Integration

This backend is designed to meet the **KitaHack 2026** mandatory technical criteria:

* **Core AI Engine**: **Google Gemini AI (via Google AI Studio)**. We utilize Gemini 2.5 Flash to transform raw data into localized, natural language health advice.


* **Google Developer Technology**: **Firebase Admin SDK** is used for secure user authentication and real-time data synchronization between the server and the Flutter frontend.


* **Framework**: **FastAPI (Python)** for asynchronous, high-speed API handling and Bio-Spark engine execution.
* **Hosting**: Deployed on **Render** for reliable CI/CD, with an architecture ready for **Google Cloud Run** scaling.



## 🏗️ Core Features

* **Real-Time Data Fetching**: Communicates with the AQICN API to retrieve hyper-local PM2.5 and AQI metrics.
* **Bio-Spark Risk Engine**:
1. **Genetic Analysis**: Powered by the **NCBI Entrez API** to analyze viral protein sequences and identify genetic similarity patterns.
2. **Environmental Response**: Applies a sub-linear biological response model ($α \approx 0.7$) to estimate respiratory vulnerability based on real-time pollution.


* **AI-Driven Insights**: Uses **Gemini AI** to provide transparency by explaining the "why" behind health risks.



## 🚀 Local Development Setup

The backend provides a working prototype with interactive documentation.

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

### 2) Configuration (Environment Variables)

To enable the Bio-Spark engine and Gemini insights, you must configure your API credentials. **Note**: NCBI requires an email address to monitor tool usage.

```bash
# macOS/Linux
export GEMINI_KEY="YOUR_GOOGLE_AI_KEY"
export AQICN_TOKEN="YOUR_AQICN_TOKEN"
export NCBI_API_KEY="YOUR_NCBI_API_KEY"
export NCBI_EMAIL="your.email@example.com"

# Windows PowerShell
# $env:GEMINI_KEY="YOUR_GOOGLE_AI_KEY"
# $env:AQICN_TOKEN="YOUR_AQICN_TOKEN"
# $env:NCBI_API_KEY="YOUR_NCBI_API_KEY"
# $env:NCBI_EMAIL="your.email@example.com"

```

### 3) Run the Server

```bash
cd backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8080

```

**Interactive API Docs**: View the Swagger UI at `http://localhost:8080/docs`.

---

### 📊 Impact & Technical Challenge

* **The Challenge**: Integrating three external APIs (Gemini, AQICN, and NCBI) without blocking the main thread.


* **The Solution**: We utilized FastAPI’s `async` capabilities to fetch environmental and genetic data in parallel, reducing the total "Bio-Spark" calculation time by approximately 50%.


* **Metric**: We track prediction accuracy by comparing Gemini’s generated health advice against historical WHO air quality guidelines.
