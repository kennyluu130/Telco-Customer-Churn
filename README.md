# Telco Customer Churn Prediction

[![Docker Support](https://img.shields.io/badge/Docker-Supported-blue.svg?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat&logo=render)](https://render.com/)

An end-to-end Machine Learning solution to identify at-risk customers in the telecommunications sector. This project features a high-performance **XGBoost** model served via a **FastAPI** backend and an interactive **Gradio** web interface.

---

## Live Demo

**[Click here to open the Churn Predictor UI](https://telco-customer-churn-2fqg.onrender.com/ui/)** _(Note: Please allow 30-50 seconds for the free-tier server to wake up if it has been inactive.)_

---

## Tech Stack

- **Modeling:** Python, XGBoost, Scikit-learn, Pandas
- **API Framework:** FastAPI (Uvicorn)
- **Interactive UI:** Gradio
- **Containerization:** Docker
- **Deployment:** Render

---

## Project Overview

Predicting customer churn is critical for subscription-based businesses. This project automates the churn prediction process by analyzing 19+ customer features including contract type, monthly charges, and service usage.

### Key Features

- **Real-time Prediction:** Enter customer data and get instant churn probability.
- **Production-Ready API:** FastAPI endpoint (`/predict`) available for integration with other apps.
- **Dockerized Environment:** Guaranteed consistency from development to production.

---

## Running Locally with Docker

If you have Docker installed, you can run the entire app with these commands:

1. Build the image:

```bash
docker build -t telco-churn
```

2. Run the container:

```bash
docker build -t telco-churn
```

3. Access the app:

- Web UI: http://localhost:8000/ui
- API Docs: http://localhost:8000/docs
