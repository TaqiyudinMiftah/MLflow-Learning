# Machine Learning Deployment & Versioning Control

This repository documents my **learning journey and hands-on experiments** in building **reproducible, deployable, and version-controlled machine learning systems**.

The main focus of this repo is **understanding the end-to-end ML lifecycle**, from training models to managing versions and preparing them for production deployment.

---

## 🚀 Learning Objectives

Through this repository, I aim to:

* Understand **machine learning deployment workflows**
* Practice **experiment tracking and model versioning**
* Learn how to manage the **ML lifecycle using MLflow**
* Apply **best practices for reproducibility and MLOps-ready projects**
* Explore modern tooling such as **uv** for dependency and environment management

---

## 🧠 Topics Covered

* Machine Learning experiment tracking
* Model versioning and lifecycle management
* MLflow Tracking, Models, and Model Registry
* Input validation and inference workflows
* Reproducible ML environments using `uv`
* Preparing models for local and production serving

---

## 🛠️ Tools & Technologies

* **Python**
* **MLflow** – experiment tracking, model packaging, and registry
* **scikit-learn** – model training and evaluation
* **uv** – fast Python package and environment manager
* **Git & GitHub** – version control and progress documentation

---

## 📁 Repository Structure

```text
.
├── main.py              # Model training & MLflow logging
├── validate.py          # Model serving input validation
├── inference.py         # Model inference examples
├── register_model.py    # Model Registry operations
├── pyproject.toml       # Project dependencies and metadata
├── uv.lock              # Locked dependencies for reproducibility
└── README.md
```

---

## 🔄 Workflow Overview

1. **Train a model** and log parameters, metrics, and artifacts using MLflow
2. **Track experiments** to compare model performance
3. **Package models** in MLflow’s standardized format
4. **Validate serving inputs** before deployment
5. **Register models** to manage versions and lifecycle stages
6. **Run inference** using registered models without relying on run IDs

This workflow mirrors **real-world MLOps practices** used in research and industry.

---

## 📌 Notes

* This repository is **educational and experimental**
* Models and scripts are designed for **learning and iteration**, not production scale
* Concepts explored here are directly transferable to **industry-grade MLOps pipelines**

---

## 📈 Progress Tracking

This repository will continue to evolve as I explore:

* Model promotion strategies (champion vs candidate)
* REST API-based model serving
* CI/CD integration for ML systems
* Scalable ML deployment patterns

---

## 👤 Author

**Taqiyudin Miftah Adn**
Computer Engineering Student | Aspiring AI & MLOps Engineer

> *"Learning MLOps is not just about training better models, but about building systems that can survive change."*
