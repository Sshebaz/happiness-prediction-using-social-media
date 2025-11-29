# **Happiness Index Prediction – README**

## **1. Title**

### **Happiness Index Prediction Using Social Media & Lifestyle Data**

---

## **2. Executive Summary**

This project predicts an individual’s Happiness Index by analyzing lifestyle habits and social media usage patterns. Using machine learning, the system evaluates factors such as screen time, sleep quality, stress, exercise frequency, and platform preference to generate real-time happiness predictions through a Flask-based interactive web application.

---

## **3. Business Problem**

With increasing digital dependency, organizations, mental‑health platforms, and wellness companies need insights into how online behavior affects emotional well‑being. This project aims to quantify happiness levels and provide actionable insights that can help improve productivity, mental health initiatives, and user engagement strategies.

---

## **4. Methodology**

* Data collection and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training using supervised machine learning
* Model evaluation and tuning
* Flask web deployment with interactive UI
* Real‑time prediction generation

---

## **5. Flowchart / Project Structure**

```
Project Folder
│
├── app.py                → Main Flask application
├── requirements.txt      → Dependencies
├── venv/                 → Enviroment
├── templates/            → HTML templates (index, form, results)
├── artifacts/            → Trained ML model.pkl, preprocessor.pkl, row.csv, train.csv, test.csv
├── src/
│   ├── components/       → Data ingestion, data transformation, model training modules
│   ├── pipeline/         → Training & prediction pipeline
│   └── utils.py          → Helper functions
└── Notebook/data         → Dataset files
```

## **6. Skills Used**

* Python
* Flask Web Framework
* Machine Learning (Classification/Regression)
* EDA & Data Visualization
* HTML/CSS/Bootstrap UI Design
* Model Deployment

---

## **7. Result & Business Recommendation**

**Result:** The model successfully predicts a Happiness Index score based on user behavior and lifestyle choices.

**Business Recommendation:**

* Use predictions to offer personalized wellness suggestions.
* Integrate insights into corporate wellness programs.
* Track digital habits to improve mental‑health outcomes.
* Build a mobile app for continuous happiness monitoring.

---

## **8. Next Steps**

* Add more psychological and behavioral features.
* Train the model on a larger, more diverse dataset.
* Implement predictive analytics dashboards.
* Deploy a cloud‑based version with user accounts.
* Introduce recommendation engine for lifestyle improvement.
