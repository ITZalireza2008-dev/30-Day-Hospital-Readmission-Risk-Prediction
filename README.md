# Predicting 30-Day Hospital Readmissions using Machine Learning

## 🏥 Executive Summary & Clinical Impact
Hospital readmissions within 30 days are a major financial penalty risk for healthcare systems and a critical marker of post-discharge care coordination breakdowns. This project builds an end-to-end predictive machine learning pipeline using real-world hospital data to identify high-risk patients **before discharge**, enabling proactive interventions such as home nurse visits and targeted telehealth follow-ups.

* **Clinical Actionability:** Patients are segmented into automated Risk Tiers (>70% High Risk, 30%–70% Medium Risk) to optimize hospital care-management staffing and resource allocation.
* **Domain Engineering:** Developed custom clinical proxies—including a **Service Utilization Index**, a **Polypharmacy Indicator**, and **ICD-9 Comorbidity Mapping**—to accurately capture patient frailty and chronic disease burden.

---

## 📊 Key Model Evaluation Metrics
* **Algorithm:** Random Forest Classifier (optimized via RandomizedSearchCV with `class_weight='balanced'` to handle severe class imbalance).
* **ROC-AUC Score:** ~0.685 *(Demonstrates strong ranking capability across imbalanced readmission classes)*.
* **Recall & Precision:** Tuned to capture a high percentage of true readmissions, balancing the need to prevent missed high-risk patients with operational care-management bandwidth.

---

## 🔍 Top Clinical Risk Drivers (Discovered via SHAP)
Using **SHAP (SHapley Additive exPlanations)**, the model uncovers the root behavioral and clinical factors driving readmissions:
1. **Prior Inpatient Admissions:** The single strongest historical predictor of future readmission risk.
2. **Service Utilization Index:** High cumulative healthcare dependency across outpatient, emergency, and inpatient settings correlates sharply with acute readmission probability.
3. **Polypharmacy Count:** Patients managing complex multi-drug regimens experience higher post-discharge failure rates unless structured medication reconciliation is performed.
4. **Primary Diagnosis Categories:** Circulatory and Endocrine/Metabolic conditions drive the highest clustering of early readmissions.

---

## 📂 Repository Structure
```text
30-Day-Hospital-Readmission-Risk-Prediction/
│
├── notebooks/
│   └── hospital_readmission_pipeline.ipynb   # Complete end-to-end notebook (Cleaning, EDA, Modeling, SHAP)
├── diabetic_data.csv
│
└── README.md