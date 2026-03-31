# Adversarial Fraud Detection in Financial ML Systems

This project explores the robustness of machine learning models used for financial fraud detection under adversarial conditions.

## Overview

Fraud detection systems rely on ML models to identify suspicious transactions. However, small and realistic changes in input features can cause these models to fail.

In this project, I:
- Built a fraud detection model
- Improved recall using class balancing
- Simulated adversarial attacks by modifying key features
- Evaluated how easily the model can be bypassed
- Built a secondary model to detect attacked samples

---

## Key Results

- Fraud Detection Recall: ~0.83  
- Adversarial Attack Success Rate: ~0.57  
- Attack Detection Recall (unseen data): ~0.43  

---

## Key Insight

Even strong ML models can be fooled with small, realistic feature changes, highlighting the importance of robustness in financial systems.

---

## Project Pipeline

1. Data preprocessing and encoding  
2. Train baseline fraud detection model (Random Forest)  
3. Handle class imbalance to improve recall  
4. Identify important features  
5. Generate adversarial samples  
6. Evaluate attack success  
7. Train attack detection model  

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
