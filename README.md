# ⚡ Energy Consumption Forecasting with Temporal Fusion Transformer (TFT)

This project implements an **end-to-end deep learning pipeline** for **short-term electricity demand forecasting** using the **Temporal Fusion Transformer (TFT)**.  
It demonstrates how modern Transformer-based time-series models can accurately forecast energy consumption while remaining interpretable.

---

## 🚀 Project Highlights

- Multi-consumer energy forecasting using real-world data  
- Probabilistic **24-hour ahead predictions** using quantile regression  
- End-to-end pipeline from raw data to trained model  
- Built-in **model explainability** for feature importance analysis  
- GPU-accelerated training with PyTorch Lightning  

---

## 🧠 Model

Uses the **Temporal Fusion Transformer (TFT)**, a state-of-the-art architecture for time-series forecasting that combines:
- Sequence modeling and attention mechanisms  
- Static and time-varying feature selection  
- Multi-horizon probabilistic outputs  
- Interpretable feature importance

---

## 📊 Dataset

- **Electricity Load Diagrams 2011–2014**
- Real electricity consumption data
- Resampled to **hourly resolution**
- Multiple consumers (MT_002, MT_004, MT_005, MT_006, MT_008)

---

## 🔧 Pipeline Overview

1. **Data preprocessing**
   - Missing value handling
   - Hourly resampling
   - Consumer-wise time-series alignment

2. **Feature engineering**
   - Hour, day, weekday, month
   - Hours and days from start

3. **Model training**
   - Encoder: past 7 days (168 hours)
   - Decoder: next 24 hours
   - Quantile loss for probabilistic forecasting
   - Early stopping and gradient clipping

4. **Evaluation & visualization**
   - Validation MAE (p50)
   - Consumer-wise performance
   - Forecast vs actual plots

5. **Explainability**
   - Built-in TFT feature importance analysis

---

## 📈 Results (Validation)

- **Overall MAE (p50): ~6.45**
- Strong performance on stable consumers
- Captures daily and weekly seasonality effectively

---

## 🧠 Why This Matters

This project mirrors **real-world energy forecasting and trading scenarios**, where:
- Accurate short-term forecasts reduce operational risk
- Probabilistic outputs support better decision-making
- Model interpretability is critical for trust and deployment

---

## 🛠️ Tech Stack

- Python
- PyTorch
- PyTorch Lightning
- PyTorch Forecasting
- Temporal Fusion Transformer
- TensorBoard
- GPU (CUDA)

---



