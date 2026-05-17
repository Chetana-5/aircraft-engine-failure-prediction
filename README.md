# ✈️ AI-Based Aircraft Engine Failure Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Dataset](https://img.shields.io/badge/NASA-CMAPSS%20FD001-lightgrey?style=flat-square)

Predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using a Bidirectional LSTM, with SHAP explainability and an interactive Streamlit dashboard.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python 1_lstm_model.py          # train
python 2_shap_explainability.py # explain
streamlit run 3_dashboard.py    # launch UI → http://localhost:8501
```

**Google Colab:** Open the notebook, run all cells. The last cell generates a public `trycloudflare.com` link automatically.

---

## 📁 Files

| File | Description |
|------|-------------|
| `aircraft_engine_rul.ipynb` | Main notebook — full pipeline |
| `1_lstm_model.py` | BiLSTM training + evaluation |
| `2_shap_explainability.py` | SHAP feature importance |
| `3_dashboard.py` | Streamlit dashboard |
| `4_future_enhancements.py` | MC-Dropout, anomaly detection, streaming |

> **Data files needed:** `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`  
> Download from the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## 📊 Results (NASA CMAPSS FD001)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| BiLSTM | 14.82 | 11.53 | 0.831 |
| Attention-LSTM | 13.41 | 10.87 | 0.856 |
| Random Forest | 19.28 | 15.34 | 0.731 |

---

## 🛠️ Tech Stack

`TensorFlow` · `SHAP` · `Streamlit` · `Plotly` · `scikit-learn` · `pandas` · `NumPy`

---

## 📜 Dataset Citation

Saxena et al., *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, PHM 2008, NASA Ames.
