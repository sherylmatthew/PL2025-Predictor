# 🏆 Premier League Winner Prediction

This repository contains a machine learning project that predicts the **winner of the English Premier League (EPL)** by analyzing historical match and player statistics. The project combines **data analytics** and **predictive modeling** to simulate season outcomes and estimate winning probabilities.  

---

## ⚙️ Installation

### Step 1: Setup a virtualenv / conda environment

**For conda:**
```bash
conda create -n epl_predict python=3.10
conda activate epl_predict
```
**For virtualenv:**
```bash
python -m venv epl_predict
source epl_predict/bin/activate   # Linux/Mac
epl_predict\Scripts\activate      # Windows
```
Step 2: Install dependencies
```bash
pip install -r requirements.txt
```
📂 Project Structure
```bash
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and model experiments
├── src/                # Data preprocessing, ML models, simulations
├── outputs/            # Results and visualizations
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

🚀 Usage

1. Preprocess data
   ```bash
   python src/data_preprocessing.py
2. Train model
   ```bash
   python src/train_model.py
3. Run season simulation
   ```bash
   python src/simulate_season.py

---
## 📊 Results

### Example output (2025/26 EPL season forecast):

Man City: 58% chance of winning

Arsenal: 27%

Liverpool: 12%

Others: 3%

## 📌 Future work

Add player-level performance data

Integrate injury/transfer updates

Monte Carlo season simulations

Deploy interactive web dashboard
