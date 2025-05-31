# 🇯🇲 Jamaican Airbnb Price Estimator

This is a beginner-friendly machine learning project that estimates the nightly price of an Airbnb in Jamaica based on property features like location, number of bedrooms, amenities, and more. Built using Python, scikit-learn, and Streamlit, this web app is intended to be a practical introduction to end-to-end machine learning for real estate and travel data.

---

## 🚀 Demo

> Try it locally: enter your property details and instantly receive a predicted nightly rate in USD.

---

## 📦 Features

- 🧠 Trained ML model using scikit-learn’s Linear Regression
- 📊 One-hot encoding and feature engineering for locations and amenities
- 🖥️ Streamlit app for easy UI interaction
- 📁 Clean project structure and modular code for training and prediction
- ✅ Ready to deploy or extend with advanced models in future versions

---

## 🛠 Tech Stack

| Tool          | Purpose                         |
|---------------|----------------------------------|
| Python        | Core programming language       |
| pandas        | Data cleaning and manipulation  |
| scikit-learn  | Machine learning pipeline       |
| Streamlit     | Web application framework       |
| matplotlib / seaborn | Data visualization       |
| Jupyter       | Exploratory analysis notebook   |

---

## 🧪 How to Run This Project

### 1. Clone the Repo

```bash
git clone https://github.com/stefenewers/jamaican-airbnb-price-estimator.git
cd jamaican-airbnb-price-estimator
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install All Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the ML Model

```bash
python src/train_model.py
```

> This script loads the dataset, trains a regression model, prints the Mean Absolute Error, and saves the trained model to a `.pkl` file.

### 5. Run the Streamlit App

```bash
streamlit run app/app.py
```

> A browser window will open where you can interact with the price estimator.

---

## 🧠 How It Works

### 📂 Dataset
A mock dataset of 1,000 rows was created using real-world-inspired statistics for Airbnb properties across Jamaican locations (Negril, Kingston, Montego Bay, etc.). Features include:

- `location`
- `bedrooms`, `bathrooms`, `accommodates`
- `has_wifi`, `has_pool`, `has_ac`, `has_kitchen`, `has_parking`
- `price` (target)

### 🏗️ Model Training
- Data is cleaned and encoded (using one-hot for `location`)
- A `LinearRegression` model is trained on 80% of the data
- Evaluated using Mean Absolute Error (MAE)
- The model is serialized with `joblib` for reuse

### 💡 Prediction
- The Streamlit app accepts inputs from the user
- Features are transformed to match the model
- The model returns a nightly price in USD

---

## 🧭 Project Structure

```
jamaican-airbnb-price-estimator/
├── app/
│   └── app.py                 # Streamlit app
├── data/
│   └── jamaican_airbnb_mock_dataset.csv
├── notebooks/
│   └── eda.ipynb              # Data exploration
├── src/
│   ├── train_model.py         # ML training pipeline
│   ├── predict.py             # (optional) standalone predictor
│   └── linear_model.pkl       # Trained model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Sample Prediction

Example:  
A 2-bedroom villa in Negril with WiFi, AC, and a pool → **Estimated Price: $140.25 USD/night**

---

## 🧩 Next Steps (Version 2 Ideas)

- Replace Linear Regression with Random Forest or XGBoost
- Add real data using Airbnb scraping or APIs
- Include confidence intervals or SHAP visualizations
- Support uploading CSV for batch predictions
- Deploy the app on Streamlit Cloud or HuggingFace Spaces

---

## 🤝 Credits

Developed by **Stefen Ewers**  
Connect on [LinkedIn](https://www.linkedin.com/in/stefen-ewers/)  
Follow on [GitHub](https://github.com/stefenewers)

---

## 📜 License

This project is for educational and non-commercial use.
