# 🚗 CarDekho Used Car Price Predictor

## 📖 Overview
This project is an end-to-end Machine Learning pipeline that predicts the market value of used cars based on the CarDekho dataset. It features a fully trained XGBoost Regressor deployed as an interactive web application using Streamlit. 

## 📊 Dataset
The data used for training this model is the **CarDekho Used Car Dataset**, which contains extensive details on used car listings in India, including manufacturing year, kilometers driven, fuel type, transmission, and engine specifications. 

* **Source:** [Download the Dataset on Kaggle]([YOUR_KAGGLE_LINK_HERE](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho/data))

*(Note: The raw 70MB CSV dataset is not hosted in this repository due to size constraints, but the fully trained XGBoost model and preprocessing pipelines are included and ready to use.)*

## 🌟 Key Features
* **Exploratory Data Analysis (EDA):** Comprehensive analysis of car features, handling extreme outliers, and identifying key price drivers using VIF to eliminate multicollinearity.
* **Advanced Preprocessing:** Log transformations for skewed data, Label Encoding for high-cardinality features (like Car Models), and One-Hot Encoding for standard categories.
* **Model Optimization:** Compared multiple algorithms including Random Forest, Ridge Regression, and XGBoost, selecting the optimal model based on R² and Mean Absolute Error (MAE).
* **Interactive Web App:** A clean, user-friendly Streamlit interface for real-time price predictions.

## 🛠️ Technologies Used
* **Language:** Python
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit

## 🚀 How to Run Locally
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   
