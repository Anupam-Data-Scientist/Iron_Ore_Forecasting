# Iron Ore Price Forecasting 📈

This project aims to build a predictive model for forecasting iron ore prices using a combination of machine learning and deep learning techniques. It provides insights to support decision-making in the mining and commodities sectors.

---

## 📌 Objective
To forecast future iron ore prices based on historical data using models such as Random Forest, XGBoost, SVR, LSTM, and GRU.

---

## 🛠️ Tech Stack

- **Languages & Frameworks**: Python, Streamlit
- **ML/DL Models**: Random Forest, XGBoost, SVR, LSTM, GRU
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Sweetviz, TensorFlow, Keras

---

## 📊 Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Used `Sweetviz` to uncover hidden patterns in the historical data.
   - Visualized trends and correlations to guide feature engineering.

2. **Data Preprocessing**  
   - Handled missing values, feature scaling, and data splitting for model training and evaluation.

3. **Model Building & Evaluation**  
   - Trained multiple models and compared them using RMSE, MAE, and cross-validation.
   - Achieved highest accuracy using **Random Forest**.

4. **Deployment**  
   - Built an interactive web interface using **Streamlit** to forecast prices dynamically.

5. **Hyperparameter Tuning**  
   - Optimized key model parameters to improve prediction accuracy by over 15%.

---

## 🚀 Key Achievements

- ✅ Achieved top performance with Random Forest over deep learning models.
- 📉 Reduced forecasting error through effective preprocessing and tuning.
- 🖥️ Deployed a real-time forecasting tool using Streamlit for business usability.

---

## 🧠 Insights

- Time series modeling of commodity prices can significantly enhance supply chain and investment decisions.
- Feature importance analysis helped identify the most impactful factors affecting iron ore pricing.

---

## 📂 Project Structure

Iron_Ore_Forecasting/
│
├── Code/ # All source code and scripts
├── Data_set_Iron/ # Dataset used for training and evaluation
├── README.md # Project documentation


---

## 📎 How to Run the Project

1. Clone the repository  
```bash
git clone https://github.com/Anupam-Data-Scientist/Iron_Ore_Forecasting.git
cd Iron_Ore_Forecasting

## 📎 Install Dependencies
pip install -r requirements.txt

streamlit run Code/app.py
