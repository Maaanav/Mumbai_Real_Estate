#  Mumbai Real Estate Price Analysis & Prediction

An end-to-end data analytics and machine learning project to analyze and predict property prices in Mumbai city using Python and SQL.

##  Project Overview
This project demonstrates the complete data lifecycle:
1. **Data Cleaning & Preprocessing**  
   - Converted Lakh/Crore to INR  
   - Engineered features like `price_per_sqft`, `area_per_bhk`  
   - Removed outliers using IQR filtering  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized price distributions, locality trends, and BHK insights  

3. **Modeling**  
   - Built baseline (median-based) predictor  
   - Trained and evaluated **Random Forest** and **Gradient Boosting** regressors  
   - Saved best model using `joblib`

4. **SQL Integration**  
   - Exported cleaned data into a **SQLite database**  
   - Wrote analytical SQL queries (price/sqft by region, median by BHK)

5. **Prediction Helper**  
   - Custom function to predict price given property features  

---

##  Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, Joblib  
- **Database:** SQLite  
- **Environment:** VS Code / Jupyter Notebook  

---

