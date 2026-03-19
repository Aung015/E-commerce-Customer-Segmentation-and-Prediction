# 🛒 E-commerce Customer Segmentation and Prediction

## 📌 Project Overview
This project develops a comprehensive machine learning solution for **customer segmentation and purchase prediction** in an e-commerce environment. By leveraging customer transaction data, the system identifies distinct customer groups and predicts future purchasing behavior, enabling businesses to optimize marketing strategies, improve customer retention, and enhance decision-making.

The project integrates **unsupervised learning (clustering)** and **supervised learning (classification)** techniques, and is deployed using an interactive **Streamlit web application**.

---

## 🎯 Objectives
- Segment customers based on purchasing behavior
- Predict future purchase likelihood
- Enable targeted marketing strategies
- Improve customer retention and engagement
- Support business decision-making with data-driven insights

---

## 📊 Key Features
- 🔍 **Exploratory Data Analysis (EDA)** to understand customer behavior  
- 🧮 **RFM Analysis** (Recency, Frequency, Monetary) for segmentation  
- 🤖 **Clustering Algorithms**
  - K-Means
  - Hierarchical Clustering
  - DBSCAN  
- 📈 **Model Evaluation Metrics**
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index  
- 🌲 **Predictive Modeling**
  - Random Forest Classifier for:
    - Customer segment prediction
    - Future purchase prediction  
- 📊 **Feature Importance Analysis**  
- 🌐 **Streamlit Web Application** for real-time predictions  

---

## 🧠 Methodology

### 1. Data Preprocessing
- Removed invalid and returned transactions  
- Handled missing values  
- Converted data types  
- Filtered relevant features  

### 2. Feature Engineering
Constructed key customer features:
- **Recency** – Days since last purchase  
- **Frequency** – Number of transactions  
- **Monetary** – Total spending  
- **TotalItems** – Total quantity purchased  
- **AvgOrderValue** – Average order value  

### 3. Customer Segmentation
- Applied multiple clustering techniques  
- Evaluated performance using clustering metrics  
- Selected **K-Means** as the optimal model  
- Generated interpretable customer segments  

### 4. Predictive Modeling
- Built a **Random Forest classifier**  
- Predicted:
  - Customer segment
  - Future purchase likelihood  
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC Score  

---

## 👥 Customer Segments
- **High-Value Customers**  
  Frequent buyers with high spending  

- **Regular Customers**  
  Moderate activity and consistent purchases  

- **Inactive Customers**  
  Low engagement and high churn risk  

---

## 💡 Business Insights & Recommendations

### High-Value Customers
- Offer loyalty programs and exclusive deals  
- Provide early access to new products  

### Regular Customers
- Use upselling and cross-selling strategies  
- Send personalized recommendations  

### Inactive Customers
- Run re-engagement campaigns  
- Provide discounts and reminders  

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit app that allows:
- Real-time customer segmentation  
- Future purchase prediction  
- Batch prediction using CSV upload  
- Business recommendations based on customer segment  

### ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
