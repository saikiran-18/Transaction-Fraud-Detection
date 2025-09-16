
# **Transaction Fraud Detection**

This project demonstrates an **unsupervised machine learning** approach to detect potentially fraudulent transactions using clustering and anomaly detection techniques. The model is trained on a synthetic dataset to identify unusual transaction patterns without relying on pre-labeled fraud data. This approach is highly effective for discovering new, unseen types of fraudulent activity.

## **📋 Table of Contents**

* [Project Overview](https://www.google.com/search?q=%23project-overview)  
* [Project Structure](https://www.google.com/search?q=%23project-structure)  
* [Methodology](https://www.google.com/search?q=%23methodology)  
* [Key Findings](https://www.google.com/search?q=%23key-findings)  
* [Recommendations](https://www.google.com/search?q=%23recommendations)  
* [Installation and Usage](https://www.google.com/search?q=%23installation-and-usage)

## **📝 Project Overview**

The primary goal of this project is to identify suspicious financial transactions using an unsupervised learning model. Instead of predicting a specific "fraud" label, the model analyzes the transaction data to find natural groupings and flag transactions that deviate from the norm.

* **Domain:** Finance / E-commerce  
* **Dataset:** Synthetic Fraud Dataset (CSV)  
* **Core Subject:** Unsupervised Machine Learning (Clustering)

## **📁 Project Structure**

The project is structured as a Jupyter Notebook that follows a clear, multi-phase methodology:

1. **Problem Understanding:** Defines the project's goal and justifies the use of an unsupervised approach for fraud detection.  
2. **Data Collection & Preparation:** Loads and explores the dataset, checking for missing values and data types.  
3. **Exploratory Data Analysis (EDA):** Visualizes the data to understand the distribution and correlations of key features.  
4. **Feature Engineering & Preprocessing:** Scales numerical features and handles categorical data to prepare the dataset for the model.  
5. **Dimensionality Reduction:** Reduces the number of features using **Principal Component Analysis (PCA)** to make the data more manageable for clustering.  
6. **Clustering & Anomaly Detection:** Utilizes the **K-Means algorithm** to group transactions into clusters and then analyzes those clusters to identify the most likely "fraud-prone" group.  
7. **Insights & Recommendations:** Interprets the model's findings and provides actionable recommendations.  
8. **Deployment:** Demonstrates how to save and load the trained models and includes a Streamlit dashboard for real-time predictions on new data.

## **🧠 Methodology**

The core of the methodology is a robust clustering workflow designed for efficiency and accuracy:

1. **Elbow Method:** The Elbow Method is used on the inertia values of the K-Means model to determine the optimal number of clusters (k).  
2. **Silhouette Score Validation:** A subset of the data is used to calculate the silhouette score for the best candidate values of k from the Elbow Method. This balances speed with the reliability of the score.  
3. **Model Training:** The final K-Means model is trained on the full, PCA-reduced dataset using the optimal value of k.  
4. **Cluster Profiling:** The average values of key features (e.g., Risk\_Score, Transaction\_Amount) are calculated for each cluster to understand its characteristics.  
5. **Fraud Identification:** A "High-Risk / Fraud-Prone" cluster is identified based on profiling, typically by looking for small, distinct clusters with a high average risk score.

## **💡 Key Findings**

* The dataset was clean, with no missing values, allowing for a straightforward preprocessing pipeline.  
* PCA successfully reduced the dimensionality of the data, retaining a significant portion of the variance.  
* K-Means clustering effectively segmented the transaction data into distinct groups.  
* The profiling of these clusters revealed one as a clear outlier, characterized by a higher average Risk\_Score and other fraud indicators. This cluster was labeled as "High-Risk".

## **🚀 Recommendations**

* **Prioritize the high-risk cluster:** Any new transaction falling into the "High-Risk / Fraud-Prone" cluster should be flagged for immediate review.  
* **Categorical Feature Analysis:** Conduct further analysis of the categorical features (Transaction\_Type, Merchant\_Category, etc.) within the high-risk cluster to uncover specific fraud patterns.  
* **Model Enhancements:** Explore other anomaly detection algorithms (e.g., Isolation Forest, Local Outlier Factor) to complement this clustering approach and potentially improve detection accuracy.

## **⚙️ Installation and Usage**

1. Clone this repository:  
   git clone \[repository\_url\]  
   cd Transaction\_Fraud\_Detection

2. Install the required Python libraries:  
   pip install \-r requirements.txt

3. Run the Jupyter Notebook to train the model and save the necessary files (scaler.pkl, pca.pkl, fraud\_model.pkl, fraud\_cluster.pkl).  
4. Launch the Streamlit dashboard:  
   streamlit run dashboard.py

Here is the sample images of streamlit outputs:
<h3>🔹 Dashboard Input</h3>
![Transaction Fraud Detection Input]("D:\saikiran's folder\files\Picture\Pictures\Screenshots\Screenshot (28).png")

<h3>🔹 Dashboard Output</h3>
![Transaction Fraud Detection Output]("D:\saikiran's folder\files\Picture\Pictures\Screenshots\Screenshot (29).png")


