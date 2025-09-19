
# **Transaction Fraud Detection-Unsupervised learning**

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

 # **Transaction Fraud Detection- supervised learning**

* **Project Title:** Start with a clear and distinct title for the project, such as "**Transaction Fraud Detection: Supervised Learning Approach**".
* **Core Subject:** Specify that this project falls under **Supervised Machine Learning**.
* **Problem Statement:** Briefly explain the goal of this project, which is to build a classification model to **predict fraudulent transactions** using a labeled dataset.
* **Dataset:** Mention the dataset used (e.g., Synthetic Fraud Dataset) and clarify that unlike the unsupervised project, this notebook uses the `Fraud_Label` column as the target variable for training.
* **Methodology:**
    * **Feature Selection:** List the specific features you used for training (`Transaction_Amount`, `Daily_Transaction_Count`, `Account_Balance`, `Transaction_Distance`) and mention that these were selected because they are "user-knowable".
    * **Data Preprocessing:** Describe the steps taken, such as using `StandardScaler` to normalize the feature data.
    * **Model:** State that you chose a **Random Forest Classifier** and explain why it was suitable for this task (e.g., it handles class imbalance using `class_weight="balanced"`).
* **Results and Evaluation:**
    * **Performance Metrics:** Share the key evaluation metrics from your model's performance on the test set. Specifically, mention the **Accuracy**, **Precision**, **Recall**, and **F1-score**.
    * **Low Recall:** Highlight the low recall score (0.0624) as a critical finding. This demonstrates the model's limitation in correctly identifying all fraudulent transactions, which is an important insight to share.
    * **Feature Importance:** Explain which features the model found most significant in its predictions (e.g., `Transaction_Distance`, `Account_Balance`, `Transaction_Amount`).
* **Dashboard:** Describe the Streamlit dashboard (`app.py`) that you created. Explain that it uses the trained model to predict whether a new transaction is fraudulent and provides a "Calculated Risk_Score".
* **Conclusion:** Briefly summarize the project's outcome and the limitations of this specific supervised model, particularly its struggle with identifying all fraud cases, as indicated by the low recall. This shows that you've critically analyzed the model's performance.




