# Customer Segmentation Project: Online Retail Store (2009-2010)


## 1. Introduction

This project focuses on performing customer segmentation for an online retail store using two years of transactional data (2009-2010). The primary goal is to identify distinct customer groups based on their purchasing behavior, which will enable the implementation of more targeted marketing strategies and enhance customer relationship management. Beyond analysis, the project aims to develop a practical application that can classify new or existing customers into these predefined clusters.

## 2. Dataset

The dataset utilized for this analysis is the "Online Retail II" dataset, publicly available from the UCI Machine Learning Repository.

**Dataset Link:** <https://archive.ics.uci.edu/dataset/502/online+retail+ii>

**Key Attributes:**

* `InvoiceNo`: Invoice number. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.

* `StockCode`: Product (item) code. A 5-digit integral number uniquely assigned to each distinct product.

* `Description`: Product (item) name.

* `Quantity`: The quantities of each product (item) per transaction.

* `InvoiceDate`: Invoice date and time. The day and time when a transaction was generated.

* `UnitPrice`: Unit price. Product price per unit in sterling (£).

* `CustomerID`: Customer number. A 5-digit integral number uniquely assigned to each customer.

* `Country`: Country name. The name of the country where a customer resides.

## 3. Project Process

The project followed a standard data science methodology, outlined below:

### 3.1 Data Exploration & EDA

Initial exploration revealed critical data quality issues:

* **Missing Values:** Significant nulls in `CustomerID`, crucial for segmentation.

* **Negative Values:** `Quantity` and `UnitPrice` contained negative values, often linked to cancellations or administrative adjustments.

* **Invoice Types:** Identified normal invoices, cancellation invoices (prefix 'C'), and 'Adjust Debt Back' invoices (prefix 'A').

* **Irrelevant StockCodes:** Several `StockCode` entries (e.g., 'D', 'M', 'BANK CHARGES', 'AMAZONFEE') were found to be non-product related and excluded.

![StockCode Image](/images/table_01.png)


### 3.2 Data Cleaning

The following steps were performed to ensure data quality:

* **Invoice Filtering:** Only valid, non-cancellation, non-adjustment invoices were retained.

* **StockCode Filtering:** Only legitimate product `StockCode` entries were kept, excluding administrative or non-product codes.

* **Missing Customer IDs:** Rows with missing `CustomerID` were removed, as these records cannot be attributed to a specific customer.

**Impact:** Approximately 23% of the original dataset was removed during this cleaning phase to ensure the integrity of the customer segmentation. The cleaned data was saved as `cleaned_df.csv`.

### 3.3 Feature Engineering

To capture customer behavior, RFM (Recency, Frequency, Monetary) features were engineered:

* **MonetaryValue:** Sum of `Quantity` \* `UnitPrice` for each customer.

* **Frequency:** Number of unique invoices per customer.

* **Recency:** Days since the customer's last purchase (calculated from the latest `InvoiceDate` in the dataset).

**Outlier Treatment:**
Histograms and box plots revealed significant outliers in `MonetaryValue` and `Frequency`. The Interquartile Range (IQR) method was applied to remove these outliers, ensuring a more robust clustering process.

![Box Plot with Outliers](/images/box.png)

![Box Plot without Outliers](/images/box2.png)

**Feature Scaling:**
Standardization (Z-score normalization) using `StandardScaler` was applied to `MonetaryValue`, `Frequency`, and `Recency`. This step is crucial to prevent features with larger scales from disproportionately influencing the clustering algorithm. The scaled data was saved as `scaled.csv`.

![Scatter Plot](/images/scatter1.png)

### 3.4 Clustering

**Algorithm:** K-Means clustering was employed due to its effectiveness and interpretability for segmentation tasks.

**Optimal K Determination:**
The optimal number of clusters (k) was determined using two common methods:

* **Elbow Method (Inertia):** The inertia plot showed a clear "elbow" at k=4, indicating that additional clusters beyond this point offer diminishing returns in reducing within-cluster variance.

* **Silhouette Score:** The silhouette score, which measures how well each object is clustered, peaked at k=4.

Based on these evaluations, **k=4 was selected as the optimal number of clusters**.


![Inertia Plot](/images/inertia.png)

**Clustering Execution:**
K-Means was trained with `n_clusters=4`, and the resulting cluster labels were assigned back to the `filtered_df`. The final silhouette score for the model was **0.42**.

![Clustering scatter plot](/images/scatter2.png)

## 4. Customer Segments & Recommendations

Four distinct customer segments were identified, each with unique characteristics and tailored strategic recommendations:

### Cluster 0: "Loyal & Engaged" (Blue)

* **Interpretation:** These are your core customers, demonstrating consistent engagement (mid-to-high frequency) and a respectable monetary contribution. Their low recency signifies recent activity, confirming their active and loyal status.

* **Recommendations:**

  * Implement exclusive loyalty programs and tiered rewards.

  * Provide personalized communications and product recommendations.

  * Encourage feedback and leverage them as brand advocates through referral programs.

### Cluster 1: "At-Risk & Dormant" (Orange)

* **Interpretation:** This segment comprises customers showing clear signs of disengagement. They have low monetary value, low frequency, and high recency, indicating a significant period without purchases and limited past interaction. They are either churning or have already become inactive.

* **Recommendations:**

  * Develop targeted re-engagement campaigns with compelling offers.

  * Offer irresistible win-back incentives for a return purchase.

  * Investigate reasons for inactivity to prevent future churn.

### Cluster 2: "Occasional Buyers" (Green)

* **Interpretation:** These customers have made a purchase or two but lack consistent engagement. Their lower monetary value and frequency, coupled with mid-level recency, suggest they might be price-sensitive, seasonal, or still exploring options.

* **Recommendations:**

  * Introduce them to a wider range of products through personalized recommendations or bundles.

  * Reinforce your value proposition to encourage future purchases.

  * Offer targeted promotions aligned with their past behavior to encourage repeat transactions.

### Cluster 3: "Champions & VIPs" (Red)

* **Interpretation:** This is your most valuable and influential customer segment. They exhibit exceptional engagement (high frequency), contribute significantly to revenue (high monetary value), and have made recent purchases. They are highly satisfied, brand loyal, and likely your biggest advocates.

* **Recommendations:**

  * Provide exclusive VIP treatment, priority service, and early access to new offerings.

  * Send personalized appreciation and recognition.

  * Leverage their enthusiasm through robust referral programs.

  * Engage them in product development or service improvement discussions for invaluable insights.

## 5. Future Work

The project will proceed with the following phases:

* **Backend Development (FastAPI):** Develop a robust API using FastAPI to host the trained K-Means model. This API will classify new customer data (MonetaryValue, Frequency, Recency) into one of the four identified clusters.

* **Frontend Development:** Create a user-friendly web application that allows users to input customer parameters and visualize the assigned cluster, providing a practical tool for sales and marketing teams.

## 6. How to Run the Project

*(This section will be populated once the backend and frontend development are complete.)*

## 7. Dependencies

*(This section will list all required Python libraries and their versions once the project is finalized.)*

## 8. Author

*(Your Name/Team Name)*
*(Optional: Contact Information, GitHub Profile, LinkedIn Profile)*