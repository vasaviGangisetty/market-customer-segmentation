# Market & Customer Segmentation

## Project Overview

This project performs customer segmentation using the Mall Customers Dataset. The goal is to group customers based on their purchasing behavior so businesses can create targeted marketing campaigns.

## Objectives

* Perform customer segmentation using K-Means Clustering
* Apply RFM-style analysis
* Reduce dimensions using PCA
* Generate customer cluster visualizations
* Create customer summary reports

## Dataset

The dataset contains:

* CustomerID
* Gender
* Age
* Annual Income (k$)
* Spending Score (1-100)

Example:

| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
| ---------- | ------ | --- | ------------------ | ---------------------- |
| 1          | Male   | 19  | 15                 | 39                     |
| 2          | Male   | 21  | 15                 | 81                     |
| 3          | Female | 20  | 16                 | 6                      |

## Project Structure

Market_Customer_Segmentation/

├── dataset/

│ └── Mall_Customers.csv

├── outputs/

│ ├── elbow_method.png

│ ├── customer_clusters.png

│ ├── pca_clusters.png

│ └── cluster_report.csv

├── src/

│ ├── data_preprocessing.py

│ ├── rfm_analysis.py

│ ├── pca_analysis.py

│ ├── clustering.py

│ └── visualization.py

├── main.py

├── requirements.txt

├── README.md

└── .gitignore

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

## Installation

Install required libraries:

pip install -r requirements.txt

## Run the Project

Execute:

py main.py

## Outputs

The project generates:

* Customer Segmentation Table
* Customer Summary Table
* Elbow Method Graph
* Customer Cluster Graph
* PCA Cluster Graph
* Cluster Report CSV

## Learning Outcomes

* Data Preprocessing
* Feature Engineering
* RFM Analysis
* Principal Component Analysis (PCA)
* K-Means Clustering
* Cluster Profiling
* Data Visualization
"# market-customer-segmentation" 
