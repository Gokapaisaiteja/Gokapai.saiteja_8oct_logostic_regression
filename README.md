
# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will leave (churn), using the Telco Customer Churn dataset from Kaggle.

## Dataset

- Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers, 21 columns
- About 26.5% of customers churned

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in the project folder. The file is not included in this repo.

## What the script does

1. Loads and cleans the data (fixes `TotalCharges`, drops 11 blank rows and `customerID`)
2. Encodes categorical columns into numbers
3. Splits the data 80/20 into training and test sets
4. Trains Logistic Regression and Random Forest models
5. Saves the final model as `churn_model.pkl`

## Results

| Model                          | Accuracy | Precision | Recall | F1    | AUC   |
| ------------------------------ | -------- | --------- | ------ | ----- | ----- |
| Logistic Regression            | 0.804    | 0.648     | 0.575  | 0.609 | 0.836 |
| Random Forest                  | 0.787    | 0.621     | 0.508  | 0.559 | 0.818 |
| Logistic Regression (balanced) | 0.726    | 0.491     | 0.797  | 0.608 | 0.835 |

Precision, recall and F1 are for the churn class.

The final model is the **balanced Logistic Regression**. It catches about 80% of customers who actually churn, at the cost of more false alarms. That trade-off suits a retention campaign, where missing a churner usually costs more than contacting a customer who would have stayed.

## Key findings

- Fiber optic internet customers and customers with high total charges are more likely to churn.
- Streaming TV and streaming movies add-ons are also linked to higher churn.

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python churn_prediction.py
```

## Tech stack

Python, pandas, scikit-learn, joblib
