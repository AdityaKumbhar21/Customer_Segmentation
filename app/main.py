from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime, timedelta


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = 'utils/model.pkl'
SCALER_PATH = 'utils/scaler.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")


CLUSTER_INTERPRETATIONS = {
    0: {
        "name": "Loyal & Engaged",
        "interpretation": "Core customer base with consistent engagement, mid-to-high frequency, and mid-level monetary contribution. Lower recency indicates recent activity.",
        "recommendations": [
            "Implement exclusive loyalty programs and tiered rewards.",
            "Tailor personalized communications and product recommendations.",
            "Encourage feedback and leverage them as brand advocates."
        ]
    },
    1: {
        "name": "At-Risk & Dormant",
        "interpretation": "Customers showing signs of disengagement: low monetary value, low frequency, and high recency. They are either churning or already inactive.",
        "recommendations": [
            "Develop targeted re-engagement campaigns with compelling offers.",
            "Offer irresistible win-back incentives for a return purchase.",
            "Investigate reasons for inactivity to prevent future churn."
        ]
    },
    2: {
        "name": "Occasional Buyers",
        "interpretation": "Customers who have made a purchase or two but lack consistent engagement. Lower monetary value and frequency, with mid-level recency. Might be price-sensitive or exploring options.",
        "recommendations": [
            "Introduce them to a wider range of products through personalized recommendations or bundles.",
            "Reinforce your value proposition to encourage future purchases.",
            "Offer targeted promotions aligned with their past behavior."
        ]
    },
    3: {
        "name": "Champions & VIPs",
        "interpretation": "Most valuable and influential customer segment. Exceptional engagement (high frequency), significant monetary contribution, and recent purchases. Highly satisfied and brand loyal.",
        "recommendations": [
            "Provide exclusive VIP treatment, priority service, and early access to new offerings.",
            "Send personalized appreciation and recognition.",
            "Leverage their enthusiasm through robust referral programs.",
            "Engage them in product development or service improvement discussions."
        ]
    }
}


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['Invoice','StockCode','Description','Quantity','InvoiceDate','UnitPrice','Customer ID','Country']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain all required columns: {required_cols}")
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Invoice'] = df['Invoice'].astype("str")
    df = df[df['Invoice'].str.match("^\\d{6}$") == True]
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    max_invoice_date = df['InvoiceDate'].max()
    aggregated_df = df.groupby(by='Customer ID', as_index=False).agg(
        MonetaryValue=('TotalSales', 'sum'),
        Frequency=('Invoice', 'nunique'),
        LastPurchase=('InvoiceDate', 'max')
    )
    print(df.info())
    aggregated_df['Recency'] = (max_invoice_date - aggregated_df['LastPurchase']).dt.days
    rfm_features = aggregated_df[['MonetaryValue', 'Frequency', 'Recency']]

    return rfm_features, aggregated_df['Customer ID']


@app.post('/classify_customers')
async def classify_customers(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Invalid File Type. Please uplaod csv file only')
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        rfm_features, customer_ids = calculate_rfm(df)
        print(df.head())

        print(rfm_features.head())
        print(type(rfm_features))
        scaled_rfm_features = scaler.transform(rfm_features)
        cluster_labels = model.predict(scaled_rfm_features)


        results = []
        for i, customer_id in enumerate(customer_ids):
            cluster_id = cluster_labels[i]
            cluster_info = CLUSTER_INTERPRETATIONS.get(cluster_id, {
                "name": "Unknown Cluster",
                "interpretation": "No specific interpretation available.",
                "recommendations": []
            })
            results.append({
                "customer_id": str(customer_id),
                "predicted_cluster_id": int(cluster_id), 
                "cluster_name": cluster_info["name"],
                "cluster_interpretation": cluster_info["interpretation"],
                "cluster_recommendations": cluster_info["recommendations"],
                "rfm_values": {
                    "MonetaryValue": float(rfm_features.iloc[i]['MonetaryValue']),
                    "Frequency": int(rfm_features.iloc[i]['Frequency']),
                    "Recency": int(rfm_features.iloc[i]['Recency'])
                }
            })

            return JSONResponse(content=results)


    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {e}")


