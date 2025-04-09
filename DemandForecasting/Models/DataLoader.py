import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('DemandForecasting/data/DemandData.xlsx')

#extract the datetime features to analyze the correlation
def datetimeFeatures(df):
    df['Month'] = df['Datetime'].dt.month
    df['day'] = df['Datetime'].dt.day
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.day_of_week
    df['is_weekend'] = df['Datetime'].dt.weekday >= 5
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    return df

data = datetimeFeatures(data)

def plotHeatMap(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Heatmap")

    script_path = os.path.dirname(os.path.abspath(__file__))  
    save_path = os.path.join(script_path, "plots/correlation_heatmap.png")  

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to: {save_path}")
    
plotHeatMap(data)

def dataPreprocessing(df):
    scaler = MinMaxScaler()
    df['redistributed'] = scaler.fit_transform(df[['redistributed']])
    return data

data = dataPreprocessing(data)


