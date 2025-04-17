import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#extract the datetime features to analyze the correlation
def datetimeFeatures(df):
    df['Month'] = df['Datetime'].dt.month
    df['day'] = df['Datetime'].dt.day
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.day_of_week
    df['is_weekend'] = df['Datetime'].dt.weekday >= 5
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    return df

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

def dataNormalizer(df):
    scaler = MinMaxScaler()
    df['redistributed'] = scaler.fit_transform(df[['redistributed']])
    return df,scaler

def dataSequence(df,window_size, step_size, forecast_horizon):
    demand = df['redistributed'].values
    num_samples = len(demand)
    X = []
    y = []

    for start in range(0, num_samples - window_size - forecast_horizon + 1, step_size):
        end = start + window_size
        forecast_end = end + forecast_horizon
        window_dem = demand[start:end-1]
        X.append([window_dem])
        y.append(demand[end-1:forecast_end-1])  

    return np.array(X), np.array(y)

current_file_dir = os.path.dirname(os.path.abspath(__file__))

def plotTrainTestLoss(history,modelName):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(current_file_dir, f'plots/{modelName}')
    os.makedirs(plot_path, exist_ok=True)
    plot_path = os.path.join(current_file_dir, f'plots/{modelName}/TrainAndTestloss.png')
    plt.savefig(plot_path)
    plt.show()

data = pd.read_excel('DemandForecasting/data/DemandData.xlsx')
data = datetimeFeatures(data)
data,scaler = dataNormalizer(data)
x,y = dataSequence(data,49,1,1)

#make the train, val and test data

x_train,x_val,x_test = x[:int(len(x)*0.8)],x[int(len(x)*0.8):int(len(x)*0.9)],x[int(len(x)*0.9):]
y_train,y_val,y_test = y[:int(len(x)*0.8)],y[int(len(x)*0.8):int(len(x)*0.9)],y[int(len(x)*0.9):]