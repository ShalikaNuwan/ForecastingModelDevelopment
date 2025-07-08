import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
    genScaler = MinMaxScaler()
    irrScaler = MinMaxScaler()
    df['Adjusted_Generation'] = genScaler.fit_transform(df[['Adjusted_Generation']])
    df['solarradiation'] = irrScaler.fit_transform(df[['solarradiation']])

    return df,genScaler

def dataSequence(df,window_size, step_size, forecast_horizon):
    generation = df['Adjusted_Generation'].values
    solar_irr = df['solarradiation'].values
    num_samples = len(generation)
    X = []
    y = []

    for start in range(0, num_samples - window_size - forecast_horizon + 1, step_size):
        end = start + window_size
        forecast_end = end + forecast_horizon
        window_dem = generation[start:end-1]
        window_irr = solar_irr[start:end-1]
        X.append([window_dem,window_irr])
        y.append(generation[end-1:forecast_end-1])  

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

data = pd.read_excel('SolarForecasting/Data/filterdSolarGenerationUOM.xlsx')
data,scaler = dataNormalizer(data)
x,y = dataSequence(data,49,1,1)

#make the train, val and test data

x_train,x_val,x_test = x[:int(len(x)*0.8)],x[int(len(x)*0.8):int(len(x)*0.9)],x[int(len(x)*0.9):]
y_train,y_val,y_test = y[:int(len(x)*0.8)],y[int(len(x)*0.8):int(len(x)*0.9)],y[int(len(x)*0.9):]


print(x_train.shape)