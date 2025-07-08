import os
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score,mean_squared_error
from DataLoader import y_test,x_test,scaler
from keras.models import load_model
import matplotlib.pyplot as plt
import random

model = load_model('solarModel2/ 24- 0.0019.ckpt')

pred = model.predict(x_test).flatten()
pred_ns = scaler.inverse_transform([pred]).flatten()

mae = mean_absolute_error(pred_ns,scaler.inverse_transform(y_test).flatten())
mse = mean_squared_error(pred_ns,scaler.inverse_transform(y_test).flatten())
mape = mean_absolute_percentage_error(pred_ns,scaler.inverse_transform(y_test).flatten())
r2 = r2_score(pred_ns,scaler.inverse_transform(y_test).flatten())

print("MSE :" + str(mse))
print("MAE :" + str(mae))
print("R2 :" + str(r2))
print("MAPE :" + str(mape))

current_file_dir = os.path.dirname(os.path.abspath(__file__))

# data visaulization
for i in range(5):
    idx = random.randint(0,3460)
    plt.figure(figsize=(15,5))
    plt.plot(pred_ns[idx:idx+24],label = 'Predictions')
    plt.plot(scaler.inverse_transform(y_test).flatten()[idx:idx+24],label = 'Actual')
    plt.title('Model evaluaton for Solar forecasting')
    plt.legend()
    plot_path = os.path.join(current_file_dir, f'plots/solarmodel1')
    os.makedirs(plot_path, exist_ok=True)
    plot_path = os.path.join(current_file_dir, f'plots/solarmodel1/loss_plot{i}.png')
    plt.savefig(plot_path)
    plt.savefig(plot_path)
    plt.show()


