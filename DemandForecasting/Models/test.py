import os
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score,mean_squared_error
from DataLoader import y_test,x_test,scaler
from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model('/Users/shalikanuwan/Documents/Academics/FYP/ForecastingModelDevelopment/demandModel1/ 10- 0.0017.ckpt')

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

#data visaulization

plt.figure(figsize=(15,5))
plt.plot(pred_ns,label = 'Predictions')
plt.plot(scaler.inverse_transform(y_test).flatten(),label = 'Actual')
plt.title('Model evaluaton for Solar Generation')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
os.makedirs('demandModel1', exist_ok=True)
plot_path = os.path.join('demandModel1', 'loss_plot.png')
plt.savefig(plot_path)
plt.savefig(plot_path)
plt.show()


