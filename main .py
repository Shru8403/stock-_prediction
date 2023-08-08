#Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as matp

import torch
import torch.nn as nn

data_frame = pd.read_csv("data.csv.csv")
close_p = data_frame["Close"]

# Sequence length = 15
minmax_var = MinMaxScaler()
scale_p = minmax_var.fit_transform(np.array(close_p)[... , None]).squeeze()

X = []
y = []

for i in range(len(scale_p) - 15):
    X.append(scale_p[i : i + 15])
    y.append(scale_p[i + 15])

X = np.array(X)[... , None]
y = np.array(y)[... , None]

training_data_x = torch.from_numpy(X[:int(0.8 * X.shape[0])]).float()
training_data_y = torch.from_numpy(y[:int(0.8 * X.shape[0])]).float()
testing_data_x = torch.from_numpy(X[int(0.8 * X.shape[0]):]).float()
testing_data_y = torch.from_numpy(y[int(0.8 * X.shape[0]):]).float()

class Model(nn.Module):
    def __init__(self , input_size , hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size , batch_first = True)
        self.fc = nn.Linear(hidden_size , 1)
    def forward(self , x):
        output , (hidden , cell) = self.lstm(x)
        return self.fc(hidden[-1 , :])
model = Model(1 , 64)

opt = torch.optim.Adam(model.parameters() , lr = 0.001)
loss_fn = nn.MSELoss()

#number of epochs = 150
for epoch in range(150):
    output = model(training_data_x)
    loss = loss_fn(output , training_data_y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 10 == 0 and epoch != 0:
        print(epoch , "epoch loss" , loss.detach().numpy())

model.eval()
with torch.no_grad():
    output = model(testing_data_x)

prediction = minmax_var.inverse_transform(output.numpy())
real = minmax_var.inverse_transform(testing_data_y.numpy())

matp.plot(prediction.squeeze() , color = "blue" , label = "predicted")
matp.plot(real.squeeze() , color = "red" , label = "real")
matp.show()
