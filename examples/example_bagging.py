"""
Example of using FedHEONN method for a regression task.

In this example, the clients and coordinator are created on the same machine.
In a real environment, the clients and also the coordinator can be created on
different machines. In that case, some communication mechanism must be
established between the clients and the coordinator to send the computations
performed by the clients.
"""

# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only
import time
import tenseal as ts
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from algorithms.client.regressor import FedHEONN_regressor
from algorithms.coordinator.base import FedHEONN_coordinator
from metrics.general import Metrics

# Configuring the TenSEAL context for the CKKS encryption scheme
ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
ctx.generate_galois_keys()
ctx.global_scale = 2**40


# Function to create bootstrap samples
def bootstrap_sample(X, y):
    n_samples = X.shape[1]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[:, indices], y[indices, :]


# Bagging function
def bagging(X, y, estimators, cliente):
    Mc_lst, Usc_lst = [], []
    for _ in range(estimators):
        X_sample, y_sample = bootstrap_sample(X, y)
        cliente.fit(X_sample, y_sample)
        Mc, USc = client.get_param()
        Mc_lst.append(Mc)
        Usc_lst.append(USc)
    return Mc_lst, Usc_lst


# Predict vote average
def predict(array, n_out, n_est):
    subarrays = np.vsplit(array, array.shape[0] // n_out)
    vote_avg = np.sum(subarrays, axis=0) / n_est
    return vote_avg


# Bagging estimators
n_estimators = 10

n_outputs = 3

# Number of clients
n_clients = 20

# The data set is loaded (Carbon Nanotubes)
# Source: https://archive.ics.uci.edu/dataset/448/carbon+nanotubes
Data = pd.read_csv('../datasets/carbon_nanotubes.csv', delimiter=';')
Inputs = Data.iloc[:, :-3].to_numpy()
Targets = Data.iloc[:, -3:].to_numpy()  # 3 outputs to predict
train_X, test_X, train_t, test_t = train_test_split(Inputs, Targets, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = train_X.T
test_X = test_X.T

# Number of training and test data
n = len(train_t)
n_test = len(test_t)

# Number of outputs
_, n_outputs = train_t.shape

# Create a list of clients
clients = []
for i in range(0, n_clients):
    clients.append(FedHEONN_regressor(f='linear', context=ctx))

# Create the coordinator
coordinator = FedHEONN_coordinator(f='lin', lam=0.01)

start_time = time.time()
# Fit the clients with their local data
M = []
US = []
for i, client in enumerate(clients):
    rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
    print('Training client:', i + 1, 'of', n_clients, '(', min(rang), '-', max(rang), ')')
    lst_Mc, lst_USc = bagging(train_X[:, rang], train_t[rang, :], n_estimators, client)
    M.extend(lst_Mc)
    US.extend(lst_USc)

# The coordinator aggregates the information provided by the clients
# to obtain the weights of the collaborative model
# TODO: Agregar cada estimador por separado?
coordinator.aggregate(M, US)
end_time = time.time()

print(f"Tiempo transcurrido:  {end_time-start_time} (s)")

# Send the weights of the aggregate model, obtained by the coordinator,
# to all the clients
for client in clients:
    client.set_weights(coordinator.send_weights())

# BAGGING
test_y = clients[0].predict(test_X)
train_y = clients[0].predict(train_X)

test_y = predict(test_y, n_outputs, n_estimators)
train_y = predict(train_y, n_outputs, n_estimators)

# Predictions for the test set using one client
print("Test MSE: %0.8f" % (100 * mean_squared_error(test_t, test_y.T)))


params = Metrics.fill_params(x_train=train_X, x_test=test_X, d_train=train_t, d_test=test_t, y_train=train_y.T, y_test=test_y.T)
Metrics.run(params, 'carbon_nanotubes')
