import time
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from algorithms.client.regressor import FedHEONN_regressor
from algorithms.coordinator.base import FedHEONN_coordinator
from metrics.general import Metrics
from sklearn.datasets import make_regression


# Configuring the TenSEAL context for the CKKS encryption scheme
ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
ctx.generate_galois_keys()
ctx.global_scale = 2**40

# Bagging estimators
n_estimators = 50
# Number of clients
n_clients = 2

X, y = make_regression(n_samples=1000, n_features=2, noise=5, random_state=42, n_targets=1)
train_X, test_X, train_t, test_t = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# Number of training and test data
n = len(train_t)

# Create a list of clients
clients = []
for i in range(0, n_clients):
    clients.append(FedHEONN_regressor(f='linear', context=ctx, bagging=False, n_estimators=n_estimators, encrypted=False))

start_time = time.time()
# Fit the clients with their local data
M = []
US = []
for i, client in enumerate(clients):
    rang = range(int(i * n / n_clients), int(i * n / n_clients) + int(n / n_clients))
    print('Training client:', i + 1, 'of', n_clients, '(', min(rang), '-', max(rang), ')')
    client.fit(train_X[rang], train_t[rang])
    M_lst, US_lst = client.get_param()
    M.append(M_lst)
    US.append(US_lst)

# The coordinator aggregates the information provided by the clients
# to obtain the weights of the collaborative model

# Create the coordinator
coordinator = FedHEONN_coordinator(f='lin', lam=0.01, bagging=False, encrypted=False)
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

# Predictions for the test set using one client
print("Test MSE: %0.8f" % (100 * mean_squared_error(test_t, test_y.T)))

params = Metrics.fill_params(x_train=train_X, x_test=test_X, d_train=train_t, d_test=test_t, y_train=train_y.T, y_test=test_y.T)
Metrics.run(params, 'make_regression')
