import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tenseal as ts
from algorithms.client.classifier import FedHEONN_classifier
from algorithms.coordinator.base import FedHEONN_coordinator
from metrics.general import Metrics

ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
ctx.generate_galois_keys()
ctx.global_scale = 2**40

# Number of clients
n_clients = 50
n_estimators = 50

# IID or non-IID scenario (True or False)
iid = False

X, y = make_classification(n_samples=1000, random_state=42)
train_X, test_X, train_t, test_t = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# Number of training and test data
n = len(train_t)

# Non-IID option: Sort training data by class
if not iid:
    ind = np.argsort(train_t)
    train_t = train_t[ind]
    train_X = train_X[ind]
    print('non-IID scenario')
else:        
    ind_list = list(range(n))
    np.random.shuffle(ind_list) # Data are shuffled in case they come ordered by class
    train_X  = train_X[ind_list]
    train_t = train_t[ind_list]
    print('IID scenario')
    
# Number of classes
n_classes = len(np.unique(train_t))

# One hot encoding for the targets
t_onehot = np.zeros((n, n_classes))
for i, value in enumerate(train_t):
    t_onehot[i, value] = 1

# Create a list of clients
clients = []
for i in range(0, n_clients):
    clients.append(FedHEONN_classifier(f='logs', context=ctx, encrypted=True, bagging=True, n_estimators=n_estimators))

# Fit the clients with their local data    
M  = []
US = []
for i, client in enumerate(clients):
    rang = range(int(i * n/n_clients), int(i * n/n_clients) + int(n / n_clients))
    print('Training client:', i+1, 'of', n_clients, '(', min(rang), '-', max(rang), ') - Classes:', np.unique(train_t[rang]))
    client.fit(train_X[rang], t_onehot[rang])
    M_c, US_c = client.get_param()
    M.append(M_c)    
    US.append(US_c)        

# Create the coordinator
coordinator = FedHEONN_coordinator(f='logs', lam=0.01, encrypted=True, bagging=True)

# The coordinator aggregates the information provided by the clients
# to obtain the weights of the collaborative model
coordinator.aggregate(M, US)

# Send the weights of the aggregate model, obtained by the coordinator,
# to all the clients
for client in clients:
    client.set_weights(coordinator.send_weights())

# Predictions for the test set using one client    
test_y = clients[0].predict(test_X)
train_y = clients[0].predict(train_X)

params = Metrics.fill_params(x_train=train_X, x_test=test_X, d_train=train_t, d_test=test_t,
                             y_train=train_y, y_test=test_y, classification=True, n_classes=n_classes)
Metrics.run(params, 'make_classification')
