from sklearn.model_selection import train_test_split
from search import *

np.random.seed(666666)
Xtrain = np.load('X_train.npy')
ytrain = np.load('y_train.npy')
Xtest = np.load('X_test.npy')
ytest = np.load('y_test.npy')
X, Xvalid, y, yvalid = train_test_split(Xtrain, ytrain, test_size=1/12)

# standardize
X, mu, sigma = standardizeCols(X)

n, d = X.shape
nLabels = np.max(y) + 1
y_oh = one_hot(y, nLabels)
t1 = Xvalid.shape[0]
t2 = Xtest.shape[0]
X = np.hstack((np.ones((n, 1)), X))
d = d + 1

# apply transformation to the validation / test data
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xvalid = np.hstack((np.ones((t1, 1)), Xvalid))
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)
Xtest = np.hstack((np.ones((t2, 1)), Xtest))

# structure & hyperparameters
nHiddens = [[64], [256]]
lrs = [1e-2, 1e-3, 1e-4]
lams = [1e-4, 1e-5, 1e-6]
hyperparams = nHiddens, lrs, lams

# search
# best_params = search(hyperparams, X, y_oh, Xvalid, yvalid, nLabels)
# nHidden, lr, lam = best_params
nHidden, lr, lam = [256], 1e-2, 1e-6
print('\nbest_params:')
print('structure:', nHidden)
print('learning rate:', lr)
print('L2 rate:', lam)

# train
print('\nstart training')
w_best, valid_err, valid_loss, train_loss = train(X, y_oh, Xvalid, yvalid, nLabels, nHidden, lr, lam, maxIter=880000)

# evaluate test error
yhat, _ = predict(w_best, Xtest, ytest, nHidden, nLabels)
print('Test error with final model = ', np.sum(yhat != ytest) / t2)
np.save('weights.npy', w_best)
np.save('structure.npy', nHidden)
np.save('valid_err.npy', valid_err)
np.save('valid_loss.npy', valid_loss)
np.save('train_loss.npy', train_loss)
