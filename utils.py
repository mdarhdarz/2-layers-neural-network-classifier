import numpy as np


def softmax(x):
    expx = np.exp(x)
    return expx / np.sum(expx)


def one_hot(ind, nLabels):
    n = len(ind)
    y = np.zeros((n, nLabels))

    for i in np.arange(n):
        y[i, ind[i]] = 1

    return y


def standardizeCols(M, mu=None, sigma2=None):
    if mu is None:
        mu = np.mean(M, axis=0)
    if sigma2 is None:
        sigma2 = np.std(M, axis=0)
        sigma2[sigma2 == 0] = 1

    S = (M - mu) / sigma2
    return S, mu, sigma2


def standardize(M):
    minimum = np.min(M)
    maximum = np.max(M)
    S = (M - minimum) / (maximum - minimum)
    return S


def predict(w, X, y, nHidden, nLabels):
    nInstances, nVars = X.shape

    # Form Weights
    offset = nVars * nHidden[0]
    inputWeights = w[0:offset].reshape(nHidden[0], nVars).T
    outputWeights = w[offset:offset + (nHidden[-1] + 1) * nLabels].reshape(nLabels, (nHidden[-1] + 1)).T

    # Compute Output
    ip = X @ inputWeights
    fp = np.tanh(ip)

    yhat = np.hstack((np.ones((nInstances, 1)), fp)) @ outputWeights
    prob = np.apply_along_axis(softmax, axis=1, arr=yhat)
    loss = -np.log(prob[np.arange(nInstances), y.reshape(-1)])
    loss = np.mean(loss)

    yhat = yhat.argmax(axis=1).reshape(-1, 1)
    return yhat, loss
