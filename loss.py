from utils import *


def loss_backward(w, X, y, nHidden, nLabels):
    nInstances, nVars = X.shape

    # Form Weights
    offset = nVars * nHidden[0]
    inputWeights = w[0:offset].reshape(nHidden[0], nVars).T
    outputWeights = w[offset:offset + (nHidden[-1] + 1) * nLabels].reshape(nLabels, (nHidden[-1] + 1)).T

    # Form Grad
    f = 0
    gInput = np.zeros_like(inputWeights)
    gOutput = np.zeros_like(outputWeights)

    # Forward
    ip = [X @ inputWeights]
    fp = [np.tanh(ip[0])]
    yhat = np.hstack((np.ones((nInstances, 1)), fp[-1])) @ outputWeights

    # Backward
    for i in np.arange(nInstances):
        # softmax
        prob = softmax(yhat[i].reshape(1, -1))
        label = y[i, :].argmax()

        # Output Weights
        relativeErr = -np.log(prob[0, label])
        f += relativeErr
        err = prob - y[i, :]
        gOutput[1:] = gOutput[1:] + fp[-1][[i]].T @ err
        gOutput[0] = gOutput[0] + err

        # Input Weights
        backprop = err @ outputWeights[1:].T * (1 - fp[0][[i]] ** 2)
        gInput = gInput + X[i].reshape(1, -1).T @ backprop

    # Put Gradient into vector
    g = np.zeros_like(w)
    offset = nVars * nHidden[0]
    g[0:offset] = gInput.flatten('F').reshape(-1, 1)
    g[offset:offset + (nHidden[-1] + 1) * nLabels] = gOutput.flatten('F').reshape(-1, 1)

    return f, g
