from loss import *


def train(X, y, Xvalid, yvalid, nLabels, nHidden, lr=1e-3, lam=1e-5, maxIter=660000, es=False):
    n, d = X.shape
    nvalid = Xvalid.shape[0]
    label = y.argmax(axis=1).reshape(-1, 1)

    # initialize weights
    nParams = d * nHidden[0]
    nParams = nParams + nHidden[-1] * nLabels + nLabels
    w = np.random.normal(0, 0.01, (nParams, 1))
    w_best = w

    # set valid param
    check_step = 1000
    check_points = int(maxIter / check_step)
    valid_err = np.ones(check_points)
    valid_loss = np.ones(check_points)
    train_err = np.ones(check_points)
    train_loss = np.ones(check_points)
    min_valid_ind = 0
    ind = 0
    backward = lambda w, i: loss_backward(w, X[i].reshape(1, -1), y[i].reshape(1, -1), nHidden, nLabels)
    # train
    for iter in np.arange(maxIter):
        if np.mod(iter + 1, check_step) == 0:
            lr = 0.999 * lr
            vyhat, vloss = predict(w, Xvalid, yvalid, nHidden, nLabels)
            tyhat, tloss = predict(w, X, label, nHidden, nLabels)
            ind = int((iter + 1) / check_step) - 1
            valid_err[ind] = np.sum(vyhat != yvalid) / nvalid
            train_err[ind] = np.sum(tyhat != label) / n
            valid_loss[ind] = vloss
            train_loss[ind] = tloss
            print('Training iteration =', iter + 1)
            print('training error   =', np.round(train_err[ind], 3), ', training loss   =', np.round(tloss, 3))
            print('validation error =', np.round(valid_err[ind], 3), ', validation loss =', np.round(vloss, 3))
            if valid_err[ind] <= np.min(valid_err):
                min_valid_ind = ind
                w_best = w
            # early stop
            if es:
                if valid_err[ind] > 0.85 and iter > maxIter / 10:
                    print("may not converge, break")
                    break
                if ind - min_valid_ind > 100:
                    print('early stop!')
                    break

        i = int(np.random.rand() * n)
        _, g = backward(w, i)
        w = w - lr * g - lam * w

    return w_best, valid_err[0:ind], valid_loss[0:ind], train_loss[0:ind]
