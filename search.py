from train import *
from itertools import product


def search(hyperparams, X, y, Xvalid, yvalid, nLabels):
    nHiddens, lrs, betas, lams = hyperparams
    params_list = list(product(nHiddens, lrs, lams, betas))
    best_params = params_list[0]
    min_err = 1
    for nHidden, lr, lam in params_list:
        print('\nstructure:', nHidden)
        print('learning rate:', lr)
        print('L2 rate:', lam)
        w, valid_err, _, _ = train(X, y, Xvalid, yvalid, nLabels, nHidden, lr, lam, es=True)
        print('valid_err:', np.min(valid_err))
        if np.min(valid_err) < min_err:
            min_err = np.min(valid_err)
            best_params = nHidden, lr, lam

    return best_params
