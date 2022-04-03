import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import standardize


weights = np.load('weights.npy')
nHidden = np.load('structure.npy')
valid_err = np.load('valid_err.npy')
valid_loss = np.load('valid_loss.npy')
train_loss = np.load('train_loss.npy')

# loss curve
fig = plt.figure()
ax1 = fig.add_subplot(121, title='loss')
ax1.plot(train_loss, label='train loss')
ax1.plot(valid_loss, label='validation loss')
ax1.legend()
plt.xlabel('iterations(x1000)')

ax2 = fig.add_subplot(122, title='validation accuracy')
ax2.plot(1-valid_err)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.xlabel('iterations(x1000)')
plt.savefig('loss curve.jpg')
plt.show()


# parameters
# Form Weights
nVars = 784+1
nLabels = 10
offset = nVars * nHidden[0]
inputWeights = weights[0:offset].reshape(nHidden[0], nVars).T[1:]
outputWeights = weights[offset:offset + (nHidden[-1] + 1) * nLabels].reshape(nLabels, (nHidden[-1] + 1)).T[1:]
# pca
pca = PCA(n_components=3)
inputWeights_pca = standardize(pca.fit_transform(inputWeights))
outputWeights_pca = standardize(pca.fit_transform(outputWeights))
# plot
img_input = inputWeights_pca.reshape(28, 28, 3)
img_output = outputWeights_pca.reshape(16, 16, 3)
plt.imshow(img_input)
plt.title('input weights')
plt.savefig('input.jpg')
plt.show()
plt.imshow(img_output)
plt.title('output weights')
plt.savefig('output.jpg')
plt.show()
