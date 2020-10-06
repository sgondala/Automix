import matplotlib.pyplot as plt
import numpy as np

# Vaules for knn, mu
loss = [2.136,2.25,2.332,2.402,2.467,2.604,2.804,2.836]
accuracy = [0.573,0.558,0.566,0.574,0.573,0.57,0.58,0.565]

# Values for layers
loss = [2.487,2.577,2.655,2.748,2.86,2.969,3.111,3.226]
accuracy = [0.5855,0.586,0.596,0.57,0.5765,0.5775,0.5595,0.5655]


loss = np.array(loss)
accuracy = np.array(accuracy)

# loss = loss - np.min(loss)
# loss = loss/np.max(loss)

# accuracy = accuracy - np.min(accuracy)
# accuracy = accuracy/np.max(accuracy)

print(np.corrcoef(loss, accuracy))
# plt.plot(list(range(len(loss))), loss)
# plt.plot(list(range(len(accuracy))), accuracy)
# plt.savefig('loss_plot.png')