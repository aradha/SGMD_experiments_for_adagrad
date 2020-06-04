import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

#"""
losses = pickle.load(open('losses/sinx_adaptive_rate_losses.p', 'rb'))
losses = [np.log(l) for l in losses]
plt.plot(list(range(1, len(losses) + 1)), losses, '-g',
         label='Our Step Size (Adaptive)')

losses = pickle.load(open('losses/sinx_train_loss.p', 'rb'))
losses = [np.log(l) for l in losses]
plt.plot(list(range(1, len(losses) + 1)), losses, '-r',
         label='Step Size 0.1')

plt.legend()
plt.savefig('training_loss.pdf')
#"""

"""
losses = pickle.load(open('train_loss.p', 'rb'))
losses = [np.log(l) for l in losses]
plt.plot(list(range(1, len(losses) + 1)), losses, '-r',
         label='Step Size 0.1')
#plt.plot(list(range(1, len(losses) + 1)), losses, '-g',
#         label='Our Step Size (Adaptive)')

plt.legend()
plt.savefig('training_loss.pdf')
#"""
