import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import sys
import pdb

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

assert(len(sys.argv) >= 1)
folder = sys.argv[1]

f = open(folder + '/log.pkl', 'rb')
b = pl.load(f)
duration = b[0]
reward = b[1]

reward_per_episode = []
cur_ind = 0
l = len(duration)

x = np.arange(l)
itp = interp1d(x, duration, kind='linear')
itpr = interp1d(x, reward, kind='linear')

if l < 101:
    window_size = l / 2
else:
    window_size = 101
poly_order = 3
print(len(x))
# print(len(reward))
# print(len(duration))
# print(l)
duration_smooth = savgol_filter(itp(x), window_size, poly_order)
reward_per_episode_smooth = savgol_filter(itpr(x), window_size, poly_order)

fig, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(x, duration_smooth)
axarr[0].plot(x, duration, alpha=0.3)
axarr[0].grid(True)
axarr[0].set_ylabel('Duration')
axarr[0].set_xlim([0, l])

axarr[1].plot(x, reward_per_episode_smooth)
axarr[1].plot(x, reward, alpha=0.3)
axarr[1].grid(True)
axarr[1].set_ylabel('Reward per Episode')
axarr[1].set_xlabel('Episode')
axarr[1].set_xlim([0, l])

plt.show()
