import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

N = 10000
T = 500/N
abs = np.linspace(0, T*N, N)
w1 = 2
w2 = 4
w3 = 6
a1 = 1
a2 = 3
a3 = 5
s1 = [a1*np.sin(w1*x) for x in abs]
s2 = [a2*np.sin(w2*x) for x in abs]
s3 = [a3*np.sin(w3*x) for x in abs]
s = [a + b + c for a,b,c in zip(s1,s2,s3)]
yf = fft(np.array(s))
xf = fftfreq(N, T)[:N//2]

# Plot
#plt.figure(figsize=(5,10))
fig, axs = plt.subplots(1,5)
axs[0].plot(abs, s1, 'b')
axs[0].set_title('s1')
axs[0].set_ylim([-10, 10])
axs[0].set_xlim([0, 10])

axs[1].plot(abs, s2, 'r')
axs[1].set_title('s2')
axs[1].set_ylim([-10, 10])
axs[1].set_xlim([0, 10])

axs[2].plot(abs, s3, 'g')
axs[2].set_title('s3')
axs[2].set_ylim([-10, 10])
axs[2].set_xlim([0, 10])

axs[3].plot(abs, s)
axs[3].set_title('total')
axs[3].set_ylim([-10, 10])
axs[3].set_xlim([0, 10])

axs[4].plot(2*np.pi*xf, 2.0/N * np.abs(yf[0:N//2]))
axs[4].set_title('FFT')
axs[4].set_xlim([0, 10])

fig.set_figheight(5)
fig.set_figwidth(30)

fig.savefig('test.png')