import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

t = list(range(100))

gsa = open('GSA_precisions','r')

gsa_precisions = []
for line in gsa.readlines():
    gsa_precisions.append(float(line))

tfidf = open('TFIDF_precisions','r')

tfidf_precisions = []
for line in tfidf.readlines():
    tfidf_precisions.append(float(line))

prf = open('PRF_precisions','r')

prf_precisions = []
for line in prf.readlines():
    prf_precisions.append(float(line))

reconstruction = open('prf_gsa_precisions','r')

rec_precisions = []
for line in reconstruction.readlines():
    rec_precisions.append(float(line))

lower = 1
upper = 100
plt.title("20-newsgroups evaluation")
plt.xlabel('Number of documents retrieved')
plt.ylabel('Precision (%)')
plt.plot(t[lower:upper], tfidf_precisions[lower:upper], 'y-',label='TF-IDF')
plt.plot(t[lower:upper], prf_precisions[lower:upper], 'b:', label='PRF')
plt.plot(t[lower:upper], gsa_precisions[lower:upper], 'g--',label='GSA')
plt.plot(t[lower:upper],rec_precisions[lower:upper], 'r-.',label='PRF+GSA')
plt.legend()

plt.grid()
plt.show(block=True)