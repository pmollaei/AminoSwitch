import mdtraj as md
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import groupby
import numpy
import glob
import warnings
warnings.filterwarnings("ignore")

results = []
for name in glob.glob("./*.csv"):
    print(name)
    plt.figure(figsize=(4,3))
    swt = pd.read_csv(name)
    e = pd.DataFrame(eval(swt.iloc[0,2])[:110000])
    k = numpy.array(e.iloc[:,0])
    clus = KMeans(2).fit_predict(k.reshape(-1, 1))
    grp = [list(j) for i, j in groupby(clus)]
    
    clusters_grp = []
    for i in range(len(grp)):
        if len(grp[i])>1:
            clusters_grp.append(grp[i])
    
    a = ((len(clusters_grp)/len(k))*100)
    b = (str(name)[1:-4], k, a, 0)
    results.append(b)
    plt.scatter(range(len(k)), k, s = 0.01)
    plt.title(a)
    plt.show()

