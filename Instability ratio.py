#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[9]:


data = 'angle dataset'
clustering = KMeans(2).fit_predict(data.reshape(-1, 1))
grouping = [list(j) for i, j in groupby(clustering)]
# skip clusters with only one datapoint
clusters_grp = []
for i in range(len(grouping)):
    if len(grouping[i])>1:
        clusters_grp.append(grouping[i])

instability_ratio = ((len(clusters_grp)/len(data))*100)

