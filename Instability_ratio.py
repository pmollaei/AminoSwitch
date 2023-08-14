from sklearn.cluster import KMeans
from itertools import groupby

data = 'angle dataset'
clustering = KMeans(2).fit_predict(data.reshape(-1, 1))
grouping = [list(j) for i, j in groupby(clustering)]
# skip clusters with only one datapoint
clusters_grp = []
for i in range(len(grouping)):
    if len(grouping[i])>1:
        clusters_grp.append(grouping[i])

instability_ratio = ((len(clusters_grp)/len(data))*100)

