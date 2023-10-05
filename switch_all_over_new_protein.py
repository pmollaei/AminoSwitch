import mdtraj as md
import numpy as np
import pandas as pd
from itertools import combinations
from itertools import groupby
import itertools as it
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from itertools import groupby

prot = md.load('fs-peptide.pdb')
top = prot.topology

selected_atm = [('O', 'C', 'CA', 'N', 'CB', 'CG', 'NE2'),('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD', 'NE', 'NH1', 'NH2'), ('O', 'C', 'CA', 'CB', 'N'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'OE2', 'CD', 'OE1'), ('O', 'C', 'CA', 'N'), ('O', 'C', 'CA', 'N', 'CG2', 'CB', 'OG1'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CZ'), ('O', 'C', 'CA', 'N', 'CB', 'OG'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'OD1', 'OD2'), ('O', 'C', 'CA', 'N', 'CB', 'CG1', 'CG2'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'OH'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD1', 'CD2'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD', 'OE1', 'NE2'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD', 'CE', 'NZ'), ('O', 'C', 'CA', 'N', 'CB', 'CG1', 'CG2', 'CD1'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD1', 'CE2', 'CH2'), ('O', 'C', 'CA', 'N', 'CB', 'SG'), ('O', 'C', 'CA', 'CG'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'ND2', 'OD1'), ('O', 'C', 'CA', 'N', 'CB', 'CG', 'SD', 'CE')]
residues = ['HIS','ARG','ALA', 'GLU', 'GLY', 'THR', 'PHE', 'SER', 'ASP', 'VAL', 'TYR', 'LEU', 'GLN', 'LYS', 'ILE', 'TRP', 'CYS', 'PRO', 'ASN', 'MET']

#combinations of three atoms within each residue
cmb_atm = []
for i in range(len(residues)):
    c_atm = (residues[i], list(it.combinations(selected_atm[i], 3)))
    cmb_atm.append(c_atm)
    
# residue-atom (i.e HIS7-N) getting from .pdb files
residue_atom = []
for res in top.atoms:
    residue_atom.append(str(res))

# groups of unique residues
res = [list(i) for j, i in groupby(residue_atom, lambda a: a.split('-')[0])]
unq_res = []
for r in range(len(res)):
    unq_res.append(res[r][0].split('-')[0])
    
angle_ind = []
for i in range(len(unq_res)):
    b = []
    for j in range(len(cmb_atm)):
        if unq_res[i][:3]==cmb_atm[j][0]:
            c = []
            for k in range(len(cmb_atm[j][1])):
                a = (unq_res[i]+'-'+cmb_atm[j][1][k][0], unq_res[i]+'-'+cmb_atm[j][1][k][1], unq_res[i]+'-'+cmb_atm[j][1][k][2])
                d = []
                for l in range(len(residue_atom)):
                    if a[0]==residue_atom[l]:
                        f1 = l
                    if a[1]==residue_atom[l]:
                        f2 = l
                    if a[2]==residue_atom[l]:
                        f3 = l
                t = ((f1, f2, f3), a)
                c.append(t)
            b.append(c)
    if len(b)!=0:
        angle_ind.append(b)


# Switches for single Trajectory

angles = []
traj = md.load_dcd('fs_peptide15.dcd', top='fs-peptide.pdb')
for i in range(len(angle_ind)):
    t = []
    for j in range(len(angle_ind[i][0])):
        ind = [int(angle_ind[i][0][j][0][0]), int(angle_ind[i][0][j][0][1]), int(angle_ind[i][0][j][0][2])]
        angl = md.compute_angles(traj, [ind], periodic=True, opt=True)
        t.append([item for sublist in angl for item in sublist])
    angles.append(t)


dataset = pd.read_csv('training_dataset_bimodal_switches.csv')

# Angle featurization

ml_result = []
for l in range(len(angles)):
    f = []
    for t in range(len(angles[l])):
        x_dif_max = []
        maxr = []
        maxl = []
        dif_max = []
        meanr = []
        meanl = []
        meanm = []
        rm = []
        hl = []
        hr = []
        dif_meanlm = []
        dif_meanrm = []
        dif_meanrl = []
        hi = []
        mid_hist = []
        hist,bin_edges = np.histogram(angles[l][t], density=True, bins=150)
        hi.append(hist)
        cluster_id = KMeans(2).fit_predict(bin_edges[1:].reshape(-1, 1))
        x_std = (cluster_id - cluster_id.mean()) / cluster_id.std()
        step_indicator = 1*np.cumsum(x_std)
        a = step_indicator[0:len(step_indicator)-1]
        b = step_indicator[1:]
        m = np.diff(a)/np.diff(b)  
        z = np.where(m !=1)
        rounded = [np.round(x) for x in m]

        no1 = [i for i, e in enumerate(rounded) if e != 1]

        maxr_ = hist[no1[0]:].max()
        maxr.append(maxr_)
        lb_maxr_ = [i for i, e in enumerate(hist) if e == maxr_]
        x_maxr_ = bin_edges[lb_maxr_[0]]

        maxl_ = hist[:no1[0]].max()
        maxl.append(maxl_)
        lb_maxl_ = [i for i, e in enumerate(hist) if e == maxl_]
        x_maxl_ = bin_edges[lb_maxl_[0]]

        dif_max_ = abs(maxr_ - maxl_)
        dif_max.append(dif_max_)

        x_dif_max_ = abs(x_maxr_- x_maxl_)
        x_dif_max.append(x_dif_max_)

        mid_hist_ = hist[int(x_dif_max_/2)]
        mid_hist.append(mid_hist_)

        min_ = min(maxl_,maxr_)
        max_ = max(maxl_,maxr_)

        ratio_ = min_/max_
        rm.append(ratio_)

        hl_ = (maxl_-hist[no1[0]])/max_
        hl.append(hl_)
        hr_ = (maxr_-hist[no1[0]])/max_
        hr.append(hr_)

        meanl_ = hist[:no1[0]].mean()
        meanl.append(meanl_)
        meanr_ = hist[no1[0]:].mean()
        meanr.append(meanr_)
        meanm_ = hist[no1[0]-1:no1[0]+1].mean()
        meanm.append(meanm_)

        dif_meanlm_ = abs(meanl_ - meanm_)
        dif_meanlm.append(dif_meanlm_)
        dif_meanrm_ = abs(meanr_ - meanm_)
        dif_meanrm.append(dif_meanrm_)
        dif_meanrl_ = abs(meanr_ - meanl_)
        dif_meanrl.append(dif_meanrl_)

        d1 = {'maxr': maxr, 'maxl': maxl, 'dif_max': dif_max, 'meanl': meanl, 'meanr': meanr, 'meanm': meanm, 'rm': rm, 'hl': hl, 'hr': hr, 'dif_meanlm': dif_meanlm, 'dif_meanrm': dif_meanrm, 'dif_meanrl': dif_meanrl, 'x_dif_max':x_dif_max, 'mid_hist':mid_hist}
        testset = pd.DataFrame(d1)
        trainset = dataset.drop(columns=['label'])
        trainlabel = dataset.iloc[: , -1]
        forest = RandomForestClassifier()
        forest.fit(trainset, trainlabel)
        pred = forest.predict(testset)
        f.append(pred[0])
    ml_result.append(f)
    if len(ml_result) % 10 == 0:
        print(len(ml_result))


# datapoints of all switch residues

swt_ind = []
for i in range(len(ml_result)):
    for j in range(len(ml_result[i])):
        if ml_result[i][j]==1:
            a = (i,j,angle_ind[i][0][j][1],angles[i][j])
            swt_ind.append(a)

df = pd.DataFrame(swt_ind)
df.columns = ['resid', 'angle_ndx', 'atom_ndx', 'angle_values']


# list of switch residues
unq_swt = []
for item in swt_ind:
    a = item[2][0].split('-')[0]
    if a not in unq_swt:
        unq_swt.append(a)


# ANSR

over = []
for i in range(len(cmb_atm)):
    a = (cmb_atm[i][0], len(cmb_atm[i][1]))
    over.append(a)

up = []
for i in range(len(df)):
    a = df.iloc[i][2][0].split("-")[0]
    up.append(a)

rep = [list(g) for _, g in groupby(up, lambda l: l)]

residue_ANSR = []
for i in range(len(rep)):
    a = len(rep[i])
    for j in range(len(over)):
        if rep[i][0][:3] == over[j][0]:
            b = str(a)+"/"+str(over[j][1])
            c = (rep[i][0], b)
    residue_ANSR.append(c)

print("residue, ANSR : ", residue_ANSR)


# ATSC

t = []
for i in range(len(df)):
    a = df.iloc[i, 2][0].split('-')[0]
    b = (df.iloc[i, 2][0].split('-')[1], df.iloc[i, 2][1].split('-')[1], df.iloc[i, 2][2].split('-')[1])
    c = (a, b)
    t.append(c)

gp = [list(g) for _, g in groupby(t, lambda l: l[0])]

tot = []
for i in range(len(gp)):
    b = []
    for j in range(len(gp[i])):
        b.append(gp[i][j][1])
    c = (gp[i][j][0], np.reshape(b, -1))
    tot.append(c)

residues_ATSC = []
for i in range(len(tot)):
    res = {key : [key] * val for key, val in Counter(tot[i][1]).items()}
    d = []
    for j in range(len(list(res.keys()))):
        a = (list(res.keys())[j])
        b = len(list(res.values())[j])
        c = (a,b)
        d.append(c)
    d.sort(key = lambda x: x[1], reverse = True)
    e = (tot[i][0], d)
    residues_ATSC.append(e)

print("residue, ATSC : ", residues_ATSC)

