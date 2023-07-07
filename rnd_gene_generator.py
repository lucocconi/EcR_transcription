#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:53:58 2022

@author: cocconl
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import numba as nb
import pandas as pd

rnaseq_data = pd.read_csv("./genes_monotonic.csv")
rnaseq_data_unfiltered = pd.read_csv("./genes_changing_norm.csv")
rnaseq_data_mx = pd.read_csv("./all_RNA_seq_genes.csv")

# =============================================================================
# # Generate synthetic datasets from filtered data (monotonic genes)
# =============================================================================

nM0 = rnaseq_data['conc_0nM'].to_numpy()
nM20 = rnaseq_data['conc_20nM'].to_numpy()
nM200 = rnaseq_data['conc_200nM'].to_numpy()
nM2000 = rnaseq_data['conc_2000nM'].to_numpy()

m_nM0 = np.mean(nM0)
m_nM20 = np.mean(nM20)
m_nM200 = np.mean(nM200)
m_nM2000 = np.mean(nM2000)

s_nM0 = np.std(nM0)
s_nM20 = np.std(nM20)
s_nM200 = np.std(nM200)
s_nM2000 = np.std(nM2000)

m_overall = np.mean(np.concatenate((nM0,nM20,nM200,nM2000)))
s_overall = np.std(np.concatenate((nM0,nM20,nM200,nM2000)))


N = 50000

##First random dataset: use overall mean and std only (1000 genes)
synth_data_1 = rnd.normal(m_overall, s_overall,(N,4))

for ln in range(N):
    synth_data_1[ln,:] /= np.max(synth_data_1[ln,:])

##First random dataset: use mean and std for each 20E level
synth_data_2 = np.zeros((N,4))
synth_data_2[:,0] = rnd.normal(m_nM0,s_nM0,N)
synth_data_2[:,1] = rnd.normal(m_nM20,s_nM20,N)
synth_data_2[:,2] = rnd.normal(m_nM200,s_nM200,N)
synth_data_2[:,3] = rnd.normal(m_nM2000,s_nM2000,N)

for ln in range(N):
    synth_data_2[ln,:] /= np.max(synth_data_2[ln,:])

np.savetxt("synthgenes_overallmean.csv", synth_data_1, delimiter=",")
np.savetxt("synthgenes_perlevelmean.csv", synth_data_2, delimiter=",")

# =============================================================================
# # Generate synthetic datasets from unfiltered data
# =============================================================================

nM0_all = rnaseq_data_unfiltered['conc_0nM'].to_numpy()
nM20_all = rnaseq_data_unfiltered['conc_20nM'].to_numpy()
nM200_all = rnaseq_data_unfiltered['conc_200nM'].to_numpy()
nM2000_all = rnaseq_data_unfiltered['conc_2000nM'].to_numpy()

m_nM0_all = np.mean(nM0_all)
m_nM20_all = np.mean(nM20_all)
m_nM200_all = np.mean(nM200_all)
m_nM2000_all = np.mean(nM2000_all)

s_nM0_all = np.std(nM0_all)
s_nM20_all = np.std(nM20_all)
s_nM200_all = np.std(nM200_all)
s_nM2000_all = np.std(nM2000_all)

m_overall_all = np.mean(np.concatenate((nM0_all,nM20_all,nM200_all,nM2000_all)))
s_overall_all = np.std(np.concatenate((nM0_all,nM20_all,nM200_all,nM2000_all)))


N =50000

##First random dataset: use overall mean and std only (1000 genes)
synth_data_1_all = rnd.normal(m_overall_all, s_overall_all,(N,4))

for ln in range(N):
    synth_data_1_all[ln,:] /= np.max(synth_data_1_all[ln,:])

##First random dataset: use mean and std for each 20E level
synth_data_2_all = np.zeros((N,4))
synth_data_2_all[:,0] = rnd.normal(m_nM0_all,s_nM0_all,N)
synth_data_2_all[:,1] = rnd.normal(m_nM20_all,s_nM20_all,N)
synth_data_2_all[:,2] = rnd.normal(m_nM200_all,s_nM200_all,N)
synth_data_2_all[:,3] = rnd.normal(m_nM2000_all,s_nM2000_all,N)

for ln in range(N):
    synth_data_2_all[ln,:] /= np.max(synth_data_2_all[ln,:])

np.savetxt("synthgenes_overallmean_unfiltered.csv", synth_data_1_all, delimiter=",")
np.savetxt("synthgenes_perlevelmean_unfiltered.csv", synth_data_2_all, delimiter=",")

# =============================================================================
# # Generate synthetic datasets from unfiltered data (including 20E "independent")
# =============================================================================

nM0_mx = rnaseq_data_mx['conc_0nM'].to_numpy()
nM20_mx = rnaseq_data_mx['conc_20nM'].to_numpy()
nM200_mx = rnaseq_data_mx['conc_200nM'].to_numpy()
nM2000_mx = rnaseq_data_mx['conc_2000nM'].to_numpy()

m_nM0_mx = np.mean(nM0_mx)
m_nM20_mx = np.mean(nM20_mx)
m_nM200_mx = np.mean(nM200_mx)
m_nM2000_mx = np.mean(nM2000_mx)

s_nM0_mx = np.std(nM0_mx)
s_nM20_mx = np.std(nM20_mx)
s_nM200_mx = np.std(nM200_mx)
s_nM2000_mx = np.std(nM2000_mx)

m_overall_mx = np.mean(np.concatenate((nM0_mx,nM20_mx,nM200_mx,nM2000_mx)))
s_overall_mx = np.std(np.concatenate((nM0_mx,nM20_mx,nM200_mx,nM2000_mx)))


N =50000

##First random dataset: use overall mean and std only (1000 genes)
synth_data_1_mx = rnd.normal(m_overall_mx, s_overall_mx,(N,4))

for ln in range(N):
    synth_data_1_mx[ln,:] /= np.max(synth_data_1_mx[ln,:])

##First random dataset: use mean and std for each 20E level
synth_data_2_mx = np.zeros((N,4))
synth_data_2_mx[:,0] = rnd.normal(m_nM0_mx,s_nM0_mx,N)
synth_data_2_mx[:,1] = rnd.normal(m_nM20_mx,s_nM20_mx,N)
synth_data_2_mx[:,2] = rnd.normal(m_nM200_mx,s_nM200_mx,N)
synth_data_2_mx[:,3] = rnd.normal(m_nM2000_mx,s_nM2000_mx,N)

for ln in range(N):
    synth_data_2_mx[ln,:] /= np.max(synth_data_2_mx[ln,:])

np.savetxt("synthgenes_overallmean_alldata.csv", synth_data_1_mx, delimiter=",")
np.savetxt("synthgenes_perlevelmean_alldata.csv", synth_data_2_mx, delimiter=",")
