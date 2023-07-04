#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:24:10 2022
@author: cocconl

CODE USED TO GENERATE SOME OF THE MODELLING FIGURES IN THE PAPER
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import minimize


plt.close('all')
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:green", "lime", "tab:red","orange","purple"]) 


# =============================================================================
# # IMPORT REPORTER EXPRESSION DATA (and prepare for analysis)
# =============================================================================

data_10ere     = pd.read_csv("./10ere.csv")
data_10ere_Enh = pd.read_csv("./gbe.csv")
data_10ere_Sil = pd.read_csv("./brk_subtr.csv")

#break down data for enhancer case into arrys
data_10ere_Enh_means = data_10ere_Enh.groupby(['Genotype','Condition']).mean().reset_index()
data_10ere_Enh_stds  = data_10ere_Enh.groupby(['Genotype','Condition']).std().reset_index()

dem_wt = data_10ere_Enh_means.loc[data_10ere_Enh_means['Genotype'] == 'gbe10ere']
des_wt = data_10ere_Enh_stds.loc[data_10ere_Enh_stds['Genotype'] == 'gbe10ere']

dem_mut = data_10ere_Enh_means.loc[data_10ere_Enh_means['Genotype'] == 'gbemut']
des_mut = data_10ere_Enh_stds.loc[data_10ere_Enh_stds['Genotype'] == 'gbemut']

#break down data for silencer case into arrys
data_10ere_Sil_means = data_10ere_Sil.groupby(['Genotype','Condition']).mean().reset_index()
data_10ere_Sil_stds  = data_10ere_Sil.groupby(['Genotype','Condition']).std().reset_index()

dsm_wt = data_10ere_Sil_means.loc[data_10ere_Sil_means['Genotype'] == 'ere']
dss_wt = data_10ere_Sil_stds.loc[data_10ere_Sil_stds['Genotype'] == 'ere']

dsm_mut = data_10ere_Sil_means.loc[data_10ere_Sil_means['Genotype'] == 'mut']
dss_mut = data_10ere_Sil_stds.loc[data_10ere_Sil_stds['Genotype'] == 'mut']

#break down data for neutral case into arrys
data_10ere_means = data_10ere.groupby(['Genotype','Condition']).mean().reset_index()
data_10ere_stds  = data_10ere.groupby(['Genotype','Condition']).std().reset_index()

dnm_wt = data_10ere_means.loc[data_10ere_Sil_means['Genotype'] == 'ere']
dns_wt = data_10ere_stds.loc[data_10ere_Sil_stds['Genotype'] == 'ere']

#normalise 10xERE signal (which was measured on a different day) based on latest data at 200nM
scalefact = 0.7159 * dem_wt['MeanValue'][2] / dnm_wt['MeanValue'][2]
dnm_wt['MeanValue'] *= scalefact
dns_wt['MeanValue'] *= scalefact


# =============================================================================
# USEFUL FUNCTIONS FOR FITTING TO DATA OBTAINED FROM THERMODYNAMIC MODEL
# =============================================================================


def alt_modpred_forfit_acttweak(E,kT,kP,CTA,CTR,kR,CER,CEA,model):
    #calculate model predictions for all constructs studied experimentally
    #and for different choices of the activation probability function (model)
    
    kE = 1/60.
    Cnt = 1
    gamma = 1
    
    global N
    
    A_gbe_mut = kT * kP * CTA / (1+kP *CTA) 
    A_brk_mut = kT * kP * CTR / (1+kP *CTR) 
    
    P0_base   =  1   /(1 + kR + kE*E)
    Pact_base =  kE*E/(1 + kR + kE*E)
    Prep_base =  kR  /(1 + kR + kE*E)
    
    
    if model == 'onlyone':
        P0   =  P0_base
        Pact =  Pact_base
        Prep =  Prep_base

    elif model == 'tworequired':
        P0   =  (1-Prep_base)**2 - Pact_base**2
        Pact =  Pact_base**2
        Prep =  1 - (1-Prep_base)**2
        
    elif model == 'absolutemajority':
        Pact = 0
        for nA in np.arange(np.math.ceil((N+1)/2),N+1,1):
            Pact += np.math.factorial(N)/(np.math.factorial(N-nA)*np.math.factorial(nA)) * Pact_base**nA * (1-Pact_base)**(N-nA)
        Prep = 0
        for nR in np.arange(np.math.ceil((N)/2),N+1,1):
            Prep += np.math.factorial(N)/(np.math.factorial(N-nR)*np.math.factorial(nR)) * Prep_base**nR * (1-Prep_base)**(N-nR)
        #Prep += np.math.factorial(N)/(np.math.factorial(N-5)*np.math.factorial(5)) * Prep_base**5 * Pact_base**(N-5)
        P0 = 1 - Pact - Prep
        
    A_gbe = kT * kP * CTA * (P0 + CER*Prep + gamma*CEA*Pact)/(1 + kP*CTA*(P0 + CER*Prep + gamma*CEA*Pact)) 
    A_brk = kT * kP * CTR * (P0 + CER*Prep +     CEA*Pact)/(1 + kP*CTR*(P0 + CER*Prep +     CEA*Pact)) 
    A_ere = kT * kP * Cnt * (P0 + CER*Prep +     CEA*Pact)/(1 + kP*Cnt*(P0 + CER*Prep +     CEA*Pact)) 
    
    return np.array([A_gbe,A_gbe_mut,A_brk,A_brk_mut,A_ere]).T.ravel()


# =============================================================================
# FIT USING LEAST SQUARES to avoid small errors at low signal
# =============================================================================
    
def alt_singles_lambdatweak(E,kT,kP,CTA,CTR,kR,CER,CEA,construct):
    #same as previous but used to calculate model predictions for a specific construct
    
    global model
    global N
    gamma = 1
    
    kE = 1/60.
    Cnt = 1
    
    A_gbe_mut = kT * kP * CTA / (1+kP *CTA) 
    A_brk_mut = kT * kP * CTR / (1+kP *CTR) 
    
    P0_base   =  1   /(1 + kR + kE*E)
    Pact_base =  kE*E/(1 + kR + kE*E)
    Prep_base =  kR  /(1 + kR + kE*E)
    
    
    if model == 'onlyone':
        P0   =  P0_base
        Pact =  Pact_base
        Prep =  Prep_base
        
    elif model == 'tworequired':
        P0   =  (1-Prep_base)**2 - Pact_base**2
        Pact =  Pact_base**2
        Prep =  1 - (1-Prep_base)**2
        
    elif model == 'absolutemajority':
        Pact = 0
        for nA in np.arange(np.math.ceil((N+1)/2),N+1,1):
            Pact += np.math.factorial(N)/(np.math.factorial(N-nA)*np.math.factorial(nA)) * Pact_base**nA * (1-Pact_base)**(N-nA)
        Prep = 0
        for nR in np.arange(np.math.ceil((N)/2),N+1,1):
            Prep += np.math.factorial(N)/(np.math.factorial(N-nR)*np.math.factorial(nR)) * Prep_base**nR * (1-Prep_base)**(N-nR)
        P0 = 1 - Pact - Prep
        
    A_gbe = kT * kP * CTA * (P0 + CER*Prep + gamma*CEA*Pact)/(1 + kP*CTA*(P0 + CER*Prep + gamma*CEA*Pact)) 
    A_brk = kT * kP * CTR * (P0 + CER*Prep +       CEA*Pact)/(1 + kP*CTR*(P0 + CER*Prep +     CEA*Pact)) 
    A_ere = kT * kP * Cnt * (P0 + CER*Prep +       CEA*Pact)/(1 + kP*Cnt*(P0 + CER*Prep +     CEA*Pact)) 
    
    
    if construct =='gbe10ere':
        return A_gbe
    elif construct == 'gbemut':
        return A_gbe_mut
    elif construct == 'ere':
        return A_brk
    elif construct == 'mut':
        return A_brk_mut
    elif construct == '10ere':
        return A_ere



def prob_act_mods(E,model,N):
    #evaluates probability of neutral, activating and repressive form of EcR as
    #a function of 20E concentration, for a given number of EREs in the set 
    #and for a specific choice of the cooperative interaction rule (model)
    
    kR = 1.04
    kE =1/60.
    P0_base   =  1   /(1. + kR + kE*E)
    Pact_base =  kE*E/(1. + kR + kE*E)
    Prep_base =  kR  /(1. + kR + kE*E)
    
    
    if model == 'onlyone':
        P0   =  P0_base
        Pact =  Pact_base
        Prep =  Prep_base

    elif model == 'tworequired':
        P0   =  (1-Prep_base)**2 - Pact_base**2
        Pact =  Pact_base**2
        Prep =  1 - (1-Prep_base)**2
    
    elif model == 'absolutemajority':
        Pact = 0
        for nA in np.arange(np.math.ceil((N+1)/2),N+1,1):
            Pact += np.math.factorial(N)/(np.math.factorial(N-nA)*np.math.factorial(nA)) * Pact_base**nA * (1-Pact_base)**(N-nA)
        Prep = 0
        for nR in np.arange(np.math.ceil((N)/2),N+1,1):
            Prep += np.math.factorial(N)/(np.math.factorial(N-nR)*np.math.factorial(nR)) * Prep_base**nR * (1-Prep_base)**(N-nR)
        #Prep += np.math.factorial(N)/(np.math.factorial(N-5)*np.math.factorial(5)) * Prep_base**5 * Pact_base**(N-5)
        P0 = 1 - Pact - Prep
        
    return P0, Pact, Prep


def square_dev_data_model(pars):
    #evaluates loss function for minimisation summing across realisations and constructs 
    
    A0,kP,CTA,CTR,kR,CER,CEA = np.exp(pars)
    cumul = 0
    
    for construct in ['gbe10ere','gbemut','ere','mut','10ere']:
        for condition in [0,20,200,2000]:
            
            prediction = alt_singles_lambdatweak(condition,A0,kP,CTA,CTR,kR,CER,CEA,construct)
            
            if construct == 'gbe10ere' or construct == 'gbemut':
                relevant_data = data_10ere_Enh.loc[data_10ere_Enh['Genotype'] == construct].loc[data_10ere_Enh['Condition'] == condition]['MeanValue'].to_numpy()
                cumul += np.sum((relevant_data - prediction)**2) 
                
            elif construct == 'ere' or construct == 'mut':
                relevant_data = data_10ere_Sil.loc[data_10ere_Sil['Genotype'] == construct].loc[data_10ere_Sil['Condition'] == condition]['MeanValue'].to_numpy()
                cumul += np.sum((relevant_data - prediction)**2) 
                
            elif construct == '10ere':
                relevant_data = scalefact*data_10ere.loc[data_10ere['Genotype'] == construct].loc[data_10ere['Condition'] == condition]['MeanValue'].to_numpy()
                cumul += np.sum((relevant_data - prediction)**2) 
    
    #cumul += 100*(np.log(kR))**2 #np.sum(pars**2)
    
    return cumul


# =============================================================================
# FIT TO REPORTER DATA - PARAMETER EXTRACTION
# =============================================================================

  

#Abscisse coordinates for later plots
conditions = [0,20,200,2000]
cond_id = [0,1,2,3]
condcont = np.logspace(-1, np.log10(2000), 50)  

initial_guess = np.ones(7)
N = 10

plt.figure()

models = ['onlyone','tworequired','absolutemajority']
pars = ['kT','kP','CTA','CTR','kR','CER','CEA']

for m in range(len(models)):
    
    plt.subplot(1,3,m+1)
    plt.errorbar(dem_wt['Condition'],dem_wt['MeanValue'],yerr=des_wt['MeanValue'],marker='o',c='tab:green',fmt=' ', capsize=10)
    plt.errorbar(dem_mut['Condition'],dem_mut['MeanValue'],yerr=des_mut['MeanValue'],marker='x',c='lime',fmt=' ', capsize=10)
    
    plt.errorbar(dsm_wt['Condition'],dsm_wt['MeanValue'],yerr=dss_wt['MeanValue'],marker='o',c='tab:red',fmt=' ', capsize=10)
    plt.errorbar(dsm_mut['Condition'],dsm_mut['MeanValue'],yerr=dss_mut['MeanValue'],marker='x',c='orange',fmt=' ', capsize=10)
    
    plt.errorbar(dnm_wt['Condition'],dnm_wt['MeanValue'],yerr=dns_wt['MeanValue'],marker='o',c='tab:purple',fmt=' ', capsize=10)
    
    plt.xscale('symlog')
    
    plt.xlabel('[20E]'); plt.ylabel('Reporter signal [a.u.]')
    plt.legend(['GBE-ERE','GBE','Brk-ERE','Brk','ERE'])
    
    model = models[m]
    min_loc_a = minimize(square_dev_data_model,initial_guess,method='Nelder-Mead',options={'maxiter': 1e4})
    print(min_loc_a)
    min_loc_log_a = np.exp(min_loc_a.x)
    
    initial_guess = min_loc_a.x
    
    print(models[m])
    for pn in range(len(pars)):
        print(pars[pn],min_loc_log_a[pn])
    
    levels = np.array([alt_modpred_forfit_acttweak(xi,min_loc_log_a[0],min_loc_log_a[1],min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],models[m]) for xi in condcont])
    plt.plot(condcont,levels,'-',linewidth=3,alpha=0.4)    
    plt.title(models[m]+',    'r'$\sum \sigma=$'+str(int(square_dev_data_model(min_loc_a.x))))# - np.sum(min_loc_a.x**2))))
    
    plt.xlim(-0.5,4000)

plt.tight_layout()

# =============================================================================
# FIG S5E - nested temporal activation domains
# =============================================================================

plt.figure();plt.xscale('log')
plt.xlabel('[20E]'); plt.ylabel('Signal [a.u.]')
plt.xlim(-10,10**3.3)

for constr in ['gbe10ere','gbemut','ere','mut','10ere']:
    levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],5*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],constr) for xi in condcont])
    plt.plot(condcont,levels,linewidth=3,alpha=1)  

plt.legend(['Enh-ERE','Enh','Sil-ERE','Sil','ERE'])

plt.hlines(37,0.1,4000,color='k',linestyle='--')
plt.hlines(7,0.1,4000,color='k',linestyle='dotted')
       


# =============================================================================
# FIG 4G - normalised reponses
# =============================================================================

min_loc_log_a = np.array([1e2, 1e-01, 1, 1e-01, 1.0, 1e-1, 10])
condcont = [10**a for a in np.linspace(-1,3.3,100)]

plt.figure()

plt.subplot(1,2,1);plt.xscale('log')
plt.xlabel('[20E]'); plt.ylabel('Normalised signal')
plt.legend(['GBE-ERE','GBE','Brk-ERE','Brk','ERE'])
plt.xlim(-10,10**3.3)

levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],0.1*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],'gbe10ere') for xi in condcont])
plt.plot(condcont,levels/np.max(levels),linewidth=3,alpha=1,c='k')  

levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],1.0*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],'gbe10ere') for xi in condcont])
plt.plot(condcont,levels/np.max(levels),linewidth=3,alpha=1,c='tab:red')  

levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],10.0*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],'gbe10ere') for xi in condcont])
plt.plot(condcont,levels/np.max(levels),linewidth=3,alpha=1,c='tab:blue')  

levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],100.*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],'gbe10ere') for xi in condcont])
plt.plot(condcont,levels/np.max(levels),linewidth=3,alpha=1,c='tab:green')    


plt.legend([r'$C_{T} =0.1$',r'$C_{T} =1$',r'$C_{T}=10$',r'$C_{T}=100$'])
    
# =============================================================================
# FIG 4H - correlation in feature map
# =============================================================================

min_loc_log_a = np.array([1e2, 1e-01, 1, 1e-01, 1.0, 1e-1, 10])
plt.subplot(1,2,2)

xpts = []
ypts = []

for eps in [10**a for a in np.linspace(-1,2,100)]:
    levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],eps*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],'gbe10ere') for xi in conditions])
    xpts.append((levels[-1]-levels[1])/(levels[1]-levels[0]))
    ypts.append(levels[-1]/levels[0])

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "tab:red", "tab:blue","tab:green"]) 

for eps in [0.1,1.,10.,100.]:
    levels = np.array([alt_singles_lambdatweak(xi,min_loc_log_a[0],min_loc_log_a[1],eps*min_loc_log_a[2],min_loc_log_a[3],min_loc_log_a[4],min_loc_log_a[5],min_loc_log_a[6],'gbe10ere') for xi in conditions])
    x1 = (levels[-1]-levels[1])/(levels[1]-levels[0])
    x2 = levels[-1]/levels[0]
    plt.scatter(x1,x2,s=40)
    
plt.loglog(xpts,ypts,alpha=0.4)

plt.legend([r'$C_{T} =0.1$',r'$C_{T} =1$',r'$C_{T}=10$',r'$C_{T}=100$'])

plt.xlim(0.1,100);plt.ylim(1,30)

plt.xlabel(r'$\delta_{2000-20}/\delta_{20-0}$')
plt.ylabel(r'Fold change (2000 - 0 nM)')

plt.tight_layout()

# =============================================================================
# FIG S5D - EcR complex probabilities for different models
# =============================================================================

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "orange", "tab:green"]) 

plt.figure()
condcont = [10**a for a in np.linspace(-1,4,100)]
models = ['onlyone','tworequired','absolutemajority']

for m in range(len(models)):

    plt.subplot(1,3,m+1)
    plt.xscale('log')
    levels = np.array([prob_act_mods(xi,models[m],10) for xi in condcont])
    plt.plot(condcont,levels,'-',linewidth=3,alpha=1)
    
    plt.legend([r'$P_0(E)$',r'$P_{act}(E)$',r'$P_{rep}(E)$'])
    plt.xlim(1,10**4);plt.ylim(0,1)
    
    plt.xlabel('[20E]'); plt.ylabel('Probability')
    plt.title(models[m])
