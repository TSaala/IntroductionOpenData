import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import math
from sklearn.linear_model import LinearRegression
import time
from multiprocessing import Pool
import gc
from numpy.testing import assert_almost_equal
import matplotlib
import math
from textwrap import wrap

matplotlib.rcParams.update({'font.size': 20})

# Place the corresponding file downloaded from https://huggingface.co/datasets/TSaala/IntroductionToOpenData/blob/main
# In the same folder as this script
df = pd.read_feather(f'TTJets_SemiLeptMGDecays_8TeV_0.feather')


nEle = df['nEle'].to_numpy()
nMuon = df['nMuon'].to_numpy()

# Apply PT Cutoffs
muonPT = df['vecMuon_PT'].to_numpy()
elePT = df['vecEle_PT'].to_numpy()

bins = np.linspace(0, 100, 20)

flat_muonPT = []
for i in range(len(muonPT)):
    flat_muonPT.extend(muonPT[i].flatten())

flat_elePT = []
for i in range(len(elePT)):
    flat_elePT.extend(elePT[i].flatten())

plt.figure(figsize=(16.0, 9.0))
plt.hist(flat_muonPT, bins=bins, density=True, color='#004dff', label=r'Transverse Momentum ($p_T$) of Muons', rasterized=True)
plt.xlabel(r'Transverse Momentum ($p_T$) of Muons', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.16)
plt.xlim(xmin=0.0, xmax=100)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()


plt.figure(figsize=(16.0, 9.0))
plt.hist(flat_elePT, bins=bins, density=True, color='#ffb300', label=r'Transverse Momentum ($p_T$) of Electrons', rasterized=True)
plt.xlabel(r'Transverse Momentum ($p_T$) of Electrons', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.08)
plt.xlim(xmin=0.0, xmax=100)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()

muon_gt5 = []
ele_gt5 = []

for i in range(len(muonPT)):
    tmp_muon = 0
    for j in range(len(muonPT[i])):
        if muonPT[i][j] > 5.0:
            tmp_muon += 1
    tmp_ele = 0
    for j in range(len(elePT[i])):
        if elePT[i][j] > 5.0:
            tmp_ele += 1
    muon_gt5.append(tmp_muon)
    ele_gt5.append(tmp_ele)

# Histos after pt cutoff

plt.figure(figsize=(16.0, 9.0))
plt.hist(muon_gt5, bins=np.arange(0, 8, 1), density=True, color='#004dff',
         label=r'Amount of Muons (nMuon) after 5 GeV $p_T$ cutoff', rasterized=True)
plt.xlabel('Number of Muons (nMuon)', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.7)
plt.xlim(xmin=0.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()


plt.figure(figsize=(16.0, 9.0))
plt.hist(ele_gt5, bins=np.arange(0, 8, 1), density=True, color='#ffb300',
         label=r'Amount of Electrons (nEle)  after 5 GeV $p_T$ cutoff', rasterized=True)
plt.xlabel('Number of Electrons (nEle)', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.6)
plt.xlim(xmin=0.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()


df_three_events = df.sample(n=3, random_state=123)

# for three events get the number of objects contained in each event
nEle = df_three_events['nEle'].to_numpy()
nMuon = df_three_events['nMuon'].to_numpy()
nTau = df_three_events['nTau'].to_numpy()
nPhoton = df_three_events['nPhoton'].to_numpy()
nPF = df_three_events['nPF'].to_numpy()
nVertex = df_three_events['nVertex'].to_numpy()
nMctruth = df_three_events['nMctruth'].to_numpy()
nJet = df_three_events['nJets'].to_numpy()


event_0_nMuon = df_three_events['nMuon'].to_numpy()[0]
event_0_muonPT = df_three_events['vecMuon_PT'].to_numpy()[0]
event_0_muonEta = df_three_events['vecMuon_Eta'].to_numpy()[0]
event_0_muonPhi = df_three_events['vecMuon_Phi'].to_numpy()[0]

event_0_nPF = df_three_events['nPF'].to_numpy()[0]
event_0_PFPT = df_three_events['vecPF_PT'].to_numpy()[0]
event_0_PFEta = df_three_events['vecPF_Eta'].to_numpy()[0]
event_0_PFPhi = df_three_events['vecPF_Phi'].to_numpy()[0]

event_0_nEle = df_three_events['nEle'].to_numpy()[0]
event_0_elePT = df_three_events['vecEle_PT'].to_numpy()[0]
event_0_eleEta = df_three_events['vecEle_Eta'].to_numpy()[0]
event_0_elePhi = df_three_events['vecEle_Phi'].to_numpy()[0]

event_1_nMuon = df_three_events['nMuon'].to_numpy()[1]
event_1_muonPT = df_three_events['vecMuon_PT'].to_numpy()[1]
event_1_muonEta = df_three_events['vecMuon_Eta'].to_numpy()[1]
event_1_muonPhi = df_three_events['vecMuon_Phi'].to_numpy()[1]

event_1_nPF = df_three_events['nPF'].to_numpy()[1]
event_1_PFPT = df_three_events['vecPF_PT'].to_numpy()[1]
event_1_PFEta = df_three_events['vecPF_Eta'].to_numpy()[1]
event_1_PFPhi = df_three_events['vecPF_Phi'].to_numpy()[1]

event_1_nEle = df_three_events['nEle'].to_numpy()[1]
event_1_elePT = df_three_events['vecEle_PT'].to_numpy()[1]
event_1_eleEta = df_three_events['vecEle_Eta'].to_numpy()[1]
event_1_elePhi = df_three_events['vecEle_Phi'].to_numpy()[1]

event_2_nMuon = df_three_events['nMuon'].to_numpy()[2]
event_2_muonPT = df_three_events['vecMuon_PT'].to_numpy()[2]
event_2_muonEta = df_three_events['vecMuon_Eta'].to_numpy()[2]
event_2_muonPhi = df_three_events['vecMuon_Phi'].to_numpy()[2]

event_2_nPF = df_three_events['nPF'].to_numpy()[2]
event_2_PFPT = df_three_events['vecPF_PT'].to_numpy()[2]
event_2_PFEta = df_three_events['vecPF_Eta'].to_numpy()[2]
event_2_PFPhi = df_three_events['vecPF_Phi'].to_numpy()[2]

event_2_nEle = df_three_events['nEle'].to_numpy()[2]
event_2_elePT = df_three_events['vecEle_PT'].to_numpy()[2]
event_2_eleEta = df_three_events['vecEle_Eta'].to_numpy()[2]
event_2_elePhi = df_three_events['vecEle_Phi'].to_numpy()[2]

# nMuon / nPF

nMuon = df['nMuon'].to_numpy()
nPF = df['nPF'].to_numpy()
nEle = df['nEle'].to_numpy()


plt.figure(figsize=(16.0, 9.0))
plt.hist(nPF, bins=20, density=True, color='#737362', label='Amount of particle flow objects (nPF)', rasterized=True)
plt.xlabel('Number of Particle Flow Objects (nPF)', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.0015)
plt.xlim(xmin=0.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()

plt.figure(figsize=(16.0, 9.0))
plt.hist(nMuon, bins=np.max(nMuon), density=True, color='#004dff', label='Amount of muons (nMuon)', rasterized=True)
plt.xlabel('Number of Muons (nMuon)', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.35)
plt.xlim(xmin=0.0, xmax=20.0)
plt.xticks([0, 5, 10, 15, 20])
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()


plt.figure(figsize=(16.0, 9.0))
plt.hist(nEle, bins=np.max(nEle), density=True, color='#ffb300', label='Amount of electrons (nEle)', rasterized=True)
plt.xlabel('Number of Electrons (nEle)', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.5)
plt.xlim(xmin=0.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()

# PT

muon_PT = df['vecMuon_PT'].to_numpy()
flat_muonPT = []

for i in range(len(muon_PT)):
    flat_muonPT.extend(muon_PT[i].flatten())


PF_PT = df['vecPF_PT'].to_numpy()
flat_PFPT = []

for i in range(len(PF_PT)):
    flat_PFPT.extend(PF_PT[i].flatten())


bins = np.linspace(0, 200, 40)

plt.figure(figsize=(16.0, 9.0))
plt.hist(flat_PFPT, bins=np.arange(0, 8, 0.5), density=True, color='#737362', label=r'PF Transversal momentum ($p_T$)', rasterized=True)
plt.xlabel('Transversal Impulse (pT) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=1.0)
plt.xlim(xmin=0.0, xmax=8)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()

# Eta
muon_Eta = df['vecMuon_Eta'].to_numpy()
flat_muonEta = []

for i in range(len(muon_Eta)):
    flat_muonEta.extend(muon_Eta[i].flatten())


PF_Eta = df['vecPF_Eta'].to_numpy()
flat_PFEta = []

for i in range(len(PF_Eta)):
    flat_PFEta.extend(PF_Eta[i].flatten())

bins = np.linspace(int(math.floor(np.min([np.min(flat_PFEta), np.min(flat_muonEta)]))), int(math.ceil(np.max([np.max(flat_PFEta), np.max(flat_muonEta)]))), 40)

plt.figure(figsize=(16.0, 9.0))
plt.hist(flat_PFEta, bins=np.arange(-8, 8, 0.5), density=True, color='#737362', label='Particle flow \u03B7', rasterized=True)
plt.xlabel('Pseudo-rapidity \u03B7', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.3)
plt.xlim(xmin=-8, xmax=8)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()


# Phi
muon_Phi = df['vecMuon_Phi'].to_numpy()
flat_muonPhi = []

for i in range(len(muon_Phi)):
    flat_muonPhi.extend(muon_Phi[i].flatten())


PF_Phi = df['vecPF_Phi'].to_numpy()
flat_PFPhi = []

for i in range(len(PF_Phi)):
    flat_PFPhi.extend(PF_Phi[i].flatten())


plt.figure(figsize=(16.0, 9.0))
plt.hist(flat_PFPhi, bins=np.arange(-5, 5, 0.25), density=True, color='#737362', label='Particle flow \u03C6', rasterized=True)
plt.xlabel('Polar coordinate \u03C6', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.3)
plt.xlim(xmin=-5, xmax=5)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1, fontsize=22)
plt.show()
plt.close()
sys.exit()

# Create an examplary greyscale image (Muon)

single_pt = PF_PT[0]
single_eta = PF_Eta[0]
single_phi = PF_Phi[0]

image_dim = 20
bins_phi = np.linspace(-np.pi, np.pi, num=image_dim)
bins_eta = np.linspace(-2*np.pi, 2*np.pi, num=image_dim)

first_image = np.zeros((image_dim, image_dim))
for i in range(len(single_pt)):
    phi_bin = bins_phi.flat[np.abs(bins_phi - single_phi[i]).argmin()]
    eta_bin = bins_eta.flat[np.abs(bins_eta - single_eta[i]).argmin()]
    phi_idx = int(np.where(bins_phi == phi_bin)[0])
    eta_idx = int(np.where(bins_eta == eta_bin)[0])
    first_image[eta_idx][phi_idx] += single_pt[i]


first_image = first_image / np.max(first_image)
print(first_image)

plt.gray()
plt.imshow(first_image)
plt.title('Greyscale image of a single PF object')
plt.axis('off')
plt.show()
plt.close()