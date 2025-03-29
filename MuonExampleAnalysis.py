import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
from textwrap import wrap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Polygon

matplotlib.rcParams.update({'font.size': 20})

# Function to calculate deltaR with broadcasting
def deltaR(pf_Eta, pf_Phi, mc_Eta, mc_Phi):
    # Broadcasting to compute deltaR between all combinations of pf and mc elements
    dEta = mc_Eta[:, None] - pf_Eta  # mc_Eta will be broadcasted to match pf_Eta
    dPhi = mc_Phi[:, None] - pf_Phi  # mc_Phi will be broadcasted to match pf_Phi

    # Computing deltaR using the formula
    return np.sqrt(np.square(dEta) + np.square(dPhi))


matplotlib.rcParams.update({'font.size': 20})

# Place the corresponding files downloaded from https://huggingface.co/datasets/TSaala/IntroductionToOpenData/blob/main
# In the same folder as this script
df_real = pd.read_feather('SingleMu_Filtered.feather')
df_dymumu = pd.read_feather('DYToMuMu_Filtered.feather')
df_dijet = pd.read_feather('DiJet_Filtered.feather')
df_wmunu = pd.read_feather('WplusToMuNu_Filtered.feather')

df_real = df_real[df_real['nMuon'] == 1]

df_dymumu = df_dymumu[df_dymumu['nMuon'] == 1]
df_dymumu = df_dymumu[df_dymumu['vecMctruth_PdgId'].notna()]
df_dijet = df_dijet[df_dijet['nMuon'] == 1]
df_dijet = df_dijet[df_dijet['vecMctruth_PdgId'].notna()]
df_wmunu = df_wmunu[df_wmunu['nMuon'] == 1]
df_wmunu = df_wmunu[df_wmunu['vecMctruth_PdgId'].notna()]

print(len(df_real))
print(len(df_dymumu))
print(len(df_dijet))
print(len(df_wmunu))

df_real = df_real.sample(n=10000, random_state=42)
df_dymumu = df_dymumu.sample(n=10000, random_state=42)
df_dijet = df_dijet.sample(n=10000, random_state=42)
df_wmunu = df_wmunu.sample(n=10000, random_state=42)

# McTruth W+ PT

pdgid = df_wmunu['vecMctruth_PdgId'].to_numpy()
mc_pt = df_wmunu['vecMctruth_PT'].to_numpy()

pt_muons = []
for i in range(len(pdgid)):
    tmp_muons_pt = []
    for j in range(len(pdgid[i])):
        if pdgid[i][j] in {-13, 13}:
            tmp_muons_pt.append(mc_pt[i][j])

    pt_muons.append(tmp_muons_pt)

pt_muons = [item for sublist in pt_muons for item in sublist]


pf_pt = df_wmunu['vecMuon_PT'].to_numpy()
pf_pt = [item for sublist in pf_pt for item in sublist]

# Define fixed bin edges (you can adjust the range and number of bins)
min_pt = min(np.min(pt_muons), np.min(pf_pt))  # Minimum value across all data
max_pt = 100  # Maximum value across all data
bins = 20  # Number of bins
bin_edges = np.linspace(10, max_pt, bins+1)  # Create fixed bin edges

plt.figure(figsize=(16.0, 9.0))
n, bins, patches = plt.hist(pt_muons, bins=bin_edges, density=True, color='#ffb300', edgecolor=None, rasterized=True,
                            label=r'MC Muon Transverse Momentum ($p_T$), W -> $\mu$ $\nu$')

# Construct vertices for a single outline
vertices = []

# Start at the bottom-left corner (first bin edge at y=0)
vertices.append((bins[0], 0))

# Add points along the top edges of the bins
for i in range(len(n)):
    vertices.append((bins[i], n[i]))  # Left-top corner of the bin
    vertices.append((bins[i + 1], n[i]))  # Right-top corner of the bin

# Add the bottom-right corner (last bin edge at y=0)
vertices.append((bins[-1], 0))

# Close the path by connecting back to the starting point
vertices.append((bins[0], 0))

# Create a Polygon patch for the border
border = Polygon(vertices, edgecolor='black', facecolor='none', lw=2)

# Add the border to the plot
plt.gca().add_patch(border)


plt.xlabel(r'Transverse Momentum ($p_T$) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.07)
plt.xlim(xmin=10.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()


# PF W+ PT

plt.figure(figsize=(16.0, 9.0))
n, bins, patches = plt.hist(pf_pt, bins=bin_edges, density=True, color='#004dff', edgecolor=None, rasterized=True,
                            label=r'PF Muon Transverse Momentum ($p_T$), W -> $\mu$ $\nu$')

# Construct vertices for a single outline
vertices = []

# Start at the bottom-left corner (first bin edge at y=0)
vertices.append((bins[0], 0))

# Add points along the top edges of the bins
for i in range(len(n)):
    vertices.append((bins[i], n[i]))  # Left-top corner of the bin
    vertices.append((bins[i + 1], n[i]))  # Right-top corner of the bin

# Add the bottom-right corner (last bin edge at y=0)
vertices.append((bins[-1], 0))

# Close the path by connecting back to the starting point
vertices.append((bins[0], 0))

# Create a Polygon patch for the border
border = Polygon(vertices, edgecolor='black', facecolor='none', lw=2)

# Add the border to the plot
plt.gca().add_patch(border)

plt.xlabel(r'Transverse Momentum ($p_T$) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.05)
plt.xlim(xmin=10.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()

# W+ Mc Neutrino PT

pdgid = df_wmunu['vecMctruth_PdgId'].to_numpy()
mc_pt = df_wmunu['vecMctruth_PT'].to_numpy()
mc_m1 = df_wmunu['vecMctruth_Mothers.first'].to_numpy()
mc_m2 = df_wmunu['vecMctruth_Mothers.second'].to_numpy()

pt_neutrinos = []
tmp_dbg = []
for i in range(len(pdgid)):
    tmp_neutrinos_pt = []
    for j in range(len(pdgid[i])):
        # and 24 in {mc_m1[i][j], mc_m2[i][j]}
        if pdgid[i][j] in {-12, 12, -14, 14, -16, 16, -18, 18} and (mc_m1[i][j] > -1 or mc_m2[i][j] > -1) \
                and 24 in {int(pdgid[i][mc_m1[i][j]]), int(pdgid[i][mc_m2[i][j]])}:
            tmp_neutrinos_pt.append(mc_pt[i][j])

    pt_neutrinos.append(np.sum(np.abs(tmp_neutrinos_pt)))
    tmp_dbg.append(tmp_neutrinos_pt)


wp_met = df_wmunu['fMET_PT'].to_numpy()

# Define fixed bin edges (you can adjust the range and number of bins)
min_MET = min(np.min(pt_neutrinos), np.min(wp_met))  # Minimum value across all data
max_MET = 100 # Maximum value across all data
bins = 20  # Number of bins
bin_edges = np.linspace(min_MET, max_MET, bins+1)  # Create fixed bin edges


plt.figure(figsize=(16.0, 9.0))
n, bins, patches = plt.hist(pt_neutrinos, bins=bin_edges, density=True, color='#ffb300', edgecolor=None, rasterized=True,
                            label=r'MC Neutrino Transverse Momentum ($p_T$), W -> $\mu$ $\nu$')

# Construct vertices for a single outline
vertices = []

# Start at the bottom-left corner (first bin edge at y=0)
vertices.append((bins[0], 0))

# Add points along the top edges of the bins
for i in range(len(n)):
    vertices.append((bins[i], n[i]))  # Left-top corner of the bin
    vertices.append((bins[i + 1], n[i]))  # Right-top corner of the bin

# Add the bottom-right corner (last bin edge at y=0)
vertices.append((bins[-1], 0))

# Close the path by connecting back to the starting point
vertices.append((bins[0], 0))

# Create a Polygon patch for the border
border = Polygon(vertices, edgecolor='black', facecolor='none', lw=2)

# Add the border to the plot
plt.gca().add_patch(border)


plt.xlabel(r'Transverse Momentum ($p_T$) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.05)
plt.xlim(xmin=0.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()

# W+ PF MET

plt.figure(figsize=(16.0, 9.0))
n, bins, patches = plt.hist(wp_met, bins=bin_edges, density=True, color='#004dff', edgecolor=None, rasterized=True,
                            label=r'PF Missing Transverse Energy (MET), W -> $\mu$ $\nu$')

# Construct vertices for a single outline
vertices = []

# Start at the bottom-left corner (first bin edge at y=0)
vertices.append((bins[0], 0))

# Add points along the top edges of the bins
for i in range(len(n)):
    vertices.append((bins[i], n[i]))  # Left-top corner of the bin
    vertices.append((bins[i + 1], n[i]))  # Right-top corner of the bin

# Add the bottom-right corner (last bin edge at y=0)
vertices.append((bins[-1], 0))

# Close the path by connecting back to the starting point
vertices.append((bins[0], 0))

# Create a Polygon patch for the border
border = Polygon(vertices, edgecolor='black', facecolor='none', lw=2)

# Add the border to the plot
plt.gca().add_patch(border)


#plt.title("\n".join(wrap('Distribution of the Reconstructed Missing Transverse Energy (MET) for the W -> Mu Nu dataset', 35, initial_indent='')),
#          x=0.3, y=0.74)
plt.xlabel('Missing Transverse Energy (MET) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(ymin=0.0, ymax=0.035)
plt.xlim(xmin=0.0)
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()


# Next: MuonPT Signal vs Background, MET Signal vs Background, Isolation Signal vs Background
real_pt = df_real['vecMuon_PT'].to_numpy()
real_MET = df_real['fMET_PT'].to_numpy()
real_iso = df_real['vecMuon_TrkIso03'].to_numpy()
real_eta = df_real['vecMuon_Eta'].to_numpy()


signal_pt = pf_pt
signal_MET = wp_met
signal_iso = df_wmunu['vecMuon_TrkIso03'].to_numpy()
signal_eta = df_wmunu['vecMuon_Eta'].to_numpy()

dy_pt = df_dymumu['vecMuon_PT'].to_numpy()
dy_MET = df_dymumu['fMET_PT'].to_numpy()
dy_iso = df_dymumu['vecMuon_TrkIso03'].to_numpy()
dy_eta = df_dymumu['vecMuon_Eta'].to_numpy()

dijet_pt = df_dijet['vecMuon_PT'].to_numpy()
dijet_MET = df_dijet['fMET_PT'].to_numpy()
dijet_iso = df_dijet['vecMuon_TrkIso03'].to_numpy()
dijet_eta = df_dijet['vecMuon_Eta'].to_numpy()

dijet_pt = [float(arr) for arr in dijet_pt]
dy_pt = [float(arr) for arr in dy_pt]
real_pt = [float(arr) for arr in real_pt]
dijet_MET = [float(arr) for arr in dijet_MET]
dy_MET = [float(arr) for arr in dy_MET]
real_MET = [float(arr) for arr in real_MET]
dijet_iso = [float(arr) for arr in dijet_iso]
dy_iso = [float(arr) for arr in dy_iso]
signal_iso = [float(arr) for arr in signal_iso]
real_iso = [float(arr) for arr in real_iso]
dijet_eta = [float(arr) for arr in dijet_eta]
dy_eta = [float(arr) for arr in dy_eta]
signal_eta = [float(arr) for arr in signal_eta]
real_eta = [float(arr) for arr in real_eta]

# Define fixed bin edges (you can adjust the range and number of bins)
min_pt = min(np.min(signal_pt), np.min(dijet_pt), np.min(dy_pt))  # Minimum value across all data
max_pt = 100  # Maximum value across all data
bins = 25  # Number of bins
bin_edges = np.linspace(0, max_pt, bins+1)  # Create fixed bin edges


plt.figure(figsize=(16.0, 9.0))

# Calculate histogram values
signal_hist, _ = np.histogram(signal_pt, bins=bin_edges, density=True)
dijet_hist, _ = np.histogram(dijet_pt, bins=bin_edges, density=True)
dy_hist, _ = np.histogram(dy_pt, bins=bin_edges, density=True)

# Extend histogram values for step plots (prepend and append 0)
extended_signal_hist = np.concatenate(([0], signal_hist, [0]))
extended_dijet_hist = np.concatenate(([0], dijet_hist, [0]))
extended_dy_hist = np.concatenate(([0], dy_hist, [0]))

# Extend bin edges
bin_width = bin_edges[1] - bin_edges[0]  # Assume uniform bins
extended_bin_edges = np.concatenate(([bin_edges[0] - bin_width], bin_edges, [bin_edges[-1] + bin_width]))

# Plot background histograms first (so they appear behind the signal)
plt.step(extended_bin_edges[:-1], extended_dijet_hist, where='post', label='Background DiJet', color='#56B4E9', linewidth=3)  # Sky Blue
plt.step(extended_bin_edges[:-1], extended_dy_hist, where='post', label='Background Z -> Mu Mu', color='#009E73', linewidth=3)  # Teal

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Add hatching-filled areas with dense patterns
plt.fill_between(
    extended_bin_edges[:-1], extended_dijet_hist, step='post',
    color='none', hatch='\\\\', edgecolor='#56B4E9', linewidth=2
)
plt.fill_between(
    extended_bin_edges[:-1], extended_dy_hist, step='post',
    color='none', hatch='//', edgecolor='#009E73', linewidth=2
)

# Plot the signal histogram last (to ensure it's in the foreground)
plt.step(extended_bin_edges[:-1], extended_signal_hist, where='post', label='Signal', color='#E69F00', linewidth=3)  # Orange

# Add labels and legend
#plt.title("\n".join(wrap(r'Comparison of the Signal and Background Transverse Momentum ($p_T$) Distributions', 35, initial_indent='')),
#          x=0.3, y=0.74)
plt.xlabel(r'Transverse Momentum ($p_T$) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(0, max(np.max(signal_hist), np.max(dijet_hist), np.max(dy_hist)) * 1.4)
plt.xlim(xmin=0.0)  # Ensure x-axis starts at 0
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()

# Signal + Background MET

plt.figure(figsize=(16.0, 9.0))

min_met = min(np.min(signal_MET), np.min(dijet_MET), np.min(dy_MET))  # Minimum value across all data
max_met = 80  # Maximum value across all data
bins = 25  # Number of bins
bin_edges = np.linspace(min_met, max_met, bins+1)  # Create fixed bin edges

# Calculate histogram values
signal_hist, _ = np.histogram(signal_MET, bins=bin_edges, density=True)
dijet_hist, _ = np.histogram(dijet_MET, bins=bin_edges, density=True)
dy_hist, _ = np.histogram(dy_MET, bins=bin_edges, density=True)

# Extend histogram values for step plots (prepend and append 0)
extended_signal_hist = np.concatenate(([0], signal_hist, [0]))
extended_dijet_hist = np.concatenate(([0], dijet_hist, [0]))
extended_dy_hist = np.concatenate(([0], dy_hist, [0]))

# Extend bin edges
bin_width = bin_edges[1] - bin_edges[0]  # Assume uniform bins
extended_bin_edges = np.concatenate(([bin_edges[0] - bin_width], bin_edges, [bin_edges[-1] + bin_width]))

# Plot background histograms first (so they appear behind the signal)
plt.step(extended_bin_edges[:-1], extended_dijet_hist, where='post', label='Background DiJet', color='#56B4E9', linewidth=3)  # Sky Blue
plt.step(extended_bin_edges[:-1], extended_dy_hist, where='post', label='Background Z -> Mu Mu', color='#009E73', linewidth=3)  # Teal

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Add hatching-filled areas with dense patterns
plt.fill_between(
    extended_bin_edges[:-1], extended_dijet_hist, step='post',
    color='none', hatch='\\\\', edgecolor='#56B4E9', linewidth=2
)
plt.fill_between(
    extended_bin_edges[:-1], extended_dy_hist, step='post',
    color='none', hatch='//', edgecolor='#009E73', linewidth=2
)

# Plot the signal histogram last (to ensure it's in the foreground)
plt.step(extended_bin_edges[:-1], extended_signal_hist, where='post', label='Signal', color='#E69F00', linewidth=3)  # Orange

# Add labels and legend
#plt.title("\n".join(wrap('Comparison of the Signal and Background Missing Transverse Energy (MET) Distributions', 35, initial_indent='')),
#          x=0.3, y=0.74)
plt.xlabel(r'Missing Transverse Energy (MET) [GeV]', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(0, max(np.max(signal_hist), np.max(dijet_hist), np.max(dy_hist)) * 1.4)
plt.xlim(xmin=0.0)  # Ensure x-axis starts at 0
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()

# Signal + Background Isolation
plt.figure(figsize=(16.0, 9.0))

min_iso = min(np.min(signal_iso), np.min(dijet_iso), np.min(dy_iso))  # Minimum value across all data
max_iso = 6  # Maximum value across all data
bins = 25  # Number of bins
bin_edges = np.linspace(min_iso, max_iso, bins+1)  # Create fixed bin edges

print(min_iso)
print(max(np.max(signal_iso), np.max(dijet_iso), np.max(dy_iso)))
print(np.mean(signal_iso))
print(np.mean(dijet_iso))
print(np.mean(dy_iso))

# Calculate histogram values
signal_hist, _ = np.histogram(signal_iso, bins=bin_edges, density=True)
dijet_hist, _ = np.histogram(dijet_iso, bins=bin_edges, density=True)
dy_hist, _ = np.histogram(dy_iso, bins=bin_edges, density=True)

# Extend histogram values for step plots (prepend and append 0)
extended_signal_hist = np.concatenate(([0], signal_hist, [0]))
extended_dijet_hist = np.concatenate(([0], dijet_hist, [0]))
extended_dy_hist = np.concatenate(([0], dy_hist, [0]))

# Extend bin edges
bin_width = bin_edges[1] - bin_edges[0]  # Assume uniform bins
extended_bin_edges = np.concatenate(([bin_edges[0] - bin_width], bin_edges, [bin_edges[-1] + bin_width]))

# Plot background histograms first (so they appear behind the signal)
plt.step(extended_bin_edges[:-1], extended_dijet_hist, where='post', label='Background DiJet', color='#56B4E9', linewidth=3)  # Sky Blue
plt.step(extended_bin_edges[:-1], extended_dy_hist, where='post', label='Background Z -> Mu Mu', color='#009E73', linewidth=3)  # Teal

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Add hatching-filled areas with dense patterns
plt.fill_between(
    extended_bin_edges[:-1], extended_dijet_hist, step='post',
    color='none', hatch='\\\\', edgecolor='#56B4E9', linewidth=2
)
plt.fill_between(
    extended_bin_edges[:-1], extended_dy_hist, step='post',
    color='none', hatch='//', edgecolor='#009E73', linewidth=2
)

# Plot the signal histogram last (to ensure it's in the foreground)
plt.step(extended_bin_edges[:-1], extended_signal_hist, where='post', label='Signal', color='#E69F00', linewidth=3)  # Orange

# Add labels and legend
#plt.title("\n".join(wrap('Comparison of the Signal and Background Track Isolation Distributions', 35, initial_indent='')),
#          x=0.3, y=0.74)
plt.xlabel(r'Track Isolation', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(0, max(np.max(signal_hist), np.max(dijet_hist), np.max(dy_hist)) * 1.4)
plt.xlim(xmin=0.0)  # Ensure x-axis starts at 0
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()


# Signal + Background Eta
plt.figure(figsize=(16.0, 9.0))

min_eta = -4  # Minimum value across all data
max_eta = 4  # Maximum value across all data
bins = 25  # Number of bins
bin_edges = np.linspace(min_eta, max_eta, bins+1)  # Create fixed bin edges

print(min_eta)

# Calculate histogram values
signal_hist, _ = np.histogram(signal_eta, bins=bin_edges, density=True)
dijet_hist, _ = np.histogram(dijet_eta, bins=bin_edges, density=True)
dy_hist, _ = np.histogram(dy_eta, bins=bin_edges, density=True)

# Extend histogram values for step plots (prepend and append 0)
extended_signal_hist = np.concatenate(([0], signal_hist, [0]))
extended_dijet_hist = np.concatenate(([0], dijet_hist, [0]))
extended_dy_hist = np.concatenate(([0], dy_hist, [0]))

# Extend bin edges
bin_width = bin_edges[1] - bin_edges[0]  # Assume uniform bins
extended_bin_edges = np.concatenate(([bin_edges[0] - bin_width], bin_edges, [bin_edges[-1] + bin_width]))

# Plot background histograms first (so they appear behind the signal)
plt.step(extended_bin_edges[:-1], extended_dijet_hist, where='post', label='Background DiJet', color='#56B4E9', linewidth=3)  # Sky Blue
plt.step(extended_bin_edges[:-1], extended_dy_hist, where='post', label='Background Z -> Mu Mu', color='#009E73', linewidth=3)  # Teal

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Add hatching-filled areas with dense patterns
plt.fill_between(
    extended_bin_edges[:-1], extended_dijet_hist, step='post',
    color='none', hatch='\\\\', edgecolor='#56B4E9', linewidth=2
)
plt.fill_between(
    extended_bin_edges[:-1], extended_dy_hist, step='post',
    color='none', hatch='//', edgecolor='#009E73', linewidth=2
)

# Plot the signal histogram last (to ensure it's in the foreground)
plt.step(extended_bin_edges[:-1], extended_signal_hist, where='post', label='Signal', color='#E69F00', linewidth=3)  # Orange

# Add labels and legend
#plt.title("\n".join(wrap(r'Comparison of the Signal and Background Pseudo-rapidity ($\eta$) Distributions', 35, initial_indent='')),
#          x=0.3, y=0.74)
plt.xlabel(r'$\eta$', loc='right', labelpad=15, fontsize=26)
plt.ylabel('Fraction of events', loc='top', labelpad=15, fontsize=26)
plt.ylim(0, max(np.max(signal_hist), np.max(dijet_hist), np.max(dy_hist)) * 1.4)  # Ensure x-axis starts at 0
plt.minorticks_on()
plt.tick_params(which='major', length=20, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.tick_params(which='minor', length=10, width=1.5, direction='in', top=True, right=True, labelsize='large')
plt.legend(frameon=False, handlelength=1.5, handleheight=1)
plt.show()
plt.close()

sys.exit()






