import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
sys.path.append('..')
sys.path.append('..\\.venv\\lib\\site-packages')
sys.path.append('..\\xps')
import matplotlib.pyplot as plt
import matplotlib as mpl
import gudhi as gd
import pickle
import numpy as np
from matplotlib import patches

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
#mpl.rcParams['text.usetex'] = True

#load points
dir_mcmc = "points_mcmc_500_0.p"
dir_gibbs_mh_n2 = "points_gibbs_mh_500_0_tempn2_env1000_itgibbs50000.p"
dir_gibbs_mh_n3 = "points_gibbs_mh_500_0_tempn3_env1000_itgibbs200000.p"

points_mcmc = pickle.load(open(dir_mcmc,  "rb"))["points_mcmc"].T
points_gibbs_mh_n2 = pickle.load(open(dir_gibbs_mh_n2, "rb"))["points_gibbs"].T
points_gibbs_mh_n3 =pickle.load(open(dir_gibbs_mh_n3, "rb"))["points_gibbs"].T

ac_mcmc = gd.AlphaComplex(points_mcmc)
st = ac_mcmc.create_simplex_tree()
st.compute_persistence()
diagram_mcmc = st.persistence_intervals_in_dimension(2)

ac_gibbs_mh = gd.AlphaComplex(points_gibbs_mh_n2)
st = ac_gibbs_mh.create_simplex_tree()
st.compute_persistence()
diagram_gibbs_n2 = st.persistence_intervals_in_dimension(2)

ac_gibbs_mh = gd.AlphaComplex(points_gibbs_mh_n3)
st = ac_gibbs_mh.create_simplex_tree()
st.compute_persistence()
diagram_gibbs_n3 = st.persistence_intervals_in_dimension(2)

# Supposons que vous avez plusieurs diagrammes de persistance
diagrams = [diagram_mcmc, diagram_gibbs_n2, diagram_gibbs_n3]  # chaque diagX est une liste de points [(birth, death), ...]

# Étape 1 : Trouver les min/max globaux pour les axes
all_points = [pt for diag in diagrams for pt in diag]
min_birth = min(pt[0] for pt in all_points)
max_birth = max(pt[0] for pt in all_points)
min_death = min(pt[1] for pt in all_points)
max_death = max(pt[1] for pt in all_points)

# Marges pour la lisibilité
margin = 0.05 * (max_death - min_birth)
min_plot = min_birth - margin
max_plot = max_death + margin

xlim = (min_plot, max_plot)
ylim = (min_plot, max_plot)

for i, diag in enumerate(diagrams):
    fig, ax = plt.subplots(figsize=(5, 5))

    # ➤ 1. Zone grisée sous la diagonale
    ax.add_patch(
        patches.Polygon(
            [[min_plot, min_plot], [max_plot, max_plot], [max_plot, min_plot]],
            closed=True,
            facecolor='lightgray',
            zorder=0
        )
    )

    # ➤ 2. Diagonale
    ax.plot([min_plot, max_plot], [min_plot, max_plot], color='black', linestyle='--', linewidth=1)

    # ➤ 3. Points de persistance
    diag = [pt for pt in diag if pt[1] > pt[0]]  # filtrer les points valides
    xs, ys = zip(*diag) if diag else ([], [])
    ax.scatter(xs, ys, s=20, label='Points')

    # ➤ 4. Format
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_aspect('equal')
    
    filename = f"persistence_diagram_{i+1}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")