import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
sys.path.append('..')
sys.path.append('..\\.venv\\lib\\site-packages')
import matplotlib.pyplot as plt
import matplotlib as mpl
import gudhi as gd
import pickle
import numpy as np

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True

points = pickle.load(open("points.p", "rb"))
points_mcmc = points["mcmc"][:, 5000:5100].T
points_gibbs_mh = points["gibbs_mh"][:, :].T
points_gibbs_gibbs = points["gibbs_gibbs"][:, :].T
points_thinning = points["thinned"][:, :].T

ac_mcmc = gd.AlphaComplex(points_mcmc)
st = ac_mcmc.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_mcmc.pdf")

plt.clf()
plt.cla()
ac_gibbs_mh = gd.AlphaComplex(points_gibbs_mh)
st = ac_gibbs_mh.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_gibbs_mh.pdf")

plt.clf()
plt.cla()
ac_gibbs_gibbs = gd.AlphaComplex(points_gibbs_gibbs)
st = ac_gibbs_gibbs.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_gibbs_gibbs.pdf")

plt.clf()
plt.cla()
ac_thinning = gd.AlphaComplex(points_thinning)
st = ac_thinning.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_thinning.pdf")
