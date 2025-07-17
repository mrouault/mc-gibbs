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

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True

#load points
dir_mcmc = "../xps/points_mcmc/points_mcmc_500_0.p"
dir_gibbs_mh_n2 = "../xps/points_gibbs_mh_tempn2_env1000_itgibbs10000/points_gibbs_mh_500_0.p"
dir_gibbs_mh_n3 = "../xps/points_gibbs_mh_temp_n3_env1000_itgibbs10000/points_gibbs_mh_500_0.p"

points_mcmc = pickle.load(open(dir_mcmc,  "rb"))["points_mcmc"]
points_gibbs_mh_n2 = pickle.load(open(dir_gibbs_mh_n2, "rb"))["points_gibbs"]
points_gibbs_mh_n3 =pickle.load(open(dir_gibbs_mh_n3, "rb"))["points_gibbs"]

ac_mcmc = gd.AlphaComplex(points_mcmc)
st = ac_mcmc.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_mcmc.pdf")

plt.clf()
plt.cla()
ac_gibbs_mh = gd.AlphaComplex(points_gibbs_mh_n2)
st = ac_gibbs_mh.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_gibbs_mh_n2.pdf")

plt.clf()
plt.cla()
ac_gibbs_gibbs = gd.AlphaComplex(points_gibbs_mh_n3)
st = ac_gibbs_gibbs.create_simplex_tree()
st.compute_persistence()
diagram = st.persistence_intervals_in_dimension(2)
axes = gd.plot_persistence_diagram(diagram)
plt.savefig("test_persistence_diagram_gibbs_mh_n3.pdf")
