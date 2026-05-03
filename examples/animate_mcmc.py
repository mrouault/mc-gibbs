import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib import colors
from scipy.stats import multivariate_normal, gaussian_kde
import os

# Set seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = "animation/mh"
save = True
# os.makedirs(output_dir, exist_ok=True)
fig_width = 8
fig_height = 5
num_bins = 25
# Define the target distribution: Mixture of two 2D Gaussians
weights = [0.5, 0.5]
means = [np.array([0, 0]), np.array([2, 1])]
covs = [np.array([[2, 0.8], [0.8, 1]]), np.array([[1, -0.6], [-0.6, 1.5]])]

def target_density(x):
    return sum(w * multivariate_normal.pdf(x, mean=mu, cov=cov)
               for w, mu, cov in zip(weights, means, covs))

def target_density_uniform_disk(x):
    center = np.array([1, 0.5])
    radius = 2
    if np.linalg.norm(x - center) <= radius:
        return 1 / (np.pi * radius**2)
    else:
        return 0


# Define MCMC sampling using Random Walk Metropolis-Hastings
def rwmh_sample(n_samples, initial, proposal_cov, target):
    samples = [initial]
    for _ in range(n_samples - 1):
        current = samples[-1]
        proposal = np.random.multivariate_normal(current, proposal_cov)
        accept_ratio = target(proposal) / target(current)
        if np.random.rand() < min(1, accept_ratio):
            samples.append(proposal)
        else:
            samples.append(current)
    return np.array(samples)

# Generate MCMC samples
n_samples = 600
proposal_cov = 0.5 * np.eye(2)
samples = rwmh_sample(n_samples, initial=np.array([0, 0]), proposal_cov=proposal_cov, target=target_density_uniform_disk)
samples = samples[::2]
n_samples = len(samples)

# Precompute the true marginals
grid_1d = np.linspace(-3, 8, 300)
true_marginal_x = sum(w * multivariate_normal.pdf(grid_1d, mean=mu[0], cov=cov[0,0])
                      for w, mu, cov in zip(weights, means, covs))
true_marginal_y = sum(w * multivariate_normal.pdf(grid_1d, mean=mu[1], cov=cov[1,1])
                      for w, mu, cov in zip(weights, means, covs))

# Create the figure range
x = y = np.linspace(-3, 5, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = np.array([target_density([x, y]) for x, y in pos.reshape(-1, 2)]).reshape(200, 200)

# Normalize for colormap
norm = colors.Normalize(vmin=0, vmax=n_samples)
base_cmap = plt.get_cmap("Greys")
colors_with_white = [(1, 1, 1)] + [base_cmap(i) for i in range(base_cmap.N)]
custom_cmap = colors.LinearSegmentedColormap.from_list("white_to_greys", colors_with_white, N=base_cmap.N)
t_BI = 10

# Generate frames
for t in range(1, n_samples):
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6], hspace=0.0, wspace=0.0)

    ax_main = fig.add_subplot(gs[1, 0])
    # ax_x = fig.add_subplot(gs[0, 0], sharex=ax_main)
    # ax_y = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main plot: 2D samples with level sets
    # ax_main.contour(X, Y, Z, levels=15, cmap=custom_cmap)
    # ax_main.contourf(X, Y, Z, levels=15, cmap=custom_cmap, alpha=0.5)

    start = max(0, t - 50)
    n_segments = t - start

    for i in range(start, t - 1):
        c = cm.viridis(norm(i))
        if i < t_BI:
            c = "red"

        alpha = (i - start) / (n_segments - 1) if n_segments > 1 else 1.0

        ax_main.plot(
            [samples[i, 0], samples[i + 1, 0]],
            [samples[i, 1], samples[i + 1, 1]],
            color=c,
            alpha=alpha,
        )

    ax_main.scatter(samples[:t, 0], samples[:t, 1], s=10, color="black", alpha=0.3)
    ax_main.set_xlim(-3, 4.5)
    ax_main.set_ylim(-3, 4.5)

    # Hide left/bottom axes of ax_main
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False,
                        right=False, top=False, labelright=False, labeltop=False)
    ax_main.set_aspect('equal', adjustable='box')

    # # Top marginal (X)
    # ax_x.plot(grid_1d, true_marginal_x, color='black', ls='--', label='True marginal')
    # if t > t_BI + 1:
    #     hist_counts, bin_edges = np.histogram(samples[t_BI:t, 0], bins=num_bins, density=True)
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     ax_x.bar(bin_centers, hist_counts, width=bin_edges[1]-bin_edges[0], color="#00284b", alpha=0.6)
    # ax_x.set_ylim(bottom=0)
    # ax_x.spines['right'].set_visible(False)
    # ax_x.spines['left'].set_visible(False)
    # ax_x.spines['top'].set_visible(False)
    # ax_x.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False,
    #                 right=False, top=False, labelright=False, labeltop=False)

    # # Right marginal (Y)
    # ax_y.plot(true_marginal_y, grid_1d, color='black', ls='--', label='True marginal')
    # if t > t_BI + 1:
    #     hist_counts, bin_edges = np.histogram(samples[t_BI:t, 1], bins=int(num_bins*(fig_height/fig_width)), density=True)
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     ax_y.barh(bin_centers, hist_counts, height=bin_edges[1]-bin_edges[0], color="#00284b", alpha=0.6)
    # ax_y.set_xlim(left=0)
    # ax_y.spines['top'].set_visible(False)
    # ax_y.spines['bottom'].set_visible(False)
    # ax_y.spines['right'].set_visible(False)
    # ax_y.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False,
    #                 right=False, top=False, labelright=False, labeltop=False)

    # Save or show
    filename = os.path.join(output_dir, f"frame_{t:03d}.pdf")
    if save:
        plt.savefig(filename, bbox_inches='tight')
    else:
        if t%50 == 0:
            print(f"Displaying frame {t}/{n_samples}")
            plt.show()
    plt.close()
