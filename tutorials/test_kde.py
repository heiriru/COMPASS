import numpy as np
from scipy.stats import gaussian_kde
from autocvd import autocvd

from compass.ModelTransfuser import ModelTransfuser


def grid_kde_map_1d(samples, n_grid=20000):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)

    kde = gaussian_kde(samples.reshape(1, -1))

    grid = np.linspace(samples.min(), samples.max(), n_grid)
    density = kde.pdf(grid.reshape(1, -1))

    idx = int(np.argmax(density))
    return grid[idx], density[idx], kde


def run_test(mtf, name, samples, atol=1e-3):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)

    # This is the actual function under test:
    map_theta, kde_std = mtf._map_kde(samples[:, None])
    map_theta = float(map_theta[0])

    # Independent reference: dense-grid maximum of the same SciPy KDE.
    grid_map, grid_density, kde = grid_kde_map_1d(samples)

    density_at_map = float(kde.pdf([map_theta])[0])
    abs_diff = abs(map_theta - grid_map)

    print(f"\n{name}")
    print(f"sample mean:        {samples.mean():.6f}")
    print(f"ModelTransfuser MAP:{map_theta:.6f}")
    print(f"grid KDE MAP:       {grid_map:.6f}")
    print(f"abs difference:     {abs_diff:.6e}")
    print(f"density(_map):      {density_at_map:.6f}")
    print(f"density(gridMAP):   {grid_density:.6f}")
    print(f"kde_std returned:   {float(kde_std[0]):.6f}")

    assert abs_diff < atol, (
        f"{name} failed: ModelTransfuser._map_kde={map_theta:.6f}, "
        f"grid_map={grid_map:.6f}, abs_diff={abs_diff:.6e}"
    )


def main():
    autocvd(num_gpus=1, interval=1)

    # Create an instance only to call the real method.
    # This works if _map_kde does not depend on initialized model state.
    mtf = object.__new__(ModelTransfuser)

    rng = np.random.default_rng(0)

    samples_gauss = rng.normal(loc=-1.2, scale=0.08, size=2000)

    samples_skewed = np.concatenate([
        rng.normal(loc=-1.22, scale=0.08, size=1850),
        rng.normal(loc=-0.35, scale=0.35, size=150),
    ])

    samples_sharp = rng.normal(loc=-1.25, scale=0.015, size=2000)

    run_test(mtf, "Gaussian", samples_gauss)
    run_test(mtf, "Right-skewed", samples_skewed)
    run_test(mtf, "Sharp Gaussian", samples_sharp)

    print("\nAll _map_kde tests passed.")


if __name__ == "__main__":
    main()