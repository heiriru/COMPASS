"""Compare population-dynamics hypotheses for each possible true model.

This is the script version of ``Population_Dynamics.ipynb``. It creates a set
of mock trajectories with each simulator in turn, evaluates all pretrained SBI
models against that set, and saves figures under
``tutorials/output/Population_Dynamics``.
"""

from pathlib import Path

from autocvd import autocvd
import torch

from compass import ModelTransfuser as MTf
from compass import ScoreBasedInferenceModel as SBIm


TUTORIAL_DIR = Path(__file__).resolve().parent
MODEL_DIR = TUTORIAL_DIR / "data" / "population_dynamics"
OUTPUT_DIR = TUTORIAL_DIR / "output" / "Population_Dynamics"

# Initial conditions [prey, predator] and log-space model parameters
INITIAL_STATE = torch.tensor([[30.0, 1.0]])
MOCK_PARAMS = torch.tensor([[-0.1, -3.0, -0.1, -3.0]])
T_MAX = 20
DT = 0.01
# Change this to control the number of trajectories used in each cumulative
# comparison. Each receives a small log-parameter perturbation.
NUM_TRAJECTORIES = 20
LOG_PARAMETER_STD = 0.1

# Both modes use the exact PF-ODE likelihood and annealed score-ascent MAP
# estimator. They differ only in the posterior sampler used by ``compare``.
COMPARISON_MODES = (
    {
        "name": "reverse_sde",
        "description": "reverse SDE (DPM order 2)",
        "equation": "reverse_sde",
        "method": "dpm",
        "order": 2,
    },
    {
        "name": "pfode_heun_exact",
        "description": "probability-flow ODE (Heun order 2)",
        "equation": "probability_flow_ode",
        "method": "heun",
        "order": 2,
    },
)


def solve_ode(model_func, initial_state, params, t_max, dt):
    """Solve a population model with the Euler integrator used in the notebook."""
    time_steps = torch.arange(0, t_max, dt)
    history = torch.zeros(initial_state.shape[0], len(time_steps), 2)
    history[:, 0, :] = initial_state
    current_state = initial_state.clone()

    for index in range(1, len(time_steps)):
        current_state += model_func(current_state, params) * dt
        current_state = torch.maximum(current_state, torch.zeros_like(current_state))
        history[:, index, :] = current_state

    return time_steps, history


def lotka_volterra(state, params):
    """Classic Lotka--Volterra dynamics."""
    prey, predator = state.T
    alpha, beta, gamma, delta = params.T
    return torch.stack(
        [alpha * prey - beta * prey * predator, delta * prey * predator - gamma * predator]
    ).T


def logistic_prey(state, params):
    """Prey with logistic growth."""
    prey, predator = state.T
    alpha, beta, gamma, delta = params.T
    carrying_capacity = delta * 1000
    conversion_rate = 0.5
    return torch.stack(
        [
            alpha * prey * (1 - prey / carrying_capacity) - beta * prey * predator,
            conversion_rate * beta * prey * predator - gamma * predator,
        ]
    ).T


def satiated_predator(state, params):
    """Predator satiation with a Holling type-II functional response."""
    prey, predator = state.T
    alpha, beta, gamma, delta = params.T
    conversion_rate = 0.5
    consumption = beta * prey / (1 + beta * delta * prey)
    return torch.stack(
        [alpha * prey - consumption * predator, conversion_rate * consumption * predator - gamma * predator]
    ).T


def rosenzweig_macarthur(state, params):
    """Logistic prey growth with predator satiation."""
    prey, predator = state.T
    alpha, beta, gamma, delta = params.T
    carrying_capacity = delta * 1000
    conversion_rate = 0.5
    handling_rate = 0.1
    consumption = beta * prey / (1 + beta * handling_rate * prey)
    return torch.stack(
        [
            alpha * prey * (1 - prey / carrying_capacity) - consumption * predator,
            conversion_rate * consumption * predator - gamma * predator,
        ]
    ).T


MODELS = {
    "Lotka-Volterra": lotka_volterra,
    "Logistic Prey": logistic_prey,
    "Satiated Predator": satiated_predator,
    "Rosenzweig-MacArthur": rosenzweig_macarthur,
}


def observations_for(model_func, num_trajectories=NUM_TRAJECTORIES):
    """Generate normalized trajectories from independently perturbed parameters."""
    log_params = MOCK_PARAMS + LOG_PARAMETER_STD * torch.randn(num_trajectories, 4)
    initial_states = INITIAL_STATE.expand(num_trajectories, -1).clone()
    time, history = solve_ode(
        model_func, initial_states, torch.exp(log_params), T_MAX, DT
    )
    return history[:, time % 1 == 0].flatten(1) / 100


def load_models(device):
    """Load all pretrained population-dynamics SBI models into one transfuser."""
    mtf = MTf(path=str(MODEL_DIR))
    for model_name in MODELS:
        checkpoint = MODEL_DIR / f"{model_name}.pt"
        mtf.add_model(model_name, SBIm.load(str(checkpoint), device=device))
    return mtf


def plot_comparisons_by_true_model(device):
    """Compare every true model with reverse-SDE and PF-ODE samplers."""
    mtf = load_models(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = {
        model_name: observations_for(model_func)
        for model_name, model_func in MODELS.items()
    }

    for mode in COMPARISON_MODES:
        for true_model_name in MODELS:
            print(f"Comparing {mode['description']} for true model: {true_model_name}")
            mtf.compare(
                x=observations[true_model_name],
                device=device,
                timesteps=50,
                method=mode["method"],
                order=mode["order"],
                equation=mode["equation"],
                likelihood_method="pfode",
                map_method="score",
            )

            plot_dir = (
                OUTPUT_DIR
                / mode["name"]
                / true_model_name.lower().replace(" ", "_")
            )
            plot_dir.mkdir(parents=True, exist_ok=True)
            mtf.plot_comparison(
                stats_dict=mtf.stats,
                path=str(plot_dir),
                show=False,
                sort="none",
            )
            print(f"Saved plots to {plot_dir}")


def main():
    autocvd(num_gpus=1, interval=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plot_comparisons_by_true_model(device)


if __name__ == "__main__":
    main()
