import matplotlib.pyplot as plt
import numpy as np


def simulate_many_paths(principal, mu, sigma, years, num_simulations):
    dt = 1 / 12
    num_months = years * 12

    # Vectorized generation of log-normal paths
    Z = np.random.standard_normal((num_simulations, num_months))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion

    cum_returns = np.column_stack(
        [np.zeros(num_simulations), np.cumsum(log_returns, axis=1)]
    )
    return principal * np.exp(cum_returns)


def summarize(all_paths, principal):
    final_values = all_paths[:, -1]
    return {
        "median": np.median(final_values),
        "p5": np.percentile(final_values, 5),
        "p95": np.percentile(final_values, 95),
        "prob_loss": np.mean(final_values < principal),
    }


def risk_metrics(all_paths, principal, confidence=0.95):
    final_values = all_paths[:, -1]
    losses = principal - final_values  # positive = loss, negative = gain

    var = np.percentile(losses, confidence * 100)
    tail_losses = losses[losses >= var]
    cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

    return {
        "VaR": var,
        "CVaR": cvar,
    }


def stress_test(principal, years, num_simulations, base_params, stress_scenarios):
    results = {}
    paths_dict = {}
    all_scenarios = {"Base case": base_params, **stress_scenarios}

    for name, (mu, sigma) in all_scenarios.items():
        paths = simulate_many_paths(principal, mu, sigma, years, num_simulations)
        stats = summarize(paths, principal)
        risk = risk_metrics(paths, principal)

        results[name] = {**stats, **risk}
        paths_dict[name] = paths

    return results, paths_dict


def plot_fan_chart(all_scenario_paths, years):
    num_months = years * 12
    x_years = np.arange(num_months + 1) / 12

    plt.figure(figsize=(11, 6))

    # Overlay each scenario on the same plot
    for name, paths in all_scenario_paths.items():
        p10 = np.percentile(paths, 10, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        # Plot median line and capture color for matching fill
        line = plt.plot(x_years, p50, label=f"{name} (Median)", linewidth=2)
        color = line[0].get_color()
        plt.fill_between(x_years, p10, p90, color=color, alpha=0.15)

    plt.xlabel("Years")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Monte Carlo Stress Test: Scenario Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    principal = 100_000
    mu = 0.07
    sigma = 0.15
    years = 20
    num_simulations = 100

    scenarios = {
        "Mild recession": (0.02, 0.20),
        "Severe crash": (-0.10, 0.35),
        "Stagflation": (0.00, 0.25),
    }

    # Run base case and stress test scenarios simultaneously
    results, scenario_paths = stress_test(
        principal, years, num_simulations, (mu, sigma), scenarios
    )

    # Print results summary
    print("Stress test results:")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Median ending value: ${r['median']:,.0f}")
        print(f"  5th percentile (downside): ${r['p5']:,.0f}")
        print(f"  95th percentile (upside): ${r['p95']:,.0f}")
        print(f"  VaR (95% confidence): ${r['VaR']:,.0f}")
        print(f"  CVaR (95% confidence): ${r['CVaR']:,.0f}")
        print(f"  Probability of loss: {r['prob_loss']:.1%}")

    # Plot all scenarios on a single chart
    plot_fan_chart(scenario_paths, years)
