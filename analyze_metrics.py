# copied from topics in cs project
# i didn't have a look at this one yet
# just in here for inspiration for how we could do our analyzing of the metrics

import os
import json
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = "models"
PERSONAS = ["survival", "combat"]
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


def load_metrics(persona):
    """Load metrics JSON for a given persona."""
    path = os.path.join(BASE_DIR, persona, "training_metrics.json")
    if not os.path.exists(path):
        print(f"[WARN] No metrics file found for {persona} persona.")
        return pd.DataFrame()
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["persona"] = persona
    return df


def plot_learning_curves(df_survival, df_combat):
    """Smoothed reward comparison over time."""
    plt.figure(figsize=(10, 5))
    for df, label in [(df_survival, "Survival"), (df_combat, "Combat")]:
        if not df.empty:
            df["reward_smoothed"] = df["total_reward"].rolling(window=10).mean()
            plt.plot(df["episode"], df["reward_smoothed"], label=f"{label} (Smoothed)")
    plt.title("Learning Curve (Smoothed Total Reward per Episode)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "learning_curve.png"))
    plt.close()


def plot_metric_histograms(df_survival, df_combat, metrics):
    """Compare distributions of key metrics between personas."""
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        if not df_survival.empty:
            plt.hist(df_survival[metric], bins=15, alpha=0.6, label="Survival")
        if not df_combat.empty:
            plt.hist(df_combat[metric], bins=15, alpha=0.6, label="Combat")
        plt.title(f"Distribution of {metric.replace('_', ' ').title()} per Episode")
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, f"hist_{metric}.png"))
        plt.close()


def plot_average_metrics(df_survival, df_combat, metrics):
    """Compare mean values of metrics per persona."""
    averages = {
        "Survival": [df_survival[m].mean() if not df_survival.empty else 0 for m in metrics],
        "Combat": [df_combat[m].mean() if not df_combat.empty else 0 for m in metrics],
    }

    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x = range(len(metrics))
    plt.bar(x, averages["Survival"], width=bar_width, label="Survival")
    plt.bar([i + bar_width for i in x], averages["Combat"], width=bar_width, label="Combat")

    plt.xticks([i + bar_width / 2 for i in x], [m.replace("_", " ").title() for m in metrics], rotation=25)
    plt.ylabel("Average Value")
    plt.title("Average Metrics per Persona")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "average_metrics.png"))
    plt.close()


def main():
    # Load data
    df_survival = load_metrics("survival")
    df_combat = load_metrics("combat")

    if df_survival.empty and df_combat.empty:
        print("[ERROR] No data available for analysis.")
        return

    # Core metrics you’re tracking in training_metrics.json
    base_metrics = ["total_reward", "average_elixir", "troops_deployed", "time_alive"]

    print("[INFO] Generating graphs...")
    plot_learning_curves(df_survival, df_combat)
    plot_metric_histograms(df_survival, df_combat, base_metrics)
    plot_average_metrics(df_survival, df_combat, base_metrics)
    print(f"[DONE] All graphs saved to '{GRAPH_DIR}/'")

if __name__ == "__main__":
    main()
