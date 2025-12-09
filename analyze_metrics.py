import os
import json
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = "models"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

LINEAR_FILE = os.path.join(MODELS_DIR, "training_DQN_LINEAR_metrics.json")
CONV_FILE = os.path.join(MODELS_DIR, "training_DQN_CONV_metrics.json")


def load_df(path, label):
    if not os.path.exists(path):
        print(f"[WARN] metrics missing: {path}")
        return pd.DataFrame()
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["model"] = label
    return df


def plot_epsilon(df_lin, df_conv):
    plt.figure(figsize=(10,5))
    if not df_lin.empty:
        plt.plot(df_lin["episode"], df_lin["epsilon"], label="Linear", color="blue")
    if not df_conv.empty:
        plt.plot(df_conv["episode"], df_conv["epsilon"], label="Conv", color="red")

    plt.title("Epsilon over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "epsilon_curve.png"))
    plt.close()
    print("Saved epsilon_curve.png")


def plot_reward(df_lin, df_conv):
    plt.figure(figsize=(10,5))
    if not df_lin.empty:
        plt.plot(df_lin["episode"], df_lin["total_reward"], label="Linear", color="blue")
    if not df_conv.empty:
        plt.plot(df_conv["episode"], df_conv["total_reward"], label="Conv", color="red")

    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "reward_curve.png"))
    plt.close()
    print("Saved reward_curve.png")


def plot_x_global(df_lin, df_conv):
    plt.figure(figsize=(10,5))

    if "x_global" in df_lin.columns:
        plt.plot(df_lin["episode"], df_lin["x_global"], label="Linear", color="blue")

    if "x_global" in df_conv.columns:
        plt.plot(df_conv["episode"], df_conv["x_global"], label="Conv", color="red")

    plt.title("X-Position Progress over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("x_global")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "x_global_curve.png"))
    plt.close()
    print("Saved x_global_curve.png")


def main():
    df_lin = load_df(LINEAR_FILE, "linear")
    df_conv = load_df(CONV_FILE, "conv")

    if df_lin.empty and df_conv.empty:
        print("[ERROR] No metrics found.")
        return

    print("[INFO] Creating graphs...")
    plot_epsilon(df_lin, df_conv)
    plot_reward(df_lin, df_conv)
    plot_x_global(df_lin, df_conv)
    print(f"[DONE] Graphs saved to {GRAPH_DIR}/")


if __name__ == "__main__":
    main()
