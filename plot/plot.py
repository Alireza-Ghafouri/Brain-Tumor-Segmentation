import matplotlib.pyplot as plt
import os


def plot_metrics(metrics, ylabel, title, model_type, save_path=None):
    """
    Plots training and validation metrics.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(range(1, len(metrics["train"]) + 1), metrics["train"], label="Train")
    ax.plot(range(1, len(metrics["val"]) + 1), metrics["val"], "r", label="Validation")
    ax.legend()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if save_path:
        plt.savefig(os.path.join(save_path, model_type + " " + title))
    plt.show()
