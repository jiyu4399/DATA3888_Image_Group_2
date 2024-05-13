from pandas import DataFrame
from plotnine import (
    aes,
    geom_abline,
    geom_density,
    geom_line,
    ggplot,
    labs,
    theme_minimal,
)
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_score_distribution(df: DataFrame):
    plot = (
        ggplot(df, aes(x="training_score"))
        + geom_density(fill="blue", alpha=0.3)
        + theme_minimal()
        + labs(title="Model scores", x="Score")
    )
    return plot


def plot_auc_curve(df: DataFrame, true_col: str, pred_col: str):
    fpr, tpr, _ = roc_curve(df[true_col], df[pred_col])
    roc_auc = auc(fpr, tpr)

    roc_df = DataFrame({"fpr": fpr, "tpr": tpr})

    plot = (
        ggplot(roc_df, aes(x="fpr", y="tpr"))
        + geom_line(color="darkorange", size=1.5, show_legend=True, linetype="solid")
        + geom_abline(intercept=0, slope=1, color="navy", linetype="dashed")
        + labs(
            title="Receiver Operating Characteristic (ROC)",
            subtitle=f"AUC: {roc_auc.round(2)}",
            x="False Positive Rate",
            y="True Positive Rate",
        )
        + theme_minimal()
    )

    return plot


def plot_precision_recall_curve(df: DataFrame, true_col: str, pred_col: str):
    precision, recall, _ = precision_recall_curve(df[true_col], df[pred_col])

    pr_df = DataFrame({"precision": precision, "recall": recall})

    plot = (
        ggplot(pr_df, aes(x="recall", y="precision"))
        + geom_line(color="darkorange", size=1.5, show_legend=True, linetype="solid")
        + labs(
            title="Precision-Recall Curve",
            x="Recall",
            y="Precision",
        )
        + theme_minimal()
    )

    return plot


def load_metrics(json_filepath):
    """Load metrics from a JSON file."""
    try:
        with open(json_filepath, "r") as file:
            data = json.load(file)
        print(f"Loaded data from {json_filepath}")
        return data
    except FileNotFoundError:
        print(f"File not found: {json_filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {json_filepath}")
        return None

def visualize_metrics(metrics, title, ylabel):
    bars = plt.bar(
        range(len(metrics)),
        metrics,
        color=plt.cm.viridis(np.linspace(0, 1, len(metrics))),
    )
    plt.title(title)
    plt.xlabel("Cluster Index")
    plt.ylabel(ylabel)
    plt.xticks(range(len(metrics)))
    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va="bottom"
        )  # va: vertical alignment

def visualize_epoch_accuracies(average_accuracies):
    #plt.figure(figsize=(5, 7))

    # Use a color map and select colors from different parts of the map for training and validation
    color_map = plt.cm.get_cmap(
        "viridis", 128
    )  # Get a color map with enough variations
    training_colors = [
        color_map(i) for i in range(30, 50)
    ]  # Select a range of colors from the color map
    validation_colors = [
        color_map(i) for i in range(70, 90)
    ]  # Different range from the same color map

    training_marker = "o"  # Circle marker for all training lines
    validation_marker = "s"  # Square marker for all validation lines

    epochs = range(1, len(average_accuracies[0]) + 1)

    # Averages with more distinctive lines using darker shades
    plt.plot(
        epochs,
        average_accuracies[0],
        "k-",
        color="orange",
        label="Average Training Accuracy",
        linewidth=2,
        marker=None,
        alpha=0.9,
    )
    plt.plot(
        epochs,
        average_accuracies[1],
        "r-",
        label="Average Validation Accuracy",
        linewidth=2,
        marker=None,
        alpha=0.9,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Average Training and Validation Accuracy per Epoch")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_epoch_accuracy(folder_path, num_folds=5, num_repeats=3):
    """Load metrics from multiple folds and repeats, and visualize them."""
    all_epoch_accuracies = []
    for fold in range(1, num_folds + 1):
        for repeat in range(1, num_repeats + 1):
            json_filepath = os.path.join(
                folder_path, f"Repeat_{repeat}_Fold_{fold}", "evaluation_metrics.json"
            )
            data = load_metrics(json_filepath)
            if data:
                all_epoch_accuracies.append(data["epoch_accuracies"])
    # Calculate average accuracy
    avg_epoch_accuracies = (
        np.mean(np.array(all_epoch_accuracies), axis=0) if all_epoch_accuracies else []
    )
    # Visualisation 
    if all_epoch_accuracies:
        visualize_epoch_accuracies(avg_epoch_accuracies)

def plot_cluster_precision(folder_path, num_folds=5, num_repeats=3):
    """Load metrics from multiple folds and repeats, and visualize them."""
    all_precision = []
    for fold in range(1, num_folds + 1):
        for repeat in range(1, num_repeats + 1):
            json_filepath = os.path.join(
                folder_path, f"Repeat_{repeat}_Fold_{fold}", "evaluation_metrics.json"
            )
            data = load_metrics(json_filepath)
            if data:
                all_precision.append(data["Precision by cluster"])
    # Calculate average metrics
    avg_precision = np.mean(all_precision, axis=0) if all_precision else None
    # Visualization
    if avg_precision is not None:
        visualize_metrics(avg_precision, "Average Precision by Cluster", "Precision")

def plot_cluster_recall(folder_path, num_folds=5, num_repeats=3):
    """Load metrics from multiple folds and repeats, and visualize them."""
    all_recall = []
    for fold in range(1, num_folds + 1):
        for repeat in range(1, num_repeats + 1):
            json_filepath = os.path.join(
                folder_path, f"Repeat_{repeat}_Fold_{fold}", "evaluation_metrics.json"
            )
            data = load_metrics(json_filepath)
            if data:
                all_recall.append(data["Recall by cluster"])
    # Calculate average metrics
    avg_recall = np.mean(all_recall, axis=0) if all_recall else None
    # Visualization
    if avg_recall is not None:
        visualize_metrics(avg_recall, "Average Recall by Cluster", "Recall")


def plot_cluster_f1(folder_path, num_folds=5, num_repeats=3):
    """Load metrics from multiple folds and repeats, and visualize them."""
    all_f1_scores = []
    for fold in range(1, num_folds + 1):
        for repeat in range(1, num_repeats + 1):
            json_filepath = os.path.join(
                folder_path, f"Repeat_{repeat}_Fold_{fold}", "evaluation_metrics.json"
            )
            data = load_metrics(json_filepath)
            if data:
                all_f1_scores.append(data["F1 score by cluster"])
    # Calculate average metrics
    avg_f1_scores = np.mean(all_f1_scores, axis=0) if all_f1_scores else None
    # Visualization
    if avg_f1_scores is not None:
        visualize_metrics(avg_f1_scores, "Average F1 Score by Cluster", "F1 Score")
 
