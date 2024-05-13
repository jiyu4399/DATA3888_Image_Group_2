# Import necessary packages 
from pathlib import Path
import json
import pandas as pd
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution, plot_epoch_accuracy, plot_cluster_precision, plot_cluster_recall, plot_cluster_f1
from shiny import App, Inputs, reactive, render, ui
import os

# Import models' performance 
app_dir = Path(__file__).parent
scores = pd.read_csv(app_dir / "scores.csv")

## Lab Model (3 transformation techniques)
labmodel_1_path = app_dir / "LabModel/Repeat_1_Fold_1/evaluation_metrics.json"
labmodel_2_path = app_dir / "LabModel/Repeat_1_Fold_2/evaluation_metrics.json"
labmodel_3_path = app_dir / "LabModel/Repeat_1_Fold_3/evaluation_metrics.json"
labmodel_4_path = app_dir / "LabModel/Repeat_1_Fold_4/evaluation_metrics.json"

with open(labmodel_1_path) as f:
    labmodel_1 = json.load(f)

with open(labmodel_2_path) as f:
    labmodel_2 = json.load(f)

with open(labmodel_3_path) as f:
    labmodel_3 = json.load(f)

with open(labmodel_4_path) as f:
    labmodel_4 = json.load(f)

labmodels = [labmodel_1, labmodel_2, labmodel_3, labmodel_4]

# Calculate average values for each metric
average_metrics = {
    "Accuracy": sum(labmodel["Accuracy"] for labmodel in labmodels) / len(labmodels),
    "Precision": sum(labmodel["Precision"] for labmodel in labmodels) / len(labmodels),
    "Recall": sum(labmodel["Recall"] for labmodel in labmodels) / len(labmodels),
    "F1 Score": sum(labmodel["F1 Score"] for labmodel in labmodels) / len(labmodels),
    "Adjusted Rand Index": sum(labmodel["Adjusted Rand Index"] for labmodel in labmodels) / len(labmodels)
}

# Create DataFrame from average_metrics
df = pd.DataFrame.from_records([average_metrics])


# TODO: borrow some ideas from
# https://github.com/evidentlyai/evidently
# https://medium.com/@dasulaakshat/model-monitoring-dashboard-for-models-in-production-d69f17b96f2c

# App inter
app_ui = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel(
        "Model Performance",
        ui.navset_card_underline(
            ui.nav_panel("Overall", 
                [
                    ui.navset_card_underline(
                        ui.nav_panel("Model Accuracy", ui.output_plot("accuracy_plot")),
                        ui.nav_panel("Overall performance", ui.output_data_frame("overall_table")),
                                            )
                ]
                         ),
            ui.nav_panel("By Clsuters", 
                [
                    ui.navset_card_underline(
                        ui.nav_panel("Precision per Cluster", ui.output_plot("clusters_precision_plot")),
                        ui.nav_panel("Recall per Cluster", ui.output_plot("clusters_recall_plot")),
                        ui.nav_panel("F1 Score per Cluster", ui.output_plot("clusters_f1_plot"))
                                            )
                ]
                        ),
            title="Evaluation Metrics",
        )
    ),
    ui.nav_panel(
        "Image Prediction",
        ui.layout_columns(
            ui.value_box(title="Row count", value=ui.output_text("row_count")),
            ui.value_box(
                title="Mean training score", value=ui.output_text("mean_score")
            ),
            fill=False,
        ),
        ui.card(ui.output_data_frame("data")),
        {"class": "bslib-page-dashboard"},
    ),
    sidebar=ui.sidebar(
        ui.input_select(
            "Models",
            "Models",
            choices=[
                "Basic CNN (Lab Model)",
                "ResNet18",
                "ResNet50"
            ],
        ),
        ui.input_select(
            "transformation",
            "Transformation Techniques",
            choices=[
                "Normalisation",
                "Random Flip",
                "Random Rotation",
                "Normalisation & Random Flip",
                "Normalisation & Random Rotation",
                "Normalisation & Random Flip & Random Rotation"
            ],
        ),
        ui.input_select(
            "masking",
            "Masking techniques",
            choices=[
                "No masking",
                "Cell boundary",
                "Canny contour",
                "Gaussian filter"
            ],
        )
    ),
    id="tabs",
    title="Cell Images Classification",
    fillable=True,
)



def server(input: Inputs):

    @render.data_frame
    def overall_table():
        labmodels = [labmodel_1, labmodel_2, labmodel_3, labmodel_4]
        metrics_list = []
        for i, labmodel in enumerate(labmodels, start=1):
            metrics = {
                "Iteration": i,
                "Accuracy": labmodel["Accuracy"],
                "Precision": labmodel["Precision"],
                "Recall": labmodel["Recall"],
                "F1 Score": labmodel["F1 Score"],
                "Adjusted Rand Index": labmodel["Adjusted Rand Index"]
            }
            metrics_list.append(metrics)
        df = pd.DataFrame(metrics_list)
        average_metrics = df.mean().to_frame().T
        average_metrics["Iteration"] = "Average"
        df_average = pd.concat([df, average_metrics])
        return df_average

    @render.plot
    def accuracy_plot():
        num_epochs = 70
        num_splits = 4
        num_repeats = 1
        lr_rate = 0.0001
        batch_size = 64
        script_dir = os.path.dirname(os.path.realpath("__file__"))
        folder_path = os.path.join(script_dir, "LabModel")
        plot_epoch_accuracy(folder_path, num_folds=num_splits, num_repeats=num_repeats)

    @render.plot
    def clusters_precision_plot():
        num_splits = 4
        num_repeats = 1
        script_dir = os.path.dirname(os.path.realpath("__file__"))
        folder_path = os.path.join(script_dir, "LabModel")
        plot_cluster_precision(folder_path, num_folds=num_splits, num_repeats=num_repeats)
    
    @render.plot
    def clusters_recall_plot():
        num_splits = 4
        num_repeats = 1
        script_dir = os.path.dirname(os.path.realpath("__file__"))
        folder_path = os.path.join(script_dir, "LabModel")
        plot_cluster_recall(folder_path, num_folds=num_splits, num_repeats=num_repeats)
    
    @render.plot
    def clusters_f1_plot():
        num_splits = 4
        num_repeats = 1
        script_dir = os.path.dirname(os.path.realpath("__file__"))
        folder_path = os.path.join(script_dir, "LabModel")
        plot_cluster_f1(folder_path, num_folds=num_splits, num_repeats=num_repeats)
    
    @reactive.calc()
    def dat() -> pd.DataFrame:
        return scores.loc[scores["account"] == input.account()]

    @render.plot
    def score_dist():
        return plot_score_distribution(dat())

    @render.plot
    def roc_curve():
        return plot_auc_curve(dat(), "is_electronics", "training_score")

    @render.plot
    def precision_recall():
        return plot_precision_recall_curve(dat(), "is_electronics", "training_score")

    @render.text
    def row_count():
        return dat().shape[0]

    @render.text
    def mean_score():
        return round(dat()["training_score"].mean(), 2)

    @render.data_frame
    def data():
        return dat()


app = App(app_ui, server)