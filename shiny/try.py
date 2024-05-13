# Import necessary packages 
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



from pathlib import Path
import json
import pandas as pd
from shiny import App, Inputs, reactive, render, ui


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

# TODO: borrow some ideas from
# https://github.com/evidentlyai/evidently
# https://medium.com/@dasulaakshat/model-monitoring-dashboard-for-models-in-production-d69f17b96f2c

# App inter
app_ui = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel(
        "Model Performance",
        ui.navset_card_underline(
            ui.nav_panel("Overall", ui.output_plot("overall_table")),
            ui.nav_panel("By Clsuters", ui.output_plot("clusters_plots")),
            title="Evaluation Metrics",
        ),
        ui.card(
            ui.card_header("Comments"),
            ui.output_plot("model_comments"),
        ),
        {"class": "bslib-page-dashboard"},
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
                "Random Rotation"
            ],
        ),
        ui.input_select(
            "masking",
            "Masking techniques",
            choices=[
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