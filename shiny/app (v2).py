from pathlib import Path
import json
from shiny import App, ui, run_app, Inputs, render, reactive
import pandas as pd
from faicons import icon_svg
from shinywidgets import output_widget, render_plotly
import os
import faicons as fa
from plots import plot_epoch_accuracy, plot_cluster_precision, plot_cluster_recall, plot_cluster_f1

app_dir = Path(__file__).parent
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
    "F1 Score": sum(labmodel["F1 Score"] for labmodel in labmodels) / len(labmodels)
}

# Create DataFrame from average_metrics
df = pd.DataFrame.from_records([average_metrics])





app_ui = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel(
        "Model Performance",
        ui.layout_sidebar(
            ui.sidebar(
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
                    "Transformation",
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
            ui.navset_card_underline(
                ui.nav_panel("Description",
                    ui.layout_columns(
                        ui.card(
                            ui.markdown("""
                                ## Application Description
                                This application provides an interface to explore stock prices and 
                                performance metrics by cluster. You can navigate through different tabs 
                                to see overall performance and detailed cluster-specific metrics.
                                
                                - **Overall**: View overall stock price information and history.
                                - **By Cluster**: Analyze precision, recall, and F1 score for different clusters.
                                - **Description**: Read about the application and how to use it.
                            """),
                            full_screen=True,
                        ),
                    ),
                ),
                ui.nav_panel("Overall",
                    ui.layout_column_wrap(
                        ui.value_box(
                            "Overall Accuracy",
                            ui.output_text("avg_accuracy"),
                            theme= "gradient-yellow-orange",
                        ),
                        ui.value_box(
                            "Overall Precision",
                            ui.output_text("avg_precision"),
                            theme= "gradient-yellow-orange",
                        ),
                        ui.value_box(
                            "Overall Recall",
                            ui.output_text("avg_recall"),
                            theme= "gradient-yellow-orange",
                        ),
                        ui.value_box(
                            "Overall F1 Score",
                            ui.output_text("avg_f1_score"),
                            theme= "gradient-yellow-orange",
                        ),
                        fill=True,
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Learning curve"),
                            ui.output_plot("accuracy_plot"),
                            full_screen=True,
                        ),
                        ui.card(
                            ui.card_header("Evaluation by cross validation"),
                            ui.input_select("selected_metric", "Select Metric", choices=["Accuracy", "Precision", "Recall", "F1 Score"]),
                            ui.output_data_frame("overall_table")
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.nav_panel("By Cluster",
                    ui.navset_tab_card(
                        ui.nav(
                            "Cluster Precision",
                            ui.card(
                                ui.output_plot("clusters_precision_plot"),
                                full_screen=True,
                            ),
                        ),
                        ui.nav(
                            "Cluster Recall",
                            ui.card(
                                ui.output_plot("clusters_recall_plot"),
                                full_screen=True,
                            ),
                        ),
                        ui.nav(
                            "Cluster F1 Score",
                            ui.card(
                                ui.output_plot("clusters_f1_plot"),
                                full_screen=True,
                            ),
                        ),
                    ),
                ),
                title="Evaluation Metrics",
            ),
        ),
    ),
    ui.nav_panel(
        "Image Prediction",
        ui.layout_columns(
            ui.card(
                ui.card_header("Upload Image"),
                ui.input_file("file1", "Choose an image to upload", accept=["image/png"], multiple=False),
                ui.output_image("uploaded_image"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Model and Techniques"),
                ui.input_select(
                    "predict_model",
                    "Models",
                    choices=[
                        "Basic CNN (Lab Model)",
                        "ResNet18",
                        "ResNet50"
                    ],
                ),
                ui.input_select(
                    "predict_transformation",
                    "Transformation",
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
                    "predict_masking",
                    "Masking techniques",
                    choices=[
                        "No masking",
                        "Cell boundary",
                        "Canny contour",
                        "Gaussian filter"
                    ],
                ),
                ui.input_action_button("predict", "Predict"),
            ),
            ui.card(
                ui.card_header("Prediction Results"),
                ui.output_text_verbatim("prediction_results"),

            ),
            col_widths=[4, 4, 4]
        ),
        {"class": "bslib-page-dashboard"},
    ),
    id="tabs",
    title="Image2 - Shiny App",
    fillable=True,
)




    

def server(input: Inputs):
    @render.data_frame
    def overall_table():
        labmodels = [labmodel_1, labmodel_2, labmodel_3, labmodel_4]
        metrics_list = []
        for i, labmodel in enumerate(labmodels, start=1):
            metrics = {
                "Iteration":  f"Folder {i}",
                "Accuracy": labmodel["Accuracy"],
                "Precision": labmodel["Precision"],
                "Recall": labmodel["Recall"],
                "F1 Score": labmodel["F1 Score"]
            }
            metrics_list.append(metrics)
        df = pd.DataFrame(metrics_list)


        # Transform the DataFrame to make it vertical
        df_vertical = df.melt(id_vars=["Iteration"], var_name="Metric", value_name="Value")

        # Filter the DataFrame based on the selected metric
        selected_metric = input.selected_metric()
        df_filtered = df_vertical[df_vertical["Metric"] == selected_metric]
        return df_filtered
    
    @render.plot
    def accuracy_plot():
        num_splits = 4
        num_repeats = 1
        script_dir = os.path.dirname(os.path.realpath("__file__"))
        folder_path = os.path.join(script_dir, "LabModel")
        return plot_epoch_accuracy(folder_path, num_folds=num_splits, num_repeats=num_repeats)
    

    @reactive.Calc
    def avg_metrics():
        return {
            "Accuracy": round(sum(labmodel["Accuracy"] for labmodel in labmodels) / len(labmodels), 4),
            "Precision": round(sum(labmodel["Precision"] for labmodel in labmodels) / len(labmodels), 4),
            "Recall": round(sum(labmodel["Recall"] for labmodel in labmodels) / len(labmodels), 4),
            "F1 Score": round(sum(labmodel["F1 Score"] for labmodel in labmodels) / len(labmodels), 4),
        }

    @render.text
    def avg_accuracy():
        return f"{avg_metrics()['Accuracy']}"

    @render.text
    def avg_precision():
        return f"{avg_metrics()['Precision']}"

    @render.text
    def avg_recall():
        return f"{avg_metrics()['Recall']}"

    @render.text
    def avg_f1_score():
        return f"{avg_metrics()['F1 Score']}"
    
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



app = App(app_ui, server)
if __name__ == "__main__":
    run_app("try")