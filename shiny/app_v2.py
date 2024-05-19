import json
from shiny import App, ui, run_app, Inputs, render, reactive, Outputs, Session
from shiny.types import ImgData
import pandas as pd
from faicons import icon_svg
from shinywidgets import output_widget, render_plotly
import os
import faicons as fa
from icecream import ic
from plots import *
from helper import *
from text import DESCRIPTION

app_ui = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel(
        "CNN Models",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "model",
                    "Models",
                    choices=MODELS,
                ),
                ui.input_select(
                    "transformation",
                    "Transformation",
                    choices=[],
                ),
                ui.input_select(
                    "masking",
                    "Masking techniques",
                    choices=[],
                ),
            ),
            ui.navset_card_underline(
                ui.nav_panel(
                    "Description",
                    ui.layout_columns(
                        ui.card(
                            ui.markdown(DESCRIPTION),
                            full_screen=True,
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Overall",
                    ui.layout_column_wrap(
                        ui.value_box(
                            "Overall Accuracy",
                            ui.output_text("avg_accuracy"),
                            theme="gradient-yellow-orange",
                        ),
                        ui.value_box(
                            "Overall Precision",
                            ui.output_text("avg_precision"),
                            theme="gradient-yellow-orange",
                        ),
                        ui.value_box(
                            "Ovearall Recall",
                            ui.output_text("avg_recall"),
                            theme="gradient-yellow-orange",
                        ),
                        ui.value_box(
                            "Overall F1 Score",
                            ui.output_text("avg_f1_score"),
                            theme="gradient-yellow-orange",
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
                            ui.input_select(
                                "selected_metric",
                                "Select Metric",
                                choices=METRICS,
                            ),
                            ui.output_data_frame("overall_table"),
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.nav_panel(
                    "By Cluster",
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
        "Best Model",
        ui.navset_card_underline(
            ui.nav_panel(
                "Overall",
                ui.layout_column_wrap(
                    ui.value_box(
                        "Best Accuracy",
                        ui.div(
                            ui.output_text("comp_accuracy"),
                            style="font-size: 22px; white-space: pre-wrap; margin-top: -20px;"
                        ),
                        theme="gradient-yellow-orange",
                    ),
                    ui.value_box(
                        "Best Precision",
                        ui.div(
                            ui.output_text("comp_precision"),
                            style="font-size: 22px; white-space: pre-wrap; margin-top: -20px;"
                        ),
                        theme="gradient-yellow-orange",
                    ),
                    ui.value_box(
                        "Best Recall",
                        ui.div(
                            ui.output_text("comp_recall"),
                            style="font-size: 22px; white-space: pre-wrap; margin-top: -20px;"
                        ),
                        theme="gradient-yellow-orange",
                    ),
                    ui.value_box(
                        "Best F1 Score",
                        ui.div(
                            ui.output_text("comp_f1_score"),
                            style="font-size: 22px; white-space: pre-wrap; margin-top: -20px;"
                        ),
                        theme="gradient-yellow-orange",
                    ),
                    fill=True,
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Learning curve"),
                        ui.output_plot("comp_accuracy_plot"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header("ViT vs ResNet18"),
                        ui.input_select(
                                "selected_vit_metric",
                                "Select Metric",
                                choices=METRICS,
                        ),
                        ui.output_data_frame("Vit_ResNet18_table"),
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "By Cluster",
                ui.navset_tab_card(
                    ui.nav(
                        "Cluster Precision",
                        ui.card(
                            ui.output_plot("comp_clusters_precision_plot"),
                            full_screen=True,
                        ),
                    ),
                    ui.nav(
                        "Cluster Recall",
                        ui.card(
                            ui.output_plot("comp_clusters_recall_plot"),
                            full_screen=True,
                        ),
                    ),
                    ui.nav(
                        "Cluster F1 Score",
                        ui.card(
                            ui.output_plot("comp_clusters_f1_plot"),
                            full_screen=True,
                        ),
                    ),
                ),
            ),
            title="Evaluation Metrics",
        ),
    ),
    ui.nav_panel(
        "Image Prediction",
        ui.layout_columns(
            ui.card(
                ui.card_header("Upload Image"),
                ui.markdown("""
                    Please upload a cell image to test prediction. Images should be in PNG format.
                """),
                ui.input_file(
                    "predict_image",
                    "Upload ↓",
                    accept=["image/png"],
                    multiple=False,
                ),
                ui.output_image("uploaded_image"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Model and Techniques"),
                ui.markdown("""
                    Similar to the 'CNN Model Performance' tab, this section lets us decide the model to test prediction on using the dropdowns below. We use a combination of **Model Architecture**, **Transformation**, and **Masking Technique**. After making selection (and uploading an image), press the button below.                """),
                ui.input_select(
                    "predict_model",
                    "Models",
                    choices=MODELS,
                ),
                ui.input_select(
                    "predict_transformation",
                    "Transformation",
                    choices=[],
                ),
                ui.input_select(
                    "predict_masking",
                    "Masking techniques",
                    choices=[],
                ),
                ui.input_action_button("predict_button", "Predict"),
            ),
            ui.card(
                ui.card_header("Prediction Results"),
                ui.markdown("""
                    The format of results is a table with the cluster numbers and probabilities given by the model for each cluster. These are in descending order, which means the first entry in the table with the highest probability can be considered the final predicted cluster.
                """),
                # ui.output_text_verbatim("prediction_results", placeholder=False),
                ui.output_data_frame("prediction_results"),
            ),
            col_widths=[4, 4, 4],
        ),
        {"class": "bslib-page-dashboard"},
    ),
    id="tabs",
    title="Explore Learning",
    fillable=True,
)



def server(input: Inputs, output: Outputs, session: Session):
    ### ----------- Dynamic events -------------
        
    @reactive.Calc
    def transformations():
        model = input.model()
        match model:
            case 'Basic CNN (Lab Model)':  # LabModel
                return TRANSFORMATIONS
            case _:
                return ['Normalisation, Random Flip & Random Rotation']
            
    @reactive.Calc
    def maskings():
        model = input.model()
        transformation = input.transformation()
        match model:
            case 'Basic CNN (Lab Model)':
                match transformation:
                    case 'Normalisation, Random Flip & Random Rotation':
                        return MASKINGS
                    case _:
                        return ['No Masking']
            case 'ResNet18':
                match transformation:
                    case 'Normalisation, Random Flip & Random Rotation':
                        return MASKINGS
                    case _:
                        return ['No Masking']
            case 'ResNet50':
                match transformation:
                    case 'Normalisation, Random Flip & Random Rotation':
                        return MASKINGS
                    case _:
                        return ['No Masking']
            case _:
                return ['No Masking']


    # Update the choices for the transformation dropdown based on the model dropdown
    @reactive.Effect
    @reactive.event(input.model)
    def _update_transformation_dropdown():
        ui.update_select("transformation", choices=transformations())
        # ui.update_select("masking", choices=['No Masking'])

    @reactive.Effect
    @reactive.event(input.transformation)
    def _update_masking_dropdown():
        ui.update_select("masking", choices=maskings())


    @reactive.Calc
    def predict_transformations():
        model = input.predict_model()
        match model:
            case 'Basic CNN (Lab Model)':  # LabModel
                return TRANSFORMATIONS
            case _:
                return ['Normalisation, Random Flip & Random Rotation']
            
    @reactive.Calc
    def predict_maskings():
        model = input.predict_model()
        transformation = input.predict_transformation()
        match model:
            case 'Basic CNN (Lab Model)':
                match transformation:
                    case 'Normalisation, Random Flip & Random Rotation':
                        return MASKINGS
                    case _:
                        return ['No Masking']
            case 'ResNet18':
                match transformation:
                    case 'Normalisation, Random Flip & Random Rotation':
                        return MASKINGS
                    case _:
                        return ['No Masking']
            case 'ResNet50':
                match transformation:
                    case 'Normalisation, Random Flip & Random Rotation':
                        return MASKINGS
                    case _:
                        return ['No Masking']
            case _:
                return ['No Masking']
            

    @reactive.Effect
    @reactive.event(input.predict_model)
    def _update_predict_transformation_dropdown():
        ui.update_select("predict_transformation", choices=predict_transformations())
        # ui.update_select("masking", choices=['No Masking'])

    @reactive.Effect
    @reactive.event(input.predict_transformation)
    def _update_predict_masking_dropdown():
        ui.update_select("predict_masking", choices=predict_maskings())

    @render.image
    @reactive.event(input.predict_image)
    def uploaded_image():
        image = input.predict_image()[0]
        img: ImgData = {"src": image['datapath'], "width": "250px"}
        return img


    @render.data_frame
    @reactive.event(input.predict_button)
    def prediction_results():
        model = input.predict_model()
        transformation = input.predict_transformation()
        masking = input.predict_masking()
        image = input.predict_image()
        if not image:
            return 'Please input an image first.'
        weights_file_path = os.path.join(APP_DIR_PATH, get_directory_name(model, transformation, masking), 'model.pth')
        probabilities = predict_cluster(model, weights_file_path, image)
        results = {i+1: prob for i, prob in enumerate(probabilities)}
        results_sorted = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        df = pd.DataFrame(results_sorted.items(), columns=['Cluster', 'Probability'])
        return df
        


    ### --------------------------------------
    @render.data_frame
    def overall_table():
        model_dir = get_directory_name(input.model(), input.transformation(), input.masking())

        metrics_list = []
        for fold in FOLDS:
            with open(os.path.join(APP_DIR_PATH, model_dir, FOLD_DIR.format(fold), METRICS_FILE)) as f:
                data = json.load(f)
                metrics = {
                    "Iteration": f"{fold}",
                    "Accuracy": data["Accuracy"],
                    "Precision": data["Precision"],
                    "Recall": data["Recall"],
                    "F1 Score": data["F1 Score"],
                }
                metrics_list.append(metrics)
        df = pd.DataFrame(metrics_list)

        # Transform the DataFrame to make it vertical
        df_vertical = df.melt(
            id_vars=["Iteration"], var_name="Metric", value_name="Value"
        )

        # Filter the DataFrame based on the selected metric
        selected_metric = input.selected_metric()
        df_filtered = df_vertical[df_vertical["Metric"] == selected_metric]
        return df_filtered


    @render.data_frame
    def Vit_ResNet18_table():
        vit_dir = 'VIT_n_weightdecay'
        resnet_dir = 'ResNet18_n_nodecay'
        metric = input.selected_vit_metric()
        metrics_list = []
        for epoch in range(1, NUM_EPOCHS+1):
            vit_file = open(os.path.join(APP_DIR_PATH, vit_dir, EPOCH_DIR.format(epoch), METRICS_FILE))
            vit_data = json.load(vit_file)
            resnet_file = open(os.path.join(APP_DIR_PATH, resnet_dir, EPOCH_DIR.format(epoch), METRICS_FILE))
            resnet_data = json.load(resnet_file)
            metrics_list.append((epoch, vit_data[metric], resnet_data[metric]))
        df = pd.DataFrame(metrics_list, columns=['Epoch', 'VIT', 'ResNet18'])
        return df

    @render.plot
    def comp_accuracy_plot():
        vit_dir_path = os.path.join(APP_DIR_PATH, 'VIT_n_weightdecay')
        resnet18_dir_path = os.path.join(APP_DIR_PATH, 'ResNet18_n_nodecay')
        return plot_epoch_accuracy_for_two_models(
            vit_dir_path,
            resnet18_dir_path,
        )

    @render.plot
    def accuracy_plot():
        folder_path = os.path.join(APP_DIR_PATH, get_directory_name(input.model(), input.transformation(), input.masking()))
        return plot_epoch_accuracy(
            folder_path, num_folds=NUM_FOLDS, num_repeats=NUM_REPEATS
        )

    @reactive.Calc
    def avg_metrics():
        model_dir = get_directory_name(input.model(), input.transformation(), input.masking())
        fold_models = [json.load(open(os.path.join(APP_DIR_PATH, model_dir, FOLD_DIR.format(fold), METRICS_FILE))) for fold in FOLDS]
        return {
            "Accuracy": round(
                sum(model["Accuracy"] for model in fold_models) / len(fold_models), 4
            ),
            "Precision": round(
                sum(model["Precision"] for model in fold_models) / len(fold_models), 4
            ),
            "Recall": round(
                sum(model["Recall"] for model in fold_models) / len(fold_models), 4
            ),
            "F1 Score": round(
                sum(model["F1 Score"] for model in fold_models) / len(fold_models), 4
            ),
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


    @reactive.Calc
    def comp_metrics():
        vit_dir = 'VIT_n_weightdecay'
        resnet_dir = 'ResNet18_n_nodecay'
        with open(os.path.join(APP_DIR_PATH, vit_dir, 'final_model', f'final_{METRICS_FILE}')) as f:
            vit_data = json.load(f)
        with open(os.path.join(APP_DIR_PATH, resnet_dir, 'final_model', f'final_{METRICS_FILE}')) as f:
            resnet_data = json.load(f)

        return {
            "Accuracy": (round(vit_data['Accuracy'], 4), round(resnet_data['Accuracy'], 4)),
            "Precision": (round(vit_data['Precision'], 4), round(resnet_data['Precision'], 4)),
            "Recall": (round(vit_data['Recall'], 4), round(resnet_data['Recall'], 4)),
            "F1 Score": (round(vit_data['F1 Score'], 4), round(resnet_data['F1 Score'], 4)),
        }
    
    @render.text
    def comp_accuracy():
        return f"VIT: {comp_metrics()['Accuracy'][0]}\nResNet18: {comp_metrics()['Accuracy'][1]}"

    @render.text
    def comp_precision():
        return f"VIT: {comp_metrics()['Precision'][0]}\nResNet18: {comp_metrics()['Precision'][1]}"

    @render.text
    def comp_recall():
        return f"VIT: {comp_metrics()['Recall'][0]}\nResNet18: {comp_metrics()['Recall'][1]}"

    @render.text
    def comp_f1_score():
        return f"VIT: {comp_metrics()['F1 Score'][0]}\nResNet18: {comp_metrics()['F1 Score'][1]}"

    @render.plot
    def clusters_precision_plot():
        num_splits = 4
        num_repeats = 1
        folder_path = folder_path = os.path.join(APP_DIR_PATH, get_directory_name(input.model(), input.transformation(), input.masking()))
        plot_cluster_precision(
            folder_path, num_folds=num_splits, num_repeats=num_repeats
        )

    @render.plot
    def clusters_recall_plot():
        num_splits = 4
        num_repeats = 1
        folder_path = folder_path = os.path.join(APP_DIR_PATH, get_directory_name(input.model(), input.transformation(), input.masking()))
        plot_cluster_recall(folder_path, num_folds=num_splits, num_repeats=num_repeats)

    @render.plot
    def clusters_f1_plot():
        num_splits = 4
        num_repeats = 1
        folder_path = folder_path = os.path.join(APP_DIR_PATH, get_directory_name(input.model(), input.transformation(), input.masking()))
        plot_cluster_f1(folder_path, num_folds=num_splits, num_repeats=num_repeats)

    @render.plot
    def comp_clusters_precision_plot():
        vit_dir_path = os.path.join(APP_DIR_PATH, 'VIT_n_weightdecay')
        resnet18_dir_path = os.path.join(APP_DIR_PATH, 'ResNet18_n_nodecay')
        return plot_cluster_metric_for_two_models(vit_dir_path, resnet18_dir_path, "Precision by cluster", "Average Precision by Cluster", "Precision")

    @render.plot
    def comp_clusters_recall_plot():
        vit_dir_path = os.path.join(APP_DIR_PATH, 'VIT_n_weightdecay')
        resnet18_dir_path = os.path.join(APP_DIR_PATH, 'ResNet18_n_nodecay')
        return plot_cluster_metric_for_two_models(vit_dir_path, resnet18_dir_path, "Recall by cluster", "Average Recall by Cluster", "Recall")

    @render.plot
    def comp_clusters_f1_plot():
        vit_dir_path = os.path.join(APP_DIR_PATH, 'VIT_n_weightdecay')
        resnet18_dir_path = os.path.join(APP_DIR_PATH, 'ResNet18_n_nodecay')
        return plot_cluster_metric_for_two_models(vit_dir_path, resnet18_dir_path, "F1 score by cluster", "Average F1 score by Cluster", "F1 Score")

app = App(app_ui, server)
if __name__ == "__main__":
    run_app("app_v2")

