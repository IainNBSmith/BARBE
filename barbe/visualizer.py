import pandas as pd
from shiny import reactive, App, render, run_app, ui
from htmltools import css

from pathlib import Path
# from shiny.express import ui
from barbe.utils.visualizer_utils import *
import barbe
import pickle
import torch
import re

# TODO/DONE: IAIN add error checking before using a dataset (error should give popup)
# TODO: add a checkbox for scale being row 0 of the given csv
# TODO: add option to select particular values from a simple file (not just 3 lines)
    # TODO/DONE Aug 14: add modifications to max and min values
    # TODO: add the ability to save all information from values
# TODO: add generic error handling (Some done today - Aug 13)
# TODO: clean up this code
# TODO/Done Aug 20: make the numeric values display only 3-4 sigdigits
# TODO: add a setting to get the prediction from the surrogate?

# TODO/Done Aug 20: make the numeric values have to be whole integers
# TODO/Done Aug 20: Add manual scale values
# TODO/Done Aug 20 (implicitly): Add selection for values to change
# TODO: Add new values after doing counterfactual

app_ui = ui.page_fluid(ui.tags.link(rel='stylesheet', href='styles.css'),
                 ui.layout_columns(ui.card(ui.input_file('data_file_name', 'Choose Data File', accept=[".csv", ".data"]),
                                                 ui.input_checkbox('data_file_option', 'User Set Ranges'),
                                                 ui.input_checkbox('data_perturbations', 'Replace Ranges with Perturbations')),
                                               ui.div(ui.input_checkbox('data_save', 'Save Ranges and Scale'),
                                                       ui.input_checkbox('explainer_save', 'Save Explainer'),
                                                       ui.input_action_button('save_data', 'Save', disabled=True), id="save_menu"),
                                               ui.card(ui.input_file('predictor_file_name', 'Choose Black Box Model', accept=[".pickle", ".torch", ".pkl"]))),
                       ui.layout_sidebar(ui.sidebar(
                                                    ui.div(id="slider_container"),
                                                    ui.input_checkbox('manual_data', 'Manually Set Values'),
                                                    ui.output_text('prediction'),
                                                    ui.input_action_button('data_info', 'Predict'), id='sidebar',
                                                    width=400, title="Data", open="closed"),
                                         # IAIN add more information from the explainer besider the plot (in a card)
                                        ui.layout_columns(
                                         ui.output_plot('explanation_plot'),
                                            ui.card(ui.card_header("BARBE Explainer"),
                                                    ui.input_select('dist_setting', 'Distribution for Perturbations',
                                                                    barbe.explainer.DIST_INFO),
                                                    ui.input_numeric('pert_setting', 'Number of Perturbations', 5000,
                                                                     min=1000, max=10000, step=1000),
                                                    ui.input_numeric('dev_setting', 'Deviation Scaling', 5, min=1,
                                                                     max=100, step=1),
                                                    ui.input_numeric('category_setting', 'Bins for Continuous Data', 5, min=2,
                                                                     max=20, step=1),
                                                    ui.input_checkbox('set_class_setting', 'Use All Classes'),
                                                    ui.output_text('info'),
                                                    ui.input_action_button('data_explain', 'Explain'), max_height="600px")
                                        ),
                                         ui.layout_columns(ui.output_table('explanation_rules'),
                                                    ui.card(ui.card_header("BARBE Counterfactual"),
                                                    ui.input_select('class_setting', 'Class Change',
                                                                    barbe.explainer.DIST_INFO),
                                                    ui.input_action_button('data_counterfactual', 'Counterfactual', disabled=True), max_height="225px"),
                                                           col_widths=(8, 4)
                                                           ))
                                         )

def server(input, output, session):
    server_data = reactive.value(None)
    server_features = reactive.value(None)
    server_ranges = reactive.value(None)
    server_types = reactive.value(None)
    server_scale = reactive.value(None)
    server_categories = reactive.value(None)

    server_predictor = reactive.value(None)
    server_explainer = reactive.value(None)
    server_prediction = reactive.value(None)
    server_rules = reactive.value(None)
    server_plot = reactive.value(None)

    active_pert = reactive.value(False)

    def reset_data():
        prev_value = list()
        prev_scale = list()
        if server_data.get() is not None:
            if server_ranges.get() is not None:  # remove existing data
                for i in range(len(server_ranges.get())):
                    prev_value.append(input['in_data_'+str(i)]())
                    prev_scale.append(input['in_scale_'+str(i)]())
                    ui.remove_ui(selector="div:has(> #in_data_" + str(i) + ")")
                    ui.remove_ui(selector="div:has(> #in_scale_" + str(i) + ")")

            data = server_data.get()
            features = server_features.get()
            ranges = produce_ranges(data)
            server_ranges.set(ranges)
            scales = server_scale.get()
            #ui.notification_show("Set all values")
            # for each feature add a slider or selection list to modify
            default_value = data.iloc[0]
            for i in range(len(ranges) - 1, -1, -1):
                # IAIN add the possibility for categorical values (dropdown)
                if input.data_file_option():
                    ui.insert_ui(
                        ui.input_numeric('in_scale_' + str(i), 'Scale', value=prev_scale[i], min=0, max=10 * scales[i]),
                        # id string of object relative to placement
                        selector="#slider_container",
                        where="afterBegin")
                else:
                    ui.insert_ui(
                        ui.input_numeric('in_scale_' + str(i), 'Scale - locked', value=scales[i], min=scales[i],
                                         max=scales[i]),
                        # id string of object relative to placement
                        selector="#slider_container",
                        where="afterBegin")

                if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                    if not input.manual_data():
                        ui.insert_ui(ui.input_slider('in_data_' + str(i),
                                                                       str(features[i]),
                                                                       value=prev_value[i],
                                                                       min=ranges[i][0],
                                                                       max=ranges[i][1],
                                                                       sep='', width="80%"),
                                     # id string of object relative to placement
                                     selector="#slider_container",
                                     where="afterBegin"
                                     )
                    else:
                        ui.insert_ui(ui.input_numeric('in_data_' + str(i),
                                                                        str(features[i]),
                                                                        prev_value[i],
                                                                        min=ranges[i][0],
                                                                        max=ranges[i][1], width="80%"),
                                     # id string of object relative to placement
                                     selector="#slider_container",
                                     where="afterBegin"
                                     )
                else:
                    ui.insert_ui(ui.input_select('in_data_' + str(i),
                                                                   str(features[i]),
                                                                   list(ranges[i]),
                                                                   selected=str(prev_value[i]),
                                                                   width="80%"),
                                 # id string of object relative to placement
                                 selector="#slider_container",
                                 where="afterBegin")

    @render.text
    def prediction():
        prediction = server_prediction.get()
        if prediction is None:
            return 'Prediction: No prediction yet!'
        return 'Prediction: ' + str(prediction)

    @render.plot
    def explanation_plot():
        return server_plot.get()

    @render.text
    def info():
        explainer = server_explainer.get()
        if explainer is not None:
            return "Fidelity:" + str(explainer.get_surrogate_fidelity())
        return None

    @render.table
    def explanation_rules():
        return server_rules.get()

    @reactive.effect
    @reactive.event(input.data_info)
    def _():
        features = server_features.get()
        predict_input = server_data.get()
        predictor = server_predictor.get()
        if predict_input is not None and features is not None and predictor is not None:
            # get the explanation rules
            reformat_input = pd.DataFrame(columns=server_features.get(), index=[0])
            for i in range(len(features)):
                # IAIN how to get dynamic names
                reformat_input.iloc[0][features[i]] = input['in_data_'+str(i)]()
            # save and use the input in barbe
            # IAIN now it works
            if predictor.check_valid_data(reformat_input):
                prediction = predictor.predict(reformat_input)
                server_prediction.set(prediction[0])
                #ui.notification_show(predictor.predict(server_data.get()), duration=100)
                ui.notification_show("Predicted: " + str(prediction[0]) + " for current sample.")
            else:
                ui.notification_show("Model Error: Invalid data format for selected model. "
                                     "Try a different model or format.")
        else:
            missing_model = 'black-box' if predictor is None else ''
            missing_data = 'data' if predict_input is None else ''
            ui.notification_show("Error: Element(s) " + missing_model + " " +
                                 missing_data + " is missing.")

    @reactive.effect
    @reactive.event(input.data_explain)
    def _():
        features = server_features.get()
        predict_input = server_data.get()
        predictor = server_predictor.get()
        if predict_input is not None and features is not None and predictor is not None:
            # get the explanation rules
            reformat_input = pd.DataFrame(columns=server_features.get(), index=[0])
            for i in range(len(features)):
                # IAIN how to get dynamic names
                reformat_input.iloc[0][features[i]] = input['in_data_' + str(i)]()
            # save and use the input in barbe
            # IAIN now it works
            if predictor.check_valid_data(reformat_input):
                prediction = predictor.predict(reformat_input)
                server_prediction.set(prediction[0])
                ui.notification_show("Predicted: " + str(prediction[0]) + " for current sample.")

                # now run the explainer
                settings = {'perturbation_type': input.dist_setting(),
                            'dev_scaling_factor': input.dev_setting(),
                            'input_sets_class': not input.set_class_setting(),
                            'n_perturbations': input.pert_setting(),
                            'n_bins': input.category_setting()}
                check_values = check_settings(settings)
                #ui.notification_show("Checked Settings")
                if check_values is not None:
                    ui.notification_show("Error: settings " + check_values + " must all be whole numbers.")
                else:
                    ui.notification_show("Starting BARBE")
                    scales = [input['in_scale_' + str(i)]() + 1e-10 for i in range(len(server_ranges.get()))]
                    ui.notification_show(str(scales))
                    ui.notification_show(str(list(features)))
                    ui.notification_show(str(server_categories.get()))
                    explainer, explanation = fit_barbe_explainer(scales, features, server_categories.get(),
                                                                 reformat_input.iloc[0], predictor,
                                                    input.data_file_option(), settings=settings)
                    ui.notification_show(explanation)
                    if explanation is not None and len(explanation) > 0:
                        ui.notification_show(str(explanation))
                        server_plot.set(feature_importance_barplot(explanation))
                        ui.notification_show("Produced BARBE plots.")
                        server_rules.set(barbe_rules_table(explainer.get_rules()))
                        ui.notification_show("Retrieved BARBE rules.")
                        server_explainer.set(explainer)

                        if input.data_perturbation():
                            server_data.set(explainer.get_perturbed_data())
                            reset_data()
                    else:
                        ui.notification_show("Error: Mismatched sample input and BARBE prediction, try again or change sample.")
                        try:
                            aa = reformat_input.iloc[0].to_numpy().reshape(1, -1)
                            ui.notification_show(str(aa))
                            ui.notification_show(
                                explainer._surrogate_model.predict(aa)[0])
                        except Exception as e:
                            ui.notification_show(str(e))

            else:
                ui.notification_show("Model Error: Invalid data format for selected model. "
                                     "Try a different model or format.")
        else:
            missing_model = 'black-box' if predictor is None else ''
            missing_data = 'data' if predict_input is None else ''
            ui.notification_show("Error: Element(s) " + missing_model + " " +
                                 missing_data + " is missing.")


    @reactive.effect
    @reactive.event(input.predictor_file_name)
    def _():
        server_prediction.set(None)
        # open predictor pickle, use later by passing to BARBE
        input_model = open_input_model(input.predictor_file_name()[0]["datapath"],
                                       input.predictor_file_name()[0]["name"])
        server_predictor.set(input_model)
        pass


    @reactive.effect
    @reactive.event(input.manual_data)
    def _():
        reset_data()

    # IAIN add that counterfactual is innactive until a explanation is produced (changing the loaded data or model reverts this)
    @reactive.effect
    @reactive.event(input.data_perturbations)
    def _():
        if not input.data_perturbations():
            if active_pert.get():
                ui.notification_show("Data may need to be reloaded to reset option.")
            else:
                active_pert.set(True)
        else:
            reset_data()

    @reactive.effect
    @reactive.event(input.data_file_option)
    def _():
        if input.data_file_option():
            ui.update_numeric('dev_setting', label='Deviation Scaling - locked to 1', value=1, min=1, max=1)
            if server_ranges.get() is not None:
                scales = server_scale.get()
                for i in range(len(server_ranges.get())):
                    ui.update_numeric('in_scale_' + str(i), label='Scale', value=scales[i], min=0, max=10*scales[i])
        else:
            ui.update_numeric('dev_setting', label='Deviation Scaling', value=5, min=1, max=100)
            if server_ranges.get() is not None:
                scales = server_scale.get()
                for i in range(len(server_ranges.get())):
                    ui.update_numeric('in_scale_' + str(i), label='Scale - locked', value=scales[i], min=scales[i], max=scales[i])

    @reactive.effect
    @reactive.event(input.data_file_name)
    def _():
        server_prediction.set(None)
        # read in elements from the file
        data, features, types, ranges, scales, categories = open_input_file(input.data_file_name()[0]["datapath"],
                                                        input.data_file_name()[0]["name"])
        if server_ranges.get() is not None:  # remove existing data
            for i in range(len(server_ranges.get())):
                ui.remove_ui(selector="div:has(> #in_data_"+str(i)+")")
                ui.remove_ui(selector="div:has(> #in_scale_" + str(i) + ")")

        server_ranges.set(ranges)
        server_features.set(features)
        server_data.set(data)
        server_types.set(types)
        server_scale.set(scales)
        server_categories.set(categories)
        # for each feature add a slider or selection list to modify
        default_value = data.iloc[0]
        for i in range(len(ranges)-1, -1, -1):
            # IAIN add the possibility for categorical values (dropdown)
            if input.data_file_option():
                ui.insert_ui(ui.input_numeric('in_scale_' + str(i), 'Scale', value=scales[i], min=0, max=10*scales[i]),
                             # id string of object relative to placement
                             selector="#slider_container",
                             where="afterBegin")
            else:
                ui.insert_ui(
                    ui.input_numeric('in_scale_' + str(i), 'Scale - locked', value=scales[i], min=scales[i], max=scales[i]),
                    # id string of object relative to placement
                    selector="#slider_container",
                    where="afterBegin")
            if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                if not input.manual_data():
                    ui.insert_ui(ui.input_slider('in_data_' + str(i),
                                                 str(features[i]),
                                                 value=default_value[features[i]],
                                                 min=ranges[i][0],
                                                 max=ranges[i][1],
                                                 sep='', width="80%"),
                                 # id string of object relative to placement
                                 selector="#slider_container",
                                 where="afterBegin"
                                 )
                else:
                    ui.insert_ui(ui.input_numeric('in_data_' + str(i),
                                                  str(features[i]),
                                                  default_value[features[i]],
                                                  min=ranges[i][0],
                                                  max=ranges[i][1], width="80%"),
                                 # id string of object relative to placement
                                 selector="#slider_container",
                                 where="afterBegin"
                                 )
            else:
                ui.insert_ui(ui.input_select('in_data_' + str(i),
                                             str(features[i]),
                                             list(ranges[i]),
                                             selected=str(default_value[features[i]]),
                                             width="80%"),
                             # id string of object relative to placement
                             selector="#slider_container",
                             where="afterBegin")
        ui.update_sidebar(id="sidebar", show=True)



css_dir = Path(__file__).parent / "styles"
app = App(app_ui, server, static_assets=css_dir)
