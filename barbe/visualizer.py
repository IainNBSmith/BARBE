from shiny import reactive, App, render, run_app, ui
from htmltools import css

import barbe.explainer
# from shiny.express import ui
from barbe.utils.visualizer_utils import *
import pickle
import torch

# TODO: IAIN add error checking before using a dataset (error should give popup)
# TODO: add a checkbox for scale being row 0 of the given csv
# TODO: add manual scale sets (and the option to save this info from app)
# TODO: add option to select particular values from a simple file (not just 3 lines)
    # TODO/DONE Aug 14: add modifications to max and min values
    # TODO: add the ability to save all information from values
# TODO: add generic error handling (Some done today - Aug 13)
# TODO: clean up this code
# TODO: fix error that occurs when clicking radio box for set manually after swapping between two datasets
# TODO: make the numeric values display only 4-5 sigdigits
# TODO: add a setting to get the prediction from the surrogate?


app_ui = ui.page_fluid(ui.layout_columns(ui.card(ui.input_file('data_file_name', 'Choose Data File', accept=[".csv", ".data"]),
                                                 ui.input_checkbox('data_file_option', 'Premade Ranges File'),
                                                 ui.input_checkbox('data_perturbations', 'Replace Ranges with Perturbations')),
                                               ui.input_file('predictor_file_name', 'Choose Black Box Model', accept=[".pickle", ".torch", ".pkl"]),
                                               ui.card(ui.input_checkbox('data_save', 'Save Ranges and Scale'),
                                                       ui.input_checkbox('explainer_save', 'Save Explainer'),
                                                       ui.input_action_button('save_data', 'Save'))),
                       ui.layout_sidebar(ui.sidebar(ui.input_checkbox('manual_data', 'Manually Set Values'),
                                                    ui.output_text('prediction'),
                                                    ui.input_action_button('data_info', 'Predict'), id='sidebar',
                                                    width=400),
                                         # IAIN add more information from the explainer besider the plot (in a card)
                                        ui.layout_columns(ui.card(ui.card_header("BARBE Explainer"),
                                                                              ui.input_select('dist_setting', 'Distribution', barbe.explainer.DIST_INFO),
                                                                              ui.input_numeric('pert_setting', 'Number of Perturbations', 5000, min=1000, max=10000, step=1000),
                                                                              ui.input_numeric('dev_setting', 'Deviation Scaling', 5, min=1, max=100, step=1),
                                                                              ui.input_checkbox('set_class_setting', 'Use All Classes'),
                                                                              ui.output_text('info'),
                                                                              ui.input_action_button('data_explain', 'Explain')),
                                         ui.output_plot('explanation_plot')),
                                        ui.output_table('explanation_rules'))
                                         )

def server(input, output, session):
    server_data = reactive.value(None)
    server_features = reactive.value(None)
    server_ranges = reactive.value(None)
    server_types = reactive.value(None)

    server_predictor = reactive.value(None)
    server_explainer = reactive.value(None)
    server_prediction = reactive.value(None)
    server_rules = reactive.value(None)
    server_plot = reactive.value(None)

    def reset_data():
        prev_value = list()
        if server_data.get() is not None:
            if server_ranges.get() is not None:  # remove existing data
                for i in range(len(server_ranges.get())):
                    prev_value.append(input['in_data_'+str(i)]())
                    # IAIN fix remove ui here
                    ui.remove_ui(selector="div:has(> #in_data_"+str(i)+")")

            data = server_data.get()
            features = server_features.get()
            ranges = produce_ranges(data)
            server_ranges.set(ranges)
            ui.notification_show("Set all values")
            # for each feature add a slider or selection list to modify
            default_value = data.iloc[0]
            for i in range(len(ranges) - 1, -1, -1):
                # IAIN add the possibility for categorical values (dropdown)
                if len(ranges[i]) == 2:
                    if not input.manual_data():
                        ui.insert_ui(ui.input_slider('in_data_' + str(i),
                                                     str(features[i]),
                                                     value=prev_value[i],
                                                     min=ranges[i][0],
                                                     max=ranges[i][1],
                                                     sep='', width="80%"),
                                     # id string of object relative to placement
                                     selector="#sidebar",
                                     where="afterBegin"
                                     )
                    else:
                        ui.insert_ui(ui.input_numeric('in_data_' + str(i),
                                                      str(features[i]),
                                                      prev_value[i],
                                                      min=ranges[i][0],
                                                      max=ranges[i][1], width="80%"),
                                     # id string of object relative to placement
                                     selector="#sidebar",
                                     where="afterBegin"
                                     )
                else:
                    ui.insert_ui(ui.input_select('in_data_' + str(i),
                                                 str(features[i]),
                                                 list(ranges[i].astype(str)),
                                                 selected=str(prev_value[i]), width="80%"),
                                 # id string of object relative to placement
                                 selector="#sidebar",
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
            for i in range(len(features)):
                # IAIN how to get dynamic names
                predict_input.iloc[0][features[i]] = input['in_data_'+str(i)]()
            # save and use the input in barbe
            # IAIN now it works
            if predictor.check_valid_data(predict_input.iloc[0].to_numpy().reshape(1,-1)):
                prediction = predictor.predict(predict_input.iloc[0].to_numpy().reshape(1,-1))
                server_prediction.set(prediction[0])
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
            for i in range(len(features)):
                # IAIN how to get dynamic names
                predict_input.iloc[0][features[i]] = input['in_data_' + str(i)]()
            # save and use the input in barbe
            # IAIN now it works
            if predictor.check_valid_data(predict_input.iloc[0].to_numpy().reshape(1, -1)):
                prediction = predictor.predict(predict_input.iloc[0].to_numpy().reshape(1, -1))
                server_prediction.set(prediction[0])
                ui.notification_show("Predicted: " + str(prediction[0]) + " for current sample.")

                # now run the explainer
                settings = {'perturbation_type': input.dist_setting(),
                            'dev_scaling_factor': input.dev_setting(),
                            'input_sets_class': not input.set_class_setting(),
                            'n_perturbations': input.pert_setting()}
                explainer, explanation = fit_barbe_explainer(predict_input, features, predict_input.iloc[0], predictor,
                                                input.data_file_option(), settings=settings)
                #ui.notification_show("Fit BARBE explainer.")
                if explanation is not None:
                    server_plot.set(feature_importance_barplot(explanation))
                    #ui.notification_show("Produced BARBE plots.")
                    server_rules.set(barbe_rules_table(explainer.get_rules()))
                    #ui.notification_show("Retrieved BARBE rules.")
                    server_explainer.set(explainer)

                    if input.data_perturbation():
                        server_data.set(explainer.get_perturbed_data())
                        reset_data()
                else:
                    ui.notification_show("Error: Mismatched sample input and BARBE prediction, try again or change sample.")

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


    @reactive.effect
    @reactive.event(input.data_perturbations)
    def _():
        if not input.data_perturbations():
            ui.notification_show("Data may need to be reloaded to reset option.")
        else:
            reset_data()

    @reactive.effect
    @reactive.event(input.data_file_name)
    def _():
        server_prediction.set(None)
        # read in elements from the file
        data, features, types, ranges = open_input_file(input.data_file_name()[0]["datapath"],
                                                        input.data_file_name()[0]["name"])
        if server_ranges.get() is not None:  # remove existing data
            for i in range(len(server_ranges.get())):
                ui.remove_ui(selector="div:has(> #in_data_"+str(i)+")")

        server_ranges.set(ranges)
        server_features.set(features)
        server_data.set(data)
        server_types.set(types)
        # for each feature add a slider or selection list to modify
        default_value = data.iloc[0]
        for i in range(len(ranges)-1, -1, -1):
            # IAIN add the possibility for categorical values (dropdown)
            if len(ranges[i]) == 2:
                if not input.manual_data():
                    ui.insert_ui(ui.input_slider('in_data_' + str(i),
                                                 str(features[i]),
                                                 value=default_value[features[i]],
                                                 min=ranges[i][0],
                                                 max=ranges[i][1],
                                                 sep='', width="80%"),
                                 # id string of object relative to placement
                                 selector="#sidebar",
                                 where="afterBegin"
                                 )
                else:
                    ui.insert_ui(ui.input_numeric('in_data_' + str(i),
                                                  str(features[i]),
                                                  default_value[features[i]],
                                                  min=ranges[i][0],
                                                  max=ranges[i][1], width="80%"),
                                 # id string of object relative to placement
                                 selector="#sidebar",
                                 where="afterBegin"
                                 )
            else:
                ui.insert_ui(ui.input_select('in_data_'+str(i),
                                             str(features[i]),
                                             list(ranges[i].astype(str)),
                                             selected=str(default_value[features[i]]), width="80%"),
                # id string of object relative to placement
                selector="#sidebar",
                where="afterBegin")


app = App(app_ui, server)
