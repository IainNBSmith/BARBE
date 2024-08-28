import pandas as pd
from shiny import reactive, App, render, run_app, ui
from htmltools import css

from pathlib import Path
# from shiny.express import ui
from barbe.utils.visualizer_utils import *
from htmltools import tags, TagAttrValue
import barbe
import pickle
import torch
import re

question_circle_fill = ui.HTML(
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/></svg>'
)

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

#********** PAGE FORMATTING **********#
app_ui = ui.page_fluid(ui.tags.link(rel='stylesheet', href='styles.css'),
                #********** LOAD/SAVE **********#
                 ui.layout_columns(
                      ui.card(ui.input_file('data_file_name', 'Choose Data File', accept=[".csv", ".data"]),
                               ui.input_checkbox('data_file_option', 'User Set Ranges'),
                               ui.input_checkbox('data_perturbations', 'Replace Ranges with Perturbations')),
                      ui.div(ui.input_checkbox('data_save', 'Save Ranges and Scale'),
                             ui.input_checkbox('explainer_save', 'Save Explainer'),
                             ui.input_action_button('save_data', 'Save', disabled=True), id="save_menu"),
                      ui.card(ui.input_file('predictor_file_name', 'Choose Black Box Model', accept=[".pickle", ".torch", ".pkl"]))),

                 ui.layout_sidebar(
                     #********** SIDEBAR DATA CONTAINER **********#
                     ui.sidebar(
                          ui.div(id="slider_container"),
                          ui.input_checkbox('manual_data', 'Manually Set Values'),
                          ui.output_text('prediction'),
                          ui.input_action_button('data_info', 'Predict'),
                          id='sidebar', width=400, title="Data", open="closed"),

                 #********** RESULTS OUTPUT **********#
                 ui.layout_columns(ui.output_plot('explanation_plot'),
                    #********** BARBE EXPLAINER SETTINGS **********#
                    ui.card(ui.card_header("BARBE Explainer"),
                            ui.tooltip(ui.span(ui.input_select('dist_setting', 'Distribution for Perturbations',
                                            barbe.explainer.DIST_INFO), id='tooltip_container'),
                                       'Uniform is consistent, skewed distributions can help find outliers in '
                                       'model.', placement='left'),
                            ui.input_numeric('pert_setting', 'Number of Perturbations', 5000, min=1000,
                                             max=10000, step=1000),
                            ui.input_numeric('dev_setting', 'Deviation Scaling', 5, min=1, max=100,
                                             step=1),
                            ui.tooltip(ui.span(ui.input_numeric('category_setting', 'Bins for Continuous Data', 5, min=2,
                                             max=20, step=1), id='tooltip_container'),
                                       'More bins captures more precise bounds of black box.', placement='left'),
                            ui.input_checkbox('set_class_setting', 'Use All Classes', value=True),
                            ui.layout_columns(ui.tooltip(ui.span(ui.output_text('info'),
                                                                 id='tooltip_container'),
                                                         'Fidelity indicates the proportion of agreements between '
                                                         'perturbed classes and SigDirect (BARBE surrogate) '
                                                         'predictions.', placement='left'),
                                                         ui.input_action_button('show_scale',
                                                                                'Manually Set Scale')),
                            ui.input_action_button('data_explain', 'Explain'),
                            max_height="650px")),
                    #********** RULES **********#
                    ui.layout_columns(ui.div(ui.output_table('counter_rules'), id='cnt_rules'),
                        #********** BARBE COUNTERFACTUAL **********#
                        ui.card(ui.card_header("BARBE Counterfactual"),
                                ui.input_select('class_setting', 'Class Change', []),
                                ui.input_select('counterfactual_suggestions', 'Use Suggested Value', []),
                                ui.input_action_button('data_counterfactual', 'Counterfactual', disabled=True),
                                max_height="400px"),
                                col_widths=(8, 4)),
                    ui.div(ui.output_table('applicable_rules'), id='exp_rules')
                    ),
                    ui.div(ui.output_table('explanation_rules'), id='exp_rules')
                 )


def server(input, output, session):
    # ********** GLOBAL VARIABLES **********#
    server_data = reactive.value(None)  # loaded data to use
    server_features = reactive.value(None)  # feature names of loaded data
    server_ranges = reactive.value(None)  # ranges or all values of features in loaded data
    server_types = reactive.value(None)  # types of features from loaded data
    server_scale = reactive.value(None)  # scale loaded by BarbePerturber
    server_categories = reactive.value(None)  # categorical information loaded by BarbePerturber

    server_predictor = reactive.value(None)  # black box model
    server_explainer = reactive.value(None)  # BARBE explainer
    server_prediction = reactive.value(None)  # prediction to display
    server_rules = reactive.value(None)  # rules table from BARBE
    server_plot = reactive.value(None)  # plot to display of feature importance
    server_counter_rules = reactive.value(None)
    server_counter = reactive.value({'': ''})
    server_prev = reactive.value(None)
    server_applicable_rules = reactive.value(None)

    active_pert = reactive.value(False)  # checks if a checkbox has been pre-checked
    scale_visible = reactive.value(False)

    def set_suggested(values=None):
        ranges = server_ranges.get()
        # ui.notification_show("Set all values")
        # for each feature add a slider or selection list to modify
        reset_flag = values is None
        if not reset_flag:
            prev_value = [None for i in range(len(ranges))]
        else:
            values = server_prev.get()
            server_prev.set(None)
            if values is None:
                return
        for i in range(len(ranges) - 1, -1, -1):
            if not reset_flag:
                prev_value[i] = (input['in_data_' + str(i)]())
            #ui.notification_show('Setting value for ' + str(i))
            if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                if not input.manual_data():
                    ui.update_slider('in_data_' + str(i), value=values[i])
                else:
                    ui.update_numeric('in_data_' + str(i), value=values[i])
            else:
                ui.update_selectize('in_data_' + str(i), selected=str(values[i]))
        if not reset_flag and server_prev.get() is None:
            server_prev.set(prev_value.copy())

    def reset_data(new_show=False):
        prev_value = list()
        prev_scale = list()
        #if server_data.get() is not None:
        #    if server_ranges.get() is not None:  # remove existing data
        if not active_pert:
            return
        if server_ranges.get() is None:  # remove existing data
            return
        for i in range(len(server_ranges.get())):
            prev_value.append(input['in_data_'+str(i)]())
            ui.update_selectize('in_data_' + str(i), label='')
            if not new_show:
                prev_scale.append(input['in_scale_'+str(i)]())
            # IAIN remove does not work for select (check files _input_select.py (parent div does not have an id to remove it) and _input_slider.py)
            ui.remove_ui(selector="div:has(> #in_data_" + str(i) + ")")
            ui.remove_ui(selector="div:has(> #in_scale_" + str(i) + ")")

        if new_show:
            prev_scale = server_scale.get()

        data = server_data.get()
        features = server_features.get()
        ranges = produce_ranges(data, server_categories.get())
        server_ranges.set(ranges)
        scales = server_scale.get()
        #ui.notification_show("Set all values")
        # for each feature add a slider or selection list to modify
        default_value = data.iloc[0]
        for i in range(len(ranges) - 1, -1, -1):
            # IAIN add the possibility for categorical values (dropdown)
            if scale_visible.get():
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
            else:
                # IAIN this is how you VERY MANUALLY have to fix the label problem
                props: dict[str, TagAttrValue] = {
                    "class": 'div',
                    "id": 'in_scale_' + str(i)
                }
                ui.insert_ui(
                    ui.div(tags.div(**props)),
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
                                                                    value=float(prev_value[i]),
                                                                    min=float(ranges[i][0]),
                                                                    max=float(ranges[i][1]), width="80%"),
                                 # id string of object relative to placement
                                 selector="#slider_container",
                                 where="afterBegin"
                                 )
            else:
                ui.insert_ui(ui.input_selectize('in_data_' + str(i),
                                                               str(features[i]),
                                                               list(ranges[i]),
                                                               selected=str(prev_value[i]),
                                                               width="80%"),
                             # id string of object relative to placement
                             selector="#slider_container",
                             where="afterBegin")

    # ********** RENDERED ELEMENTS **********#
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

    @render.table
    def applicable_rules():
        return server_applicable_rules.get()

    @render.table
    def counter_rules():
        return server_counter_rules.get()

    # ********** REACTIVE ELEMENTS **********#
    @reactive.effect
    @reactive.event(input.show_scale)
    def _():
        if not scale_visible.get():
            scale_visible.set(True)
            reset_data(new_show=True)
            ui.notification_show('Scale will now appear beside data.')

    @reactive.effect
    @reactive.event(input.counterfactual_suggestions)
    def _():
        if input.counterfactual_suggestions() is '':
            set_suggested()
            ui.notification_show('Reset data values.')
        else:
            #ui.notification_show(input.counterfactual_suggestions())
            suggest = eval(input.counterfactual_suggestions())
            #ui.notification_show(str(suggest))
            set_suggested(values=suggest)
            ui.notification_show('Set values to suggested.')

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
                if server_explainer.get() is not None:
                    explainer = server_explainer.get()
                    server_applicable_rules.set(barbe_rules_table(explainer.get_rules(applicable=reformat_input.iloc[0])))
                    ui.notification_show("Predicted: " + str(prediction[0]) + " for current sample and reset "
                                                                              "applicable rules.")
                else:
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
                    #ui.notification_show("Starting BARBE")
                    # add 1e-10 so zero is not exactly zero but should not produce variation in most cases
                    if scale_visible.get():
                        scales = [input['in_scale_' + str(i)]() + 1e-10 for i in range(len(server_ranges.get()))]
                    else:
                        scales = server_scale.get()
                    #ui.notification_show(str(scales))
                    #ui.notification_show(str(list(features)))
                    #ui.notification_show(str(server_categories.get()))
                    explainer, explanation = fit_barbe_explainer(scales, features, server_categories.get(),
                                                                 reformat_input.iloc[0], predictor,
                                                    input.data_file_option(), settings=settings)
                    #ui.notification_show(explanation)
                    if explanation is not None and len(explanation) > 0:
                        #ui.notification_show(str(explanation))
                        server_plot.set(feature_importance_barplot(explanation))
                        #ui.notification_show("Produced BARBE plots.")
                        server_rules.set(barbe_rules_table(explainer.get_rules()))
                        server_applicable_rules.set(barbe_rules_table(explainer.get_rules(applicable=reformat_input.iloc[0])))
                        #ui.notification_show("Retrieved BARBE rules.")
                        server_explainer.set(explainer)

                        ui.update_action_button('data_counterfactual', disabled=False)
                        ui.update_select('class_setting', choices=explainer.get_available_classes())

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
        ui.update_action_button('data_counterfactual', disabled=True)
        ui.update_select('class_setting', choices=[])
        server_prediction.set(None)
        # open predictor pickle, use later by passing to BARBE
        input_model = open_input_model(input.predictor_file_name()[0]["datapath"],
                                       input.predictor_file_name()[0]["name"])
        server_predictor.set(input_model)
        pass


    @reactive.effect
    @reactive.event(input.manual_data)
    def _():
        reset_data(new_show=(not scale_visible.get()))

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
    @reactive.event(input.data_counterfactual)
    def _():
        explainer = server_explainer.get()
        features = server_features.get()
        predict_input = server_data.get()
        predictor = server_predictor.get()
        # get the explanation rules
        reformat_input = pd.DataFrame(columns=server_features.get(), index=[0])
        for i in range(len(features)):
            # IAIN how to get dynamic names
            reformat_input.iloc[0][features[i]] = input['in_data_' + str(i)]()

        # ui.notification_show(str(list(reformat_input.iloc[0])))

        counterfactual_prediction, counter_rules, counter_class = explainer.get_counterfactual_explanation(reformat_input, input.class_setting())
        # ui.notification_show('succ')
        #ui.notification_show(str(counterfactual_prediction))
        #ui.notification_show(str(counter_class))
        server_counter_rules.set(barbe_counter_rules_table(counter_rules))
        temp_counter = server_counter.get()
        temp_counter[str(counterfactual_prediction[0])] = (str(counterfactual_prediction) + " -> " +
                                                           " BARBE Prediction: " + counter_class +
                                                           ", Wanted: " + input.class_setting())
        #ui.notification_show(str(temp_counter))
        ui.update_select('counterfactual_suggestions', choices=temp_counter)
        server_counter.set(temp_counter)
        ui.notification_show('Generated counterfactual, in suggestions menu.')

    @reactive.effect
    @reactive.event(input.data_file_name)
    def _():
        # IAIN make these settings into a function
        ui.update_action_button('data_counterfactual', disabled=True)
        ui.update_select('class_setting', choices=[])
        server_counter.set({'': ''})
        ui.update_select('counterfactual_suggestions', choices=server_counter.get())
        server_prev.set(None)
        server_prediction.set(None)
        # read in elements from the file
        data, features, types, ranges, scales, categories = open_input_file(input.data_file_name()[0]["datapath"],
                                                        input.data_file_name()[0]["name"])
        if server_ranges.get() is not None:  # remove existing data
            for i in range(len(server_ranges.get())):
                ui.update_selectize('in_data_' + str(i), label='')
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
            if scale_visible.get():
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
            else:
                props: dict[str, TagAttrValue] = {
                    "class": 'div',
                    "id": 'in_scale_' + str(i)
                }
                ui.insert_ui(
                    ui.div(tags.div(**props)),
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
                    # IAIN breaks on some data (do not know why...)
                    ui.insert_ui(ui.input_numeric('in_data_' + str(i),
                                                  str(features[i]),
                                                  value=float(default_value[features[i]]),
                                                  min=float(ranges[i][0]),
                                                  max=float(ranges[i][1]), width="80%"),
                                 # id string of object relative to placement
                                 selector="#slider_container",
                                 where="afterBegin"
                                 )
            else:
                ui.insert_ui(ui.input_selectize('in_data_' + str(i),
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
