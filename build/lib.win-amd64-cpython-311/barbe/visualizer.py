import string

import pandas as pd
from shiny import reactive, App, render, run_app, ui
from shiny.session import require_active_session
from shiny._utils import drop_none
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
# TODO/Done: add a setting to get the prediction from the surrogate?

# TODO/Done Aug 20: make the numeric values have to be whole integers
# TODO/Done Aug 20: Add manual scale values
# TODO/Done Aug 20 (implicitly): Add selection for values to change
# TODO/Done: Add new values after doing counterfactual
# TODO/Done: Change error notifications into type='error'

# (October 16) TODO: Add advanced option for bound handling
#       - Selectize menu with all the options for bound handling
# (October 16) TODO: Clean up visualizer code, avoid repeats, handle all exceptions
#       - Nasty one that has been there since the start, should be a function call
#       - Cleaning up when reloading should be a function call
# (October 16 - URGENT) TODO: Add checks for the values used in advanced settings against the value to perturb
#       - When bounds are outside of the value it is a problem
#       - When min is larger than max
#       - When min == max
#       - When current option for selectize is not in the possible options
# (October 16) TODO: By default make BARBE use bounds from the dataset
# (October 16) TODO: Fix visualizer issue where it needs to scroll when too many features are used
# (October 16) TODO: Fix feature importance graph scales changing when it's size changes - keeps adding on
#       - Save the graph itself and then when it is called use the original
# (Someday) TODO: Add the save button for convenient opening where a user left off
#print(Path(__file__) / 'visualizer_javascript.js')

#********** PAGE FORMATTING **********#
app_ui = ui.page_fluid(ui.tags.link(rel='stylesheet', href='styles.css'),
                       ui.include_js(path=Path(__file__).parent / 'utils' / 'visualizer_javascript.js'),
                #********** LOAD/SAVE **********#
                 ui.layout_columns(
                      ui.card(ui.input_file('data_file_name', 'Choose Data File', accept=[".csv", ".data"])),
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
                          ui.output_text('barbe_prediction'),
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
                                                         ui.input_action_button('explain_settings',
                                                                                'Advanced Settings')),
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
                    ui.div(ui.output_data_frame('applicable_rules'), id='exp_rules')
                    ),
                    ui.div(ui.output_data_frame('explanation_rules'), id='exp_rules')
                 )


def server(input, output, session):
    # ********** GLOBAL VARIABLES **********#
    server_data = reactive.value(None)  # loaded data to use
    server_features = reactive.value(None)  # feature names of loaded data
    server_ranges = reactive.value(None)  # ranges or all values of features in loaded data
    server_types = reactive.value(None)  # types of features from loaded data
    server_scale = reactive.value(None)  # scale loaded by BarbePerturber
    server_cov = reactive.value(None)
    server_categories = reactive.value(None)  # categorical information loaded by BarbePerturber

    server_predictor = reactive.value(None)  # black box model
    server_explainer = reactive.value(None)  # BARBE explainer
    server_prediction = reactive.value(None)  # prediction to display
    server_barbe_prediction = reactive.value(None)  # prediction from barbe
    server_rules = reactive.value(None)  # rules table from BARBE
    server_plot = reactive.value(None)  # plot to display of feature importance
    server_counter_rules = reactive.value(None)
    server_counter = reactive.value({'': ''})
    server_prev = reactive.value(None)
    server_applicable_rules = reactive.value(None)
    # server_explainer_popup = {'feature_i_settings': {'bounds', 'scale'} or {'possible_values', 'change'}}
    server_explainer_popup = reactive.value(None)  # settings for a popup card if known
    server_visible_events = reactive.value([])

    active_pert = reactive.value(False)  # checks if a checkbox has been pre-checked
    scale_visible = reactive.value(False)

    def get_settings_values():
        advanced_settings = server_explainer_popup.get()
        identifier = category_identifier(server_ranges.get())
        ui.notification_show('Getting Scales')
        scales = correct_scales(server_scale.get(), advanced_settings, identifier)
        ui.notification_show('Getting Categories')
        try:
            categories = correct_categories(server_categories.get(), advanced_settings, identifier, ui)
        except Exception as e:
            ui.notification_show(str(e))
            assert False
        ui.notification_show('Getting Covariance')
        covariance = correct_covariance(server_cov.get(), scales, advanced_settings, identifier)
        ui.notification_show('Getting Bounds')
        try:
            bounds = correct_bounds(advanced_settings, identifier)
        except Exception as e:
            ui.notification_show(str(e))
        return scales, categories, covariance, bounds

    def validate_settings(input_data, scales, categories, covariance, bounds):
        error_string = ""
        input_data = input_data.to_numpy()
        features = server_features.get()
        ranges = server_ranges.get()
        for i in range(len(ranges)):
            if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                try:
                    if bounds is not None and bounds[i] is not None:
                        if bounds[i][0] is not None and input_data[i] < bounds[i][0]:
                            error_string += ('lower bound for ' + features[i] + ', ' + str(bounds[i][0]) +
                                             ', is lower than input value ' + str(input_data[i]) + '\n')
                        if bounds[i][1] is not None and input_data[i] > bounds[i][1]:
                            error_string += ('upper bound for ' + features[i] + ', ' + str(bounds[i][1]) +
                                             ', is higher than input value ' + str(input_data[i]) + '\n')
                except Exception as e:
                    ui.notification_show(str(bounds))
                    ui.notification_show(str(features))
                    ui.notification_show(str(input_data))
                    ui.notification_show(str(e))
                    assert False
            else:
                if categories is not None:
                    if input_data[i] not in categories[i]:
                        error_string += ('possible values for ' + features[i] + ', ' + str(categories[i]) +
                                         ', does not contain input value ' + str(input_data[i]) + '\n')

        return error_string


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
            prediction = 'No prediction yet!'
        return 'Prediction: ' + str(prediction)

    @render.text
    def barbe_prediction():
        barbe_prediction = server_barbe_prediction.get()
        if barbe_prediction is None:
            barbe_prediction = 'BARBE is not trained yet!'
        return 'Barbe Prediction: ' + str(barbe_prediction)

    @render.plot
    def explanation_plot():
        return server_plot.get()

    @render.text
    def info():
        explainer = server_explainer.get()
        if explainer is not None:
            return "Fidelity:" + str(explainer.get_surrogate_fidelity())
        return None

    @render.data_frame
    def explanation_rules():
        if server_rules.get() is None:
            return None
        return render.DataGrid(server_rules.get(), width='100%', filters=True, summary=False)

    @render.data_frame
    def applicable_rules():
        if server_applicable_rules.get() is None:
            return None
        return render.DataGrid(server_applicable_rules.get(), width='100%', summary=False)

    @render.table
    def counter_rules():
        return server_counter_rules.get()

    # ********** REACTIVE ELEMENTS **********#
    def create_reactive_button(x, y, symbol="+"):
        #ui.notification_show("Do a Thing")
        button_label = 'button_' + str(x) + '_' + str(y)
        ui.insert_ui(ui.input_action_button(button_label, symbol if x != y else "", disabled=((x == y) or (y < x))),
                     '#column_spot',
                     where='beforeEnd')
        if button_label not in server_visible_events.get():
            new_code = string.Template("@reactive.effect\n"
                                       "@reactive.event(input.button_${x}_${y})\n"
                                       "def _():\n"
                                       "\tcurrent_server = server_explainer_popup.get()\n"
                                       "\tnew_label = get_next_label(current_server['cov'][${x}][${y}])\n"
                                       "\tui.update_action_button('button_${x}_${y}', "
                                       "label=new_label)\n"
                                       "\tui.update_action_button('button_${y}_${x}', "
                                       "label=new_label)\n"
                                       "\tsession=require_active_session(None)\n"
                                       "\tsession.send_input_message('button_${x}_${y}', "
                                       "drop_none({'id': 'button_${x}_${y}', 'new_label': new_label}))\n"
                                       "\tsession.send_input_message('button_${y}_${x}', "
                                       "drop_none({'id': 'button_${y}_${x}', 'new_label': new_label}))\n"
                                       "\tcurrent_server['cov'][${x}][${y}] = new_label\n"
                                       "\tcurrent_server['cov'][${y}][${x}] = new_label\n"
                                       "\tserver_explainer_popup.set(current_server)").substitute(locals())
            new_code = compile(new_code, '<string>', 'single')
            exec(new_code, {'input': input, 'reactive': reactive, 'ui': ui, 'get_next_label': get_next_label,
                            'server_explainer_popup': server_explainer_popup,
                            'require_active_session': require_active_session,
                            'drop_none': drop_none})
            new_events = server_visible_events.get()
            new_events.append(button_label)
            server_visible_events.set(new_events)
        #else:
        #    ui.notification_show('already there! click the button!!')

    @reactive.effect
    @reactive.event(input.show_scale)
    def _():
        if not scale_visible.get():
            scale_visible.set(True)
            reset_data(new_show=True)
            ui.update_action_button('show_scale', disabled=True)
            ui.notification_show('Scale will now appear beside data.')

    @reactive.effect
    @reactive.event(input.explain_settings)
    def _():

        # add a floating card to the ui which will include these settings
        ui.insert_ui(ui.div(ui.card(ui.input_action_button('settings_close', "X"),
                                    ui.div(id='settings_container'),
                                    id='settings_card'),
                            id='settings_background'),
           selector="#explanation_rules",
           where="afterBegin")

        prev_value = list()
        if server_ranges.get() is None:  # remove existing data
            return

        prev_scale = server_scale.get()

        data = server_data.get()
        features = server_features.get()
        ranges = produce_ranges(data, server_categories.get())
        server_ranges.set(ranges)
        scales = server_scale.get()
        # ui.notification_show("Set all values")
        # for each feature add a slider or selection list to modify
        default_value = data.iloc[0]
        settings = server_explainer_popup.get()
        for i in range(len(ranges) - 1, -1, -1):
            # IAIN add the possibility for categorical values (dropdown)
            settings_identifier = "feature_" + str(i) + "_settings"
            if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                # bounds parsed as None, (Min, Max), "Min, Max", "Min Max" (otherwise return an error)
                scale_value = '(' + str(ranges[i][0]) + "," + str(ranges[i][1]) + ")" \
                    if settings is None or settings[settings_identifier]['bounds'] is None else (
                    str(settings[settings_identifier]['bounds']))
                ui.insert_ui(
                    ui.input_text('in_bounds_' + str(i), 'Bounds', value=scale_value,
                                  placeholder="(Min, Max) or None"),
                    # id string of object relative to placement
                    selector="#settings_container",
                    where="afterBegin")
                # scale check that it is valid (>=0)
                scale_value = prev_scale[i] \
                    if settings is None else (
                    settings[settings_identifier]['scale'])
                ui.insert_ui(
                    ui.input_numeric('in_scale_' + str(i), 'Scale', value=scale_value, min=0, max=10 * scales[i]),
                    # id string of object relative to placement
                    selector="#settings_container",
                    where="afterBegin")
                ui.insert_ui(
                    ui.div(str(features[i])),
                    # id string of object relative to placement
                    selector="#settings_container",
                    where="afterBegin")
            else:
                # possible values to set for categories (default all) must contain at least the current value (or error)
                possible_value = list(ranges[i]) \
                    if settings is None else (
                    list(settings[settings_identifier]['possible_values']))
                #ui.notification_show(str(possible_value))
                ui.insert_ui(
                    ui.input_selectize('in_values_possible_' + str(i), "Possible Values",
                                       list(ranges[i]), selected=possible_value,
                                       multiple=True),
                    # id string of object relative to placement
                    selector="#settings_container",
                    where="afterBegin")
                # whether this value can change (if not then scale = 0 for it)
                # IAIN maybe add a tooltip for a title option or all of them that states 0 is no change
                change_value = False \
                    if settings is None else (
                    settings[settings_identifier]['change'])
                ui.insert_ui(
                    ui.input_checkbox('in_change_' + str(i), 'Lock Value', value=change_value),
                    # id string of object relative to placement
                    selector="#settings_container",
                    where="afterBegin")
                ui.insert_ui(
                    ui.div(str(features[i])),
                    # id string of object relative to placement
                    selector="#settings_container",
                    where="afterBegin")

            # IAIN add more options down below (or change options) based on the model selected
            #  for example, uniform needs scale + sizes (can go up or down in a different shape)
            #               t-distribution can have degrees of freedom
            #               normal can have multivariate options -1, 0, 1 for negative, none, and positive correlation
            # settings_use

        # (DONE) TODO: IAIN add the multivariate button setup (its own function)
        # (DONE) TODO:  these will all use the default behavior you just made when being created
        # (DONE) TODO: IAIN add the automatic state setting
        # (DONE) TODO: add multivariate state reading by icon
        # (DONE) TODO: add javascript to these buttons so the color changes based on the state they are in
        # TODO: make all the settings actually affect BARBE when set and saved + add reset button to ignore changes

        ui.insert_ui(ui.div(ui.layout_column_wrap(width=1 / (len(ranges)+1), id='column_spot'), id='column_container'),
                     '#settings_container',
                     where='afterEnd')
        if settings is None:
            corrected_cov = correct_cov_values(server_cov.get())
            server_explainer_popup.set({'cov': corrected_cov})
        else:
            corrected_cov = settings['cov']

        for x in range(-1, len(ranges)):
            for y in range(-1, len(ranges)):
                if x >= 0 and y >= 0:
                    create_reactive_button(x, y, symbol=corrected_cov[x][y])
                else:
                    if (x == -1) and (y == -1):
                        ui.insert_ui(ui.div('Feature Relative Variance'),
                                     '#column_spot',
                                     where='beforeEnd')
                    else:
                        feature_position = x if x > y else y
                        ui.insert_ui(ui.div(features[feature_position]),
                                     '#column_spot',
                                     where='beforeEnd')

        ui.insert_ui(ui.div(ui.layout_columns(ui.input_action_button('settings_use', 'Save Settings'),
                                       ui.input_action_button('settings_reset', 'Reset'),
                                       col_widths=(-1, 7, 3, -1))),
            # id string of object relative to placement
            selector="#column_container",
            where="afterEnd")

    @reactive.effect
    @reactive.event(input.settings_close)
    def _():
        ui.remove_ui(selector="div:has(> #settings_card)")
        ui.remove_ui(selector="div:has(> #settings_background)")

    @reactive.effect
    @reactive.event(input.settings_reset)
    def _():
        server_explainer_popup.set(None)
        ui.remove_ui(selector="div:has(> #settings_card)")
        ui.remove_ui(selector="div:has(> #settings_background)")

    @reactive.effect
    @reactive.event(input.settings_use)
    def _():
        # TODO: make all of the settings get updated and after closing if a model is trained then make sure to remove it
        # TODO:  so that a user cannot pick options that the model would not have access to and perform poorly on
        settings = server_explainer_popup.get()
        if settings is None:
            settings = {}
        ranges = server_ranges.get()
        for i in range(len(ranges) - 1, -1, -1):
            # IAIN add the possibility for categorical values (dropdown)
            settings_identifier = "feature_" + str(i) + "_settings"
            #ui.notification_show(settings_identifier)
            settings[settings_identifier] = {}
            if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                # bounds parsed as None, (Min, Max), "Min, Max", "Min Max" (otherwise return an error)
                #ui.notification_show(str(input['in_bounds_' + str(i)]()))
                temp_bounds = format_bounds(input['in_bounds_' + str(i)]())
                #ui.notification_show(temp_bounds)
                if temp_bounds is not None and isinstance(temp_bounds, str):
                    ui.notification_show('Error: ' + temp_bounds, type='error')
                    return
                settings[settings_identifier]['bounds'] = temp_bounds
                #ui.notification_show("got formatted bounds")
                settings[settings_identifier]['scale'] = input['in_scale_' + str(i)]()
                #ui.notification_show("got scale")
            else:
                # possible values to set for categories (default all) must contain at least the current value (or error)
                settings[settings_identifier]['possible_values'] = list(input['in_values_possible_' + str(i)]())
                ui.update_select('in_data_' + str(i), choices=list(input['in_values_possible_' + str(i)]()))
                settings[settings_identifier]['change'] = input['in_change_' + str(i)]()

        server_explainer_popup.set(settings)
        # close the menu
        ui.remove_ui(selector="div:has(> #settings_card)")
        ui.remove_ui(selector="div:has(> #settings_background)")

    @reactive.effect
    @reactive.event(input.counterfactual_suggestions)
    def _():
        if input.counterfactual_suggestions() is '':
            set_suggested()
            ui.notification_show('Reset data values.', type='warning')
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
                    barbe_prediction = explainer.predict(reformat_input.iloc[0].to_numpy().reshape(1, -1))
                    server_barbe_prediction.set(barbe_prediction[0])
                    server_applicable_rules.set(barbe_rules_table(explainer.get_rules(applicable=reformat_input.iloc[0])))
                    ui.notification_show("Predicted: " + str(prediction[0]) + " for current sample and reset "
                                                                              "applicable rules.")
                else:
                    ui.notification_show("Predicted: " + str(prediction[0]) + " for current sample.")
            else:
                ui.notification_show("Model Error: Invalid data format for selected model. "
                                     "Try a different model or format.",
                                     type='error')
        else:
            missing_model = 'black-box' if predictor is None else ''
            missing_data = 'data' if predict_input is None else ''
            ui.notification_show("Error: Element(s) " + missing_model + " " +
                                 missing_data + " is missing.",
                                 type='error')

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
                # ui.notification_show("Settings Extraction")
                input_bounds, modified_category, new_scale, change_category = (
                    extract_advanced_settings(server_explainer_popup.get(), server_ranges.get()))
                # now run the explainer
                settings = {'perturbation_type': input.dist_setting(),
                            'dev_scaling_factor': input.dev_setting(),
                            'input_sets_class': not input.set_class_setting(),
                            'n_perturbations': input.pert_setting(),
                            'n_bins': input.category_setting()}
                ui.notification_show("Settings Check")
                check_values = check_settings(settings, reformat_input, server_ranges.get(), server_features.get())
                #ui.notification_show("Checked Settings")
                if check_values is not None:
                    ui.notification_show("Error: settings " + check_values + " must all be whole numbers.",
                                         type='error')
                else:
                    #ui.notification_show("Starting BARBE")
                    # add 1e-10 so zero is not exactly zero but should not produce variation in most cases
                    #ui.notification_show(str(scales))
                    #ui.notification_show(str(list(features)))
                    #ui.notification_show(str(server_categories.get()))
                    ui.notification_show("Get Values from settings")
                    scales, categories, covariance, bounds = get_settings_values()
                    error_settings = validate_settings(reformat_input.iloc[0], scales, categories, covariance, bounds)
                    if error_settings != "":
                        ui.notification_show(error_settings, type='error')
                        return
                    ui.notification_show("Getting Explainer")
                    explainer, explanation = fit_barbe_explainer(scales, features, categories, covariance, bounds,
                                                                 reformat_input.iloc[0], predictor,
                                                    False, settings=settings)
                    #ui.notification_show(explanation)
                    if explanation is not None and len(explanation) > 0:
                        #ui.notification_show(str(explanation))
                        server_plot.set(feature_importance_barplot(explanation))
                        #ui.notification_show("Produced BARBE plots.")
                        server_rules.set(barbe_rules_table(explainer.get_rules()))
                        server_applicable_rules.set(barbe_rules_table(explainer.get_rules(applicable=reformat_input.iloc[0])))
                        ui.notification_show("Retrieved BARBE rules.")
                        server_explainer.set(explainer)

                        barbe_prediction = explainer.predict(reformat_input.iloc[0].to_numpy().reshape(1, -1))
                        server_barbe_prediction.set(barbe_prediction[0])
                        ui.notification_show('BARBE predicted.')

                        ui.update_action_button('data_counterfactual', disabled=False)
                        ui.update_select('class_setting', choices=explainer.get_available_classes())

                        if input.data_perturbation():
                            server_data.set(explainer.get_perturbed_data())
                            reset_data()
                    else:
                        ui.notification_show(str(explainer), type='error')
                        ui.notification_show("Error: Mismatched sample input and BARBE prediction, try again or change sample.",
                                             type='error')

            else:
                ui.notification_show("Model Error: Invalid data format for selected model. "
                                     "Try a different model or format.", type='error')
        else:
            missing_model = 'black-box' if predictor is None else ''
            missing_data = 'data' if predict_input is None else ''
            ui.notification_show("Error: Element(s) " + missing_model + " " +
                                 missing_data + " is missing.", type='error')


    @reactive.effect
    @reactive.event(input.predictor_file_name)
    def _():
        ui.update_action_button('data_counterfactual', disabled=True)
        ui.update_select('class_setting', choices=[])
        server_prediction.set(None)
        server_barbe_prediction.set(None)
        server_explainer.set(None)
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
        server_barbe_prediction.set(None)
        server_explainer_popup.set(None)
        # read in elements from the file
        data, features, types, ranges, scales, cov, categories = open_input_file(input.data_file_name()[0]["datapath"],
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
        server_cov.set(cov)
        server_categories.set(categories)
        # for each feature add a slider or selection list to modify
        default_value = data.iloc[0]
        for i in range(len(ranges)-1, -1, -1):
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
