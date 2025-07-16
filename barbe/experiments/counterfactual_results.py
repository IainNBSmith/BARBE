import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Take result files from experiments and calculate avg, std, and make plots etc.


def make_tables(df, final_file_name='barbe_counterfactual_results'):
    # if dens is in the name then use the median instead, if diff counter or time are in the name include std
    summarized_cols = df.columns[4:]  # 4 is fidelity
    final_columns = list()
    final_summary = list()
    for col_name in summarized_cols:
        mean_result = np.mean(df.loc[df[col_name] != -1, col_name]) if 'dens' not in col_name \
            else np.median(df.loc[df[col_name] != -1, col_name])
        final_columns.append(f'{col_name}_average')
        final_summary.append(mean_result)

        if 'c-hit-' in col_name:
            sum_results = np.sum(df[col_name] != -1)
            final_columns.append(f'{col_name}_total_tries')
            final_summary.append(sum_results)

        if 'diff' in col_name or 'counter' in col_name or 'time' in col_name or 'fidelity' in col_name:
            std_results = np.std(df.loc[df[col_name] != -1, col_name])
            np.mean(df.loc[df[col_name] != -1, col_name])
            final_columns.append(f'{col_name}_standard_deviation')
            final_summary.append(std_results)

    results_df = pd.DataFrame(columns=final_columns)
    results_df.loc[0] = final_summary
    results_df.to_csv(f'./SummarizedResults/{final_file_name}.csv')


def make_point_deviation_plots(df, hue_column, value_column):
    prev_ax = None
    unique_approaches = np.unique(df['bbmodel'])
    print(np.unique(df[hue_column]))
    markers_possible = ['o', 'v', 's', 'p', 'X', 'd']
    for approach in unique_approaches:
        print("HEY:", approach)
        current_marker = markers_possible.pop()
        print(current_marker)
        prev_ax = sns.lineplot(data=df.loc[df['bbmodel'] == approach],
                               style=hue_column,
                               hue=hue_column,
                               palette={'BARBE': 'red',
                                        'cv_mlp': 'blue',
                                        'cv_fieap': 'green'},
                               markers=current_marker,
                               x='n_counterfactual', y=value_column, legend=False,
                               err_style='bars', ax=prev_ax)

    #prev_ax.style.use('seaborn')
    #prev_ax.rcParams.update({'lines.markeredgewidth':3})

    plt.show()


def make_barplot():
    ax = sns.countplot(df, x='n-c-hit', hue='method',
                       palette={'BARBE-MN (fi)': (0, 114/255, 178/255),
                                'BARBE-N (fi)': (230/255, 159/255, 0),
                                'No-Neg-BARBE-MN (fi)': (86/255, 180/255, 233/255),
                                'No-Neg-BARBE-N (fi)': (213/255, 94/255, 0),
                                'DiCE-genetic': (204/255, 121/255, 167/255),
                                'LORE': (0, 158/255, 115/255)},
                       hue_order=['No-Neg-BARBE-N (fi)', 'No-Neg-BARBE-MN (fi)',
                                  'BARBE-N (fi)', 'BARBE-MN (fi)',
                                  'LORE', 'DiCE-genetic'])
    ax.set_xlabel('Number of Counterfactual Hits')
    ax.set_ylabel('Count')
    hatches = ['///', '\\\\\\', '--', '', '..', 'x']
    for hues, hatch in zip(ax.containers, hatches):
        for hue in hues:
            hue.set_hatch(hatch)
    total_subtract = 0
    for j in range(8):
        curr_misses = 0
        for i in range(6):
            if j != 7:
                if np.sum((df['method'] == ['No-Neg-BARBE-N (fi)', 'No-Neg-BARBE-MN (fi)',
                                      'BARBE-N (fi)', 'BARBE-MN (fi)',
                                      'LORE', 'DiCE-genetic'][i]) & (df['n-c-hit'] == [-1, 0, 1, 2, 3,4,5][j])) != 0:
                    #ax.patches[i + 6*j - total_subtract].set_hatch(hatches[i])
                    pass
                else:
                    curr_misses += 1
                    total_subtract += 1
            else:
                ax.patches[i + 6 * j - total_subtract].set_hatch(hatches[i])
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    #ax.set_position([box.x0 - box.width*0.5, box.y0,# + box.height * 0.5,
    #                 box.width * 2, box.height * 1.5])
    ax.legend(legend_handles, legend_labels, ncol=3, title='',
              loc='center', bbox_to_anchor=(0, -0.6, 1, 0.5), mode='expand')
    #sns.move_legend(ax, 'center', bbox_to_anchor=(0,0))
    plt.tight_layout()
    plt.savefig('./SummarizedResults/ncolPlot.png')
    plt.show()

def make_plots(ax, df, color='red', final_file_name='barbe_counterfactual_plots'):
    # make plot for regular and one for restricted...?
    #for j in range(8):
    #    curr_misses = 0
    #    for i in range(6):
    #        if j != 7:
    #            start_shape = df.shape[0]
    #            df.loc[start_shape, 'method'] = ['No-Neg-BARBE-N (fi)', 'No-Neg-BARBE-MN (fi)',
    #                              'BARBE-N (fi)', 'BARBE-MN (fi)',
    #                              'LORE', 'DiCE-genetic'][i]
    #            df.loc[start_shape, 'n-c-hit'] = [-1, 0, 1, 2, 3,4,5][j]
    #            print("START: ", start_shape)
    #        else:
    #            pass

    x_vals = list()
    y_vals = list()
    sum_in_regions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 5+1):
        curr_x = f'diff-c-{i}'
        curr_y = f'dens-c-{i}'
        ii = -1
        key = list()
        df[curr_y] = df[curr_y].apply(func=lambda y: np.log10(y))
        for region_y in [lambda x: x <= 2, lambda x: 2 < x <= 5, lambda x: 5 < x <= 20]:
            ii += 1
            jj = -1
            for region_x in [lambda x: x <= 1, lambda x: 1 < x <= 3, lambda x: 3 < x <= 6]:
                jj += 1
                key.append((jj, ii))
                sum_in_regions[jj + ii*3] += np.sum(df[curr_x].apply(func=region_x) & df[curr_y].apply(func=region_y))
        #df[curr_x] = df[curr_x].apply(func=lambda x: min(x, 24))
        df[curr_y] = df[curr_y].apply(func=lambda y: min(y, 80))
        df[curr_y] = df[curr_y].apply(func=lambda y: max(y, -5))


        x_vals += list(df[curr_x].loc[df[f'c-hit-{i}'] == 1])
        y_vals += list(df[curr_y].loc[df[f'c-hit-{i}'] == 1])

    print(key)
    print(color, ":", sum_in_regions)
    #plt = sns.displot(x=x_vals, color=color, fill=True, kind='kde', alpha=0.5, rug=True, ax=ax)
    #plt = sns.scatterplot(x=x_vals, y=y_vals, color=color, alpha=0.5)
    #plt.ylabel('Log Density Ratio')
    #plt.xlabel('Sum of Standard Deviations')
    #plt.ylim((-5.5, 100.5))
    #plt.xlim((-0.5, 24.5))

    return x_vals, y_vals
    #return plt

def make_loan_tables(names='mlp'):
    full_df = None
    x_vals = pd.DataFrame(columns=['Sum of Normalized Distances', 'Approach', 'Log10 Density Ratio', 'Dataset'])
    for bbmodel_name in names:
        nbins = 10 if 'loan' in bbmodel_name else 5
        # open barbe files and split up the normal/multivariate
        barbe_df = pd.read_csv(f'./Results/barbe_{bbmodel_name}_nruns50_nbins{nbins}_nperturb1000_devscalin1_counterfactuals_results.csv')
        barbe_n_df = barbe_df.loc[(barbe_df['distribution'] == 'standard-normal') & (barbe_df['counter-method'] == 'importance-rules')]
        barbe_mn_df = barbe_df.loc[(barbe_df['distribution'] == 'normal') & (barbe_df['counter-method'] == 'importance-rules')]

        # run tables
        make_tables(barbe_n_df, final_file_name=f'barbe_{bbmodel_name}_standard_normal_counterfactual_results')
        make_tables(barbe_mn_df, final_file_name=f'barbe_{bbmodel_name}_attrition_multivariate_normal_counterfactual_results')
        # split each table on the classes Y/N
        #barbe_n_df_Y = barbe_n_df.loc[barbe_n_df['original-class'] == 'Y']
        #barbe_n_df_N = barbe_n_df.loc[barbe_n_df['original-class'] == 'N']
        #barbe_mn_df_Y = barbe_mn_df.loc[barbe_mn_df['original-class'] == 'Y']
        #barbe_mn_df_N = barbe_mn_df.loc[barbe_mn_df['original-class'] == 'N']
        # run tables
        #make_tables(barbe_n_df_Y, final_file_name=f'barbe_{bbmodel_name}_cY_standard_normal_counterfactual_results')
        #make_tables(barbe_n_df_N, final_file_name=f'barbe_{bbmodel_name}_cN_standard_normal_counterfactual_results')
        #make_tables(barbe_mn_df_Y, final_file_name=f'barbe_{bbmodel_name}_cY_multivariate_normal_counterfactual_results')
        #make_tables(barbe_mn_df_N, final_file_name=f'barbe_{bbmodel_name}_cN_multivariate_normal_counterfactual_results')

        # open lore files
        lore_df = pd.read_csv(f'./Results/lore_{bbmodel_name}_nruns50_nbins{nbins}_nperturb1000_devscalin1_counterfactuals_results.csv')
        # run tables
        make_tables(lore_df, final_file_name=f'lore_{bbmodel_name}_counterfactual_results')
        # split lore table on Y/N
        #lore_df_Y = lore_df.loc[lore_df['original-class'] == 'Y']
        #lore_df_N = lore_df.loc[lore_df['original-class'] == 'N']
        # run tables
        #make_tables(lore_df_Y, final_file_name=f'lore_{bbmodel_name}_loan_cY_counterfactual_results')
        #make_tables(lore_df_N, final_file_name=f'lore_{bbmodel_name}_loan_cN_counterfactual_results')

        dice_df = pd.read_csv(f'./Results/dice_{bbmodel_name}_nruns50_nbins{nbins}_nperturb1000_devscalin1_counterfactuals_results.csv')
        make_tables(dice_df.loc[dice_df['distribution'] == 'evolution'], final_file_name=f'dice_{bbmodel_name}_loan_counterfactual_results')

        barbe_no_negation_df = pd.read_csv(f'./Results/barbe_no_negation_{bbmodel_name}_nruns50_nbins{nbins}_nperturb1000_devscalin1_counterfactuals_results.csv')
        barbe_no_negation_n_df = barbe_no_negation_df.loc[barbe_no_negation_df['distribution'] == 'standard-normal']
        barbe_no_negation_mn_df = barbe_no_negation_df.loc[barbe_no_negation_df['distribution'] == 'normal']


        barbe_mn_df = barbe_mn_df.loc[:, barbe_mn_df.columns[:lore_df.shape[1]]]
        barbe_n_df = barbe_n_df.loc[:, barbe_n_df.columns[:lore_df.shape[1]]]
        barbe_mn_df['method'] = 'BARBE-MN (h)'
        barbe_mn_df.loc[barbe_mn_df['counter-method'] == 'importance-rules', 'method'] = 'BARBE-MN (fi)'
        barbe_n_df['method'] = 'BARBE-N-high'
        barbe_n_df.loc[barbe_n_df['counter-method'] == 'importance-rules', 'method'] = 'BARBE-N (fi)'

        barbe_no_negation_mn_df = barbe_no_negation_mn_df.loc[:, barbe_no_negation_mn_df.columns[:lore_df.shape[1]]]
        barbe_no_negation_n_df = barbe_no_negation_n_df.loc[:, barbe_no_negation_n_df.columns[:lore_df.shape[1]]]
        barbe_no_negation_mn_df['method'] = 'No-Neg-BARBE-MN (h)'
        barbe_no_negation_mn_df.loc[barbe_no_negation_mn_df['counter-method'] == 'importance-rules', 'method'] = 'No-Neg-BARBE-MN (fi)'
        barbe_no_negation_n_df['method'] = 'No-Neg-BARBE-N (h)'
        barbe_no_negation_n_df.loc[barbe_no_negation_n_df['counter-method'] == 'importance-rules', 'method'] = 'No-Neg-BARBE-N (fi)'

        lore_df['method'] = 'LORE'
        dice_df['method'] = 'DiCE-random'
        dice_df.loc[dice_df['distribution'] == 'genetic', 'method'] = 'DiCE-genetic'

        #make_plots(dice_df.loc[dice_df['distribution'] == 'genetic'], color='red')
        #make_plots(barbe_mn_df.loc[barbe_mn_df['counter-method'] == 'high-rules'], color='green')
        #make_plots(lore_df, color='blue')
        #plt.hlines(y=2, xmin=-1, xmax=13)
        #plt.hlines(y=5, xmin=-1, xmax=13)
        #plt.vlines(x=1, ymin=-1, ymax=51)
        #plt.vlines(x=3, ymin=-1, ymax=51)
        #plt.show()

        x_lore, y_lore = make_plots(None, lore_df)
        x_dice_1, y_dice_1 = make_plots(None, dice_df.loc[dice_df['distribution'] == 'genetic'], color='green')
        x_dice_2, y_dice_2 = make_plots(None, dice_df.loc[dice_df['distribution'] == 'random'], color='black')
        x_barbe_no, y_barbe_no = make_plots(None, barbe_no_negation_mn_df.loc[barbe_no_negation_mn_df['counter-method'] == 'high-rules'], color='orange')
        x_barbe, y_barbe = make_plots(None, barbe_mn_df.loc[barbe_mn_df['counter-method'] == 'importance-rules'], color='blue')
        len_x = x_vals.shape[0]
        curr_x = 0
        for valx in x_lore:
            x_vals.loc[len_x + curr_x, 'Sum of Normalized Distances'] = valx
            x_vals.loc[len_x + curr_x, 'Approach'] = 'LORE'
            x_vals.loc[len_x + curr_x, 'Dataset'] = bbmodel_name
            curr_x += 1
        for valy in y_lore:
            x_vals.loc[len_x, 'Log10 Density Ratio'] = valy
            len_x += 1

        curr_x = 0
        for valx in x_dice_1:
            x_vals.loc[len_x + curr_x, 'Sum of Normalized Distances'] = valx
            x_vals.loc[len_x + curr_x, 'Dataset'] = bbmodel_name
            x_vals.loc[len_x + curr_x, 'Approach'] = 'DiCE Genetic'
            curr_x += 1
        for valy in y_dice_1:
            x_vals.loc[len_x, 'Log10 Density Ratio'] = valy
            len_x += 1
        curr_x = 0
        for valx in x_dice_2:
            x_vals.loc[len_x + curr_x, 'Sum of Normalized Distances'] = valx
            x_vals.loc[len_x + curr_x, 'Dataset'] = bbmodel_name
            x_vals.loc[len_x + curr_x, 'Approach'] = 'DiCE Random'
            curr_x += 1
        for valy in y_dice_2:
            x_vals.loc[len_x, 'Log10 Density Ratio'] = valy
            len_x += 1

        #curr_x = 0
        #for valx in x_barbe_no:
        #    x_vals.loc[len_x + curr_x, 'Sum of Normalized Distances'] = valx
        #    x_vals.loc[len_x + curr_x, 'Dataset'] = bbmodel_name
        #    x_vals.loc[len_x + curr_x, 'Approach'] = 'BARBE No-Neg'
        #    curr_x += 1
        #for valy in y_barbe_no:
        #    x_vals.loc[len_x, 'Log10 Density Ratio'] = valy
        #    len_x += 1
        curr_x = 0
        for valx in x_barbe:
            x_vals.loc[len_x + curr_x, 'Sum of Normalized Distances'] = valx
            x_vals.loc[len_x + curr_x, 'Dataset'] = bbmodel_name
            x_vals.loc[len_x + curr_x, 'Approach'] = 'BARBE'
            curr_x += 1
        for valy in y_barbe:
            x_vals.loc[len_x, 'Log10 Density Ratio'] = valy
            len_x += 1
        #x_lore['Algorithm'] = 'LORE'
        #x_dice_1['Algorithm'] = 'DiCE Genetic'
        #x_dice_2['Algorithm'] = 'DiCE Random'
        #x_barbe_no['Algorithm'] = 'BARBE No-Neg'
        #x_barbe['Algorithm'] = 'BARBE'
        #x_vals = pd.concat([x_lore, x_dice_1, x_dice_2, x_barbe_no, x_barbe])



    ax = sns.displot(x_vals, x='Sum of Normalized Distances', y='Log10 Density Ratio',
                     hue_order=['DiCE Genetic', 'DiCE Random', 'LORE', 'BARBE'],
                     palette={'BARBE': (0, 114/255, 178/255),
                                #'BARBE-N (fi)': (230/255, 159/255, 0),
                              'DiCE Random': (230 / 255, 159 / 255, 0),
                                #'BARBE': (213/255, 94/255, 0/255),
                                #'No-Neg-BARBE-N (fi)': (213/255, 94/255, 0),
                                'DiCE Genetic': (204/255, 121/255, 167/255),
                                'LORE': (0, 158/255, 115/255)},
                     common_norm=False, hue='Approach', fill=True, kind='kde',
                     col='Dataset', common_bins=False,
                     facet_kws={'sharex': False, 'sharey': False},
                     clip=(-10, 100), alpha=0.7)
        #ax = ax.ax
        #plt.hlines(y=2, xmin=-1, xmax=25)
        #plt.hlines(y=-2, xmin=-1, xmax=25)
        #plt.hlines(y=5, xmin=-1, xmax=25)
        #plt.vlines(x=3, ymin=-6, ymax=51)
        #plt.vlines(x=6, ymin=-6, ymax=51)
        #plt.vlines(x=12, ymin=-6, ymax=51)
        #legend_handles, legend_labels = ax.get_legend_handles_labels()
        #box = ax.get_position()
        # ax.set_position([box.x0 - box.width*0.5, box.y0,# + box.height * 0.5,
        #                 box.width * 2, box.height * 1.5])
        #ax.legend(legend_handles, legend_labels, ncol=3, title='',
        #          loc='center', bbox_to_anchor=(0, -0.6, 1, 0.5), mode='expand')
        # sns.move_legend(ax, 'center', bbox_to_anchor=(0,0))
        #plt.tight_layout()
    plt.show()
    assert False
    '''
    
        if full_df is not None:
            full_df_2 = pd.concat([
                                 barbe_mn_df.loc[barbe_mn_df['counter-method'] == 'importance-rules'],
                                 #barbe_n_df.loc[barbe_n_df['counter-method'] == 'importance-rules'],
                                 barbe_no_negation_mn_df.loc[barbe_no_negation_mn_df['counter-method'] == 'importance-rules'],
                                 #barbe_no_negation_n_df.loc[barbe_no_negation_n_df['counter-method'] == 'importance-rules'],
                                 lore_df,
                                 dice_df.loc[dice_df['distribution'] == 'genetic']
                                 ], ignore_index=True)
            full_df_2['bbmodel'] = bbmodel_name
            full_df = pd.concat([full_df, full_df_2], ignore_index=True)
        else:
            full_df = pd.concat([
                                 barbe_mn_df.loc[barbe_mn_df['counter-method'] == 'importance-rules'],
                                 #barbe_n_df.loc[barbe_n_df['counter-method'] == 'importance-rules'],
                                 barbe_no_negation_mn_df.loc[barbe_no_negation_mn_df['counter-method'] == 'importance-rules'],
                                 #barbe_no_negation_n_df.loc[barbe_no_negation_n_df['counter-method'] == 'importance-rules'],
                                 lore_df,
                                 dice_df.loc[dice_df['distribution'] == 'genetic']
                                 ], ignore_index=True)
            full_df['bbmodel'] = bbmodel_name
    full_df.reset_index(inplace=True)
    full_df = pd.wide_to_long(full_df, ['dens-c-', 'c-hit-'], i='index', j='n_counterfactual')
    #full_df = full_df.loc[full_df['n_counterfactual'].apply(func=lambda x: 'r' not in str(x))]
    full_df = full_df.loc[full_df['c-hit-'].apply(func=lambda x: x > 0)]
    full_df['dens-c-'] = np.log10(full_df['dens-c-'])
    #full_df['c-hit-'] = full_df['c-hit-'].apply(func=lambda x: max(0, x))
    #make_plots(full_df)
    make_point_deviation_plots(full_df, 'method', 'dens-c-')
    plt.show()
    '''
    pass


def make_loan_plots():
    pass


if __name__ == "__main__":
    make_loan_tables(names=['cv_mlp_attrition', 'cv_fieap_loan_acceptance', 'cv_rf_breast_cancer'])
    #make_loan_plots()
