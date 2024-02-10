import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# mpl.use('Agg')
mpl.use('Qt5Agg')
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')
PLOT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../plots/input_data')


def raw_data_etl():
    """
    Reads and processes the raw imput data, generates histogram plots,
    estimates summary statistics and saves the filtered data.

    :return: None
    """
    # load the spreadsheet and modify the column names
    df = pd.read_excel(os.path.join(DATA_DIR, 'assignment.xlsx'),
                       sheet_name='Data', parse_dates=[13, 14])
    df.columns = [col.lower().replace('member_', '').replace('membership_', '') for col in df.columns]
    df = df.rename(columns={'number': 'id'}).drop(columns=['agent_code'])

    # create the start year feature
    df['start_year'] = df['start_date'].dt.year.astype('int64')

    # set the reference date and calculate membership duration (years)
    ref_date = pd.to_datetime('2013-12-1')
    df.loc[df['end_date'].isna(), 'end_date'] = ref_date
    df['duration'] = (df['end_date'] - df['start_date']) / np.timedelta64(1, 'D') / 365

    # drop single-payment subscriptions, they won't cancel and their fees may be outliers
    df = df[df['payment_mode'] != 'SINGLE-PREMIUM']

    # log-transform the numeric features to mitigate scale effects
    df[['annual_fees', 'annual_income']] = df[['annual_fees', 'annual_income']].transform(np.log)
    df = df.rename(columns={'annual_fees': 'log_annual_fees', 'annual_income': 'log_annual_income'})

    # drop members who are not adults and create custom age ranges
    df = df[df['age_at_issue'] >= 18]
    bins = [(0, 30), (30, 40), (40, 45), (45, 50), (50, 55), (55, 65), (65, np.inf)]
    df['age_range'] = pd.cut(df['age_at_issue'], pd.IntervalIndex.from_tuples(bins))\
        .astype(str).str.replace('.0', '').str.replace('(65, inf]', '>65')

    ################################################################
    # categorical feature re-binning to mitigate large asymmetries
    # and imbalances in values that may result in model overfit
    ################################################################

    # drop missing gender to mitigate unbalanced data side effects
    df = df[~df['gender'].isna()]

    # missing occupation values can be relevant to the model and interpretable
    # (e.g. a member with unknown employment may be more likely to cancel)
    df['occupation_cd_(unadjusted)'] = df['occupation_cd']
    df.loc[df['occupation_cd'] != 1, 'occupation_cd'] = 0
    df['occupation_cd'] = df['occupation_cd'].astype('int64')
    df['occupation_cd_(unadjusted)'] = df['occupation_cd_(unadjusted)'].astype('str').str.replace('.0', '')

    # term_years clusters in the top 5 most frequent values and the rest
    # follow a skewed continuous distribution, re-bin these values
    df['term_years_(unadjusted)'] = df['term_years']
    unclustered = ~df['term_years'].isin([12, 17, 22, 27, 32])
    df.loc[(df['term_years'] < 50) & unclustered, 'term_years'] = 49
    df.loc[df['term_years'] >= 50, 'term_years'] = 51
    df['term_years'] = df['term_years'].astype('str')
    df['term_years'] = df['term_years'].str.replace('49', '<50').str.replace('51', '>50')

    # married members dominate the sample, re-bin to married/single/unknown
    df['marital_status_(unadjusted)'] = df['marital_status']
    df.loc[df['marital_status'].isin(['D', 'W']), 'marital_status'] = 'S'
    df.loc[df['marital_status'].isna(), 'marital_status'] = 'UNKNOWN'

    ################################################################

    # enforce object columns to strings (some involve mixed number/str)
    scols = df.select_dtypes(include=['object']).columns
    df[scols] = df[scols].map(str)

    # create histograms across all features
    plot_histograms(df)

    # estimate summary statistics
    summary_statistics(df)

    # clean up and save the filtered data
    df = df.drop(columns=[col for col in df.columns if '(unadjusted)' in col] +
                         ['start_date', 'end_date', 'age_range'])
    df.to_csv(os.path.join(DATA_DIR, 'filtered_data.csv'), index=False)


def plot_histograms(df):
    """
    Provides histogram plots of the processed input data. The bar
    plot centralizes the xtick labels better than matplotlib's hist.
    See also https://stackoverflow.com/questions/23246125/how-to-center-labels-in-histogram-plot

    :param df: dataframe with processed input data
    :return: saves the histogram plots
    """
    # drop certain features
    df = df.drop(columns=['id', 'start_date', 'end_date'])

    # iterate across all features
    for col in df.columns:
        # plot setup
        fig, ax = plt.subplots()

        # # configure bins and create the plot
        # if df[col].dtype == 'float':
        #     counts, labels = np.histogram(df.loc[~df[col].isna(), col], bins=20)
        #     labels = labels[1:]
        # else:
        #     labels, counts = np.unique(df[col], return_counts=True)
        #     if type(labels[0]) == 'object':
        #         labels = [x.replace('_', ' ').title() for x in labels]
        # counts = counts / counts.sum()
        # ax.bar(labels, counts, align='center')

        # configure bins and create the plot
        if df[col].dtype == 'float':
            counts, labels = np.histogram(df.loc[~df[col].isna(), col], bins=20)
            labels = labels[1:]
            counts = counts / counts.sum()
            ax.bar(labels, counts, align='center')
        else:
            if col == 'status':
                labels, counts = np.unique(df[col], return_counts=True)
                counts = counts / counts.sum()
                ax.bar(labels, counts, align='center')
            else:
                dft = pd.crosstab(index=df[col], columns=df['status'], normalize=True)
                labels = dft.index
                ax.bar(labels, dft['INFORCE'], label='INFORCE', align='center')
                ax.bar(labels, dft['CANCELLED'], label='CANCELLED', align='center', bottom=dft['INFORCE'])
                ax.legend()

        # beautify plot
        lcol = col.replace('_', ' ').title().replace('Cd', 'Code').replace('At', 'at')
        if col in ['additional_members', 'occupation_cd']:
            ax.set_xticks(labels)
        ax.set_ylim(top=ax.get_yticks().max() * 1.03)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 1))
        ax.set_title("Distribution of {} ".format(lcol))
        ax.set_xlabel(lcol)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        # save the plot
        fig.savefig(os.path.join(PLOT_DIR, '{}.pdf'.format(col).replace('(','').replace(')','')),
                    bbox_inches='tight', format='pdf')


def summary_statistics(df):
    """
    Provides summary statistics of the processed input data.

    :param df: dataframe with processed data
    :return: prints and saves the summary statistics tables
    """
    # drop the member IDs and create variable for
    # duration among cancelled subscriptions
    df = df.drop(columns=['id'])
    # df['duration_cancelled'] = np.where(df['status'] == 'CANCELLED', df['duration'], np.nan)
    # df['duration_inforce'] = np.where(df['status'] == 'INFORCE', df['duration'], np.nan)

    # summary statistics of numerical features
    prc = [0.01, 0.25, 0.5, 0.75, 0.99]
    dft = df.select_dtypes(include=np.number)
    dfs = [dft.describe(percentiles=prc),
           pd.DataFrame(dft.skew()).T.rename(index={0: 'skewness'}),
           pd.DataFrame(dft.kurt()).T.rename(index={0: 'kurtosis'})]
    stat_float = pd.concat(dfs).T.rename_axis('Variable').reset_index().drop(columns='count')
    stat_float.columns = [s if s.startswith('AR') else s.capitalize() for s in stat_float.columns]
    print('\nSummary statistics (numerical features, full sample).')
    print(stat_float)
    stat_float.to_csv(os.path.join(DATA_DIR, 'summ_stat_num_all.csv'), index=False)

    # summary statistics of key categorical features for cancelled memberships
    cols = ['age_range', 'term_years']
    stat_cat = df.loc[df['status'] == 'CANCELLED', cols]\
        .value_counts(normalize=True).map('{:.2%}'.format)\
        .rename('frequency').reset_index()
    print('\nSummary statistics (categorical features, cancelled memberships).')
    print(stat_cat)
    stat_cat.to_csv(os.path.join(DATA_DIR, 'summ_stat_cat_cancelled.csv'), index=False)
