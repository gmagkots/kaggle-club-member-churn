import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


# display options and data I/O directory
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')
PLOT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../plots/results')

def one_hot_encoding(df):
    """
    Transforms categorical variables to dummies.

    https://stackoverflow.com/questions/60153981/scikit-learn-one-hot-encoding-certain-columns-of-a-pandas-dataframe
    https://www.kaggle.com/code/marcinrutecki/one-hot-encoding-everything-you-need-to-know

    :param df: input data with categorical features only
    :return: dataframe with categorical features transformed to dummies
    """
    # transform the categorical features to dummies
    ohe = OneHotEncoder().fit(df)
    dummies = ohe.transform(df).toarray()

    # get the dummy labels
    dlabels = ohe.get_feature_names_out(df.columns)

    # return the output dataframe
    return pd.DataFrame(dummies, columns=dlabels, index=df.index).astype(int)


def stepwise(X_train, X_test, y_train, y_test, features=None):
    """
    Performs a customized forward stepwise regression
    for model estimation and feature importance.

    :param df: dataframe with input features and regressors
    :param yvar: dependent variable label
    :return: dictionary with the summary of results
    """
    # minimum accuracy improvement threshold and feature containers
    min_accuracy = 0.01
    selected_features = []
    candidate_feature_pool = ['day', 'hour', 'minute', 'second', 'log(spread)',
                              'dspread', 'imbalance', 'ticker',
                              'laglog(notional)', 'laglogdt']

    # append timestamp to yvar (for train/test split) and set dummy labels
    core = ['timestamp', yvar]
    dummies = ['D_ITXEB5', 'D_ITXES5']

    # single run per candidate feature to restrict to those
    # that contribute individually above the min threshold
    oos_r2s_with_candidate_features = []
    for candidate_feature in candidate_feature_pool:
        if candidate_feature != 'ticker':
            dft = df[core + [candidate_feature]]
        else:
            dft = df[core + dummies]
        oos_r2 = get_oos_r2(dft, yvar)
        oos_r2s_with_candidate_features.append((oos_r2, candidate_feature))
    remaining_features = [tup[1] for tup in oos_r2s_with_candidate_features
                          if tup[0] >= min_accuracy]

    # perform stepwise regression
    current_oos_r2, best_new_oos_r2, delta_oos_r2 = 0.0, 0.0, -1.0
    while (remaining_features and current_oos_r2 == best_new_oos_r2 and
           delta_oos_r2 != 0.0):
        oos_r2s_with_candidate_features = []
        for candidate_feature in remaining_features:
            if candidate_feature != 'ticker':
                dft = df[core + selected_features + [candidate_feature]]
            else:
                dft = df[core + selected_features + dummies]
            oos_r2 = get_oos_r2(dft, yvar)
            oos_r2s_with_candidate_features.append((oos_r2, candidate_feature))

        oos_r2s_with_candidate_features.sort()
        best_new_oos_r2, best_candidate = oos_r2s_with_candidate_features.pop()
        delta_oos_r2 = best_new_oos_r2 - current_oos_r2
        if delta_oos_r2 > min_accuracy:
            current_oos_r2 = best_new_oos_r2
            remaining_features.remove(best_candidate)
            if best_candidate != 'ticker':
                selected_features.append(best_candidate)
            else:
                selected_features.extend(dummies)

    # get the best-performing basket of features with its OOS R-squared
    if selected_features:
        df_final = df[core + selected_features]
        max_oos_r2 = get_oos_r2(df_final, yvar)
    else:
        selected_features = 'None'
        max_oos_r2 = 0.0

    # print results
    print('\033[1m' + 'Stepwise regression results for ' + yvar + '\033[0m')
    print('Model performance metric (OOS R-squared): {}'.format(max_oos_r2))
    print('Feature importance (left-to-right): {}'.format(selected_features))



def logistic_regression(X_train, X_test, y_train, y_test):
    """
    Performs two logistic regressions on the data. The first regression
    uses L1 regularization for feature selection. The second one uses
    L2 regularization for model estimation.

    The model selection process compares two models: one using the full
    set of features and another restricted to those suggested by the
    feature selection following the L1 regularization.

    Model performance stems from the accuracies estimated by confusion
    matrix analysis.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :return: None
    """
    # # feature selection with L1 regularization
    # lr = LogisticRegression(penalty='l1', C=0.01, fit_intercept=True, #class_weight='balanced',
    #                         random_state=123, solver='saga', max_iter=1e6, warm_start=True)
    # lr.fit(X_train, y_train)
    #
    # # show the redundant features following L1 regularization
    # coef = pd.Series(data=lr.coef_[0], index=lr.feature_names_in_)
    # print('Logistic regression, L1-suggested redundant features:')
    # print(coef[coef==0])

    ##########################################
    # model estimation with L2 regularization
    ##########################################

    # estimator setup
    lr = LogisticRegression(max_iter=1e4, warm_start=True, fit_intercept=False)  # class_weight='balanced')

    # model with full set of features
    cols = ['log_annual_fees', 'additional_members', 'gender_M', 'duration',
            'payment_mode_ANNUAL', 'payment_mode_MONTHLY', 'payment_mode_SEMI-ANNUAL'] + \
           [col for col in X_train.columns if col.startswith('term_years')]
    X_train = X_train.drop(columns=cols)
    X_test = X_test.drop(columns=cols)
    lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)

    # OOS model performance metrics (R-squared and model/class accuracies)
    R2 = OLS(y_test, add_constant(yhat), missing='drop').fit().rsquared.round(2)
    conf = confusion_matrix(y_test, yhat)
    model_accuracy = (conf.diagonal().sum() / conf.sum() * 100).round(2)
    class_accuracy = (conf.diagonal() / conf.sum(axis=1) * 100).round(2)
    # class_accuracy = (confusion_matrix(y_test, yhat, normalize='true').diagonal() * 100).round(4)
    class_accuracy = pd.Series(class_accuracy, index=['CANCELLED', 'INFORCE'])

    # print results
    print('Logistic regression OOS model performance (all features)')
    print('R-squared: {}'.format(R2))
    print('Model total accuracy: {}%'.format(model_accuracy))
    print('Class accuracies (%)')
    print(class_accuracy)

    # plot the confusion matrix and the ROC curve
    plot_model_performance(y_test, yhat)

    return None

def plot_model_performance(y_true, y_pred):
    """
    Visualizes the confusion matrix and ROC curve for the logistic regression models.

    :param y_true: observed values of dependent variable in test set
    :param y_pred: predicted values of dependent variable in test set
    :return: save the confusion matrix and ROC curve plots
    """
    # plot setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # create plots: confusion matrix (left), ROC curve (right)
    # ConfusionMatrixDisplay(confusion_matrix=conf, display_labels = ['CANCELLED', 'INFORCE']).plot(ax=ax1)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['CANCELLED', 'INFORCE'], ax=ax1)
    RocCurveDisplay.from_predictions(y_true, y_pred, name='Membership Status ROC', ax=ax2)

    # add plot titles
    ax1.set_title('Confusion Matrix for Membership Status')
    ax2.set_title('ROC Curve for Membership Status')

    # save the plot
    fig.savefig(os.path.join(PLOT_DIR, 'conf_mat_roc_curve.pdf'), bbox_inches='tight', format='pdf')

    return None


def random_forest_model(df):
    """

    :param df:
    :return:
    """

    return None


def feature_analysis():
    """
    Driver for the feature analysis (TBD)

    :return: None
    """
    # read the filtered data and save the categorical feature labels
    df = pd.read_csv(os.path.join(DATA_DIR, 'filtered_data.csv'))
    cols_cat = df.select_dtypes(exclude=np.number).drop(columns=['id', 'status']).columns.tolist()

    # create dummies for categorical features, merge
    # with the data and save the dummy labels
    dummies = one_hot_encoding(df[cols_cat])
    df = pd.concat([df.drop(columns=cols_cat), dummies], axis=1)
    cols_dummies = dummies.columns.tolist()

    # encode the dependent variable status
    df['y'] = df['status'].map({'CANCELLED': 0, 'INFORCE': 1})

    # drop missing values, both types of estimators can't handle them
    df = df.dropna()

    # stratified split into training and test sets
    X = df.drop(columns=['id', 'status', 'y'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, df['y'], stratify=df['y'], test_size=0.2, random_state=12)

    # logistic regression (manually drop one dummy per feature)
    dum_drop = ['gender_F', 'marital_status_UNKNOWN', 'package_TYPE-A',
                'payment_mode_QUARTERLY', 'term_years_32']
    logistic_regression(X_train.drop(columns=dum_drop),
                        X_test.drop(columns=dum_drop),
                        y_train, y_test)


    # print(cols_dummies)
    # print(X_train.head())
    # print(y_train.head())
    # print(df.info())

