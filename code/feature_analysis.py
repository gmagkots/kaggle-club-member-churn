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


def stepwise(X_train, X_test, y_train, y_test, categorical=None, core=None):
    """
    Performs a customized forward stepwise regression
    for model estimation and feature importance.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :param categorical: list of categorical features (aggregated names)
    :param core: list of features that are enforced in every model iteration
    :return: None
    """
    # print a message and set the minimum accuracy improvement threshold
    print('\nRunning stepwise logistic regression')
    min_accuracy = 0.01

    # initialize the granular features list and the
    # list of (dummy-aggregated) candidate features
    granular_features = X_train.columns.tolist()
    if categorical:
        numeric = [f for f in granular_features if not any(prefix in f for prefix in categorical)]
        candidate_feature_pool = numeric + categorical
    else:
        candidate_feature_pool = granular_features

    # remove from the candidate features those that are enforced in every model (core features),
    # then update the core list with the granular list of feature names (incl dummies, if any)
    if core:
        candidate_feature_pool = [f for f in candidate_feature_pool if f not in core]
        core = [f for f in granular_features if any(prefix in f for prefix in core)]
    else:
        core = []

    # single run per candidate feature to restrict to those
    # that contribute individually above the min threshold
    selected_features = []
    accuracy_with_candidate_features = []
    for candidate_feature in candidate_feature_pool:
        cols = core + [f for f in granular_features if f.startswith(candidate_feature)]
        oos_ma = get_model_accuracy(X_train[cols], X_test[cols], y_train, y_test)
        accuracy_with_candidate_features.append((oos_ma, candidate_feature))
    remaining_features = [tup[1] for tup in accuracy_with_candidate_features
                          if tup[0] >= min_accuracy]

    # perform stepwise regression
    current_oos_ma, best_new_oos_ma, delta_oos_ma = 0.0, 0.0, -1.0
    while (remaining_features and current_oos_ma == best_new_oos_ma and
           delta_oos_ma != 0.0):
        accuracy_with_candidate_features = []
        for candidate_feature in remaining_features:
            cols = (core +
                    [f for f in granular_features if any(prefix in f for prefix in selected_features)] +
                    [f for f in granular_features if f.startswith(candidate_feature)])
            oos_ma = get_model_accuracy(X_train[cols], X_test[cols], y_train, y_test)
            accuracy_with_candidate_features.append((oos_ma, candidate_feature))

        accuracy_with_candidate_features.sort()
        best_new_oos_ma, best_candidate = accuracy_with_candidate_features.pop()
        delta_oos_ma = best_new_oos_ma - current_oos_ma
        if delta_oos_ma > min_accuracy:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_oos_ma = best_new_oos_ma

    # get the best-performing basket of features with its OOS model accuracy
    if selected_features:
        cols = core + [f for f in granular_features if any(prefix in f for prefix in selected_features)]
        max_oos_ma, class_accuracy = logistic_regression(X_train[cols], X_test[cols], y_train, y_test)
    else:
        selected_features = 'None'
        max_oos_ma, class_accuracy = 0.0, 0.0

    # print results
    print('Stepwise logistic regression OOS results')
    print('Model total accuracy: {}%'.format(max_oos_ma))
    print('Class accuracies (%)')
    print(class_accuracy)
    print('Feature importance (left-to-right): {}'.format(selected_features))


def get_model_accuracy(X_train, X_test, y_train, y_test):
    """
    Performs a logistic regression with the default L2 regularization
    and estimates model accuracy from the confusion matrix.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :return: OOS model accuracy on the test data
    """
    # estimate model and get its predictions
    lr = LogisticRegression(max_iter=1e4, warm_start=True)
    lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)

    # return OOS model accuracy
    conf = confusion_matrix(y_test, yhat)
    model_accuracy = conf.diagonal().sum() / conf.sum()

    return model_accuracy


def logistic_regression(X_train, X_test, y_train, y_test):
    """
    Performs a logistic regression with the default L2 regularization,
    estimates model accuracy from the confusion matrix and plots more
    detailed performance metrics.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :return: total model accuracy and pandas series of class-specific accuracies
    """
    # estimate model and get its predictions
    lr = LogisticRegression(max_iter=1e4, warm_start=True)  # class_weight='balanced')
    lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)

    # OOS model performance metrics (R-squared and model/class accuracies)
    conf = confusion_matrix(y_test, yhat)
    model_accuracy = (conf.diagonal().sum() / conf.sum() * 100).round(2)
    class_accuracy = (conf.diagonal() / conf.sum(axis=1) * 100).round(2)
    # class_accuracy = (confusion_matrix(y_test, yhat, normalize='true').diagonal() * 100).round(4)
    class_accuracy = pd.Series(class_accuracy, index=['CANCELLED', 'INFORCE'])

    # # print results
    # print('Stepwise logistic regression OOS model performance')
    # print('Model total accuracy: {}%'.format(model_accuracy))
    # print('Class accuracies (%)')
    # print(class_accuracy)

    # plot the confusion matrix and the ROC curve
    plot_model_performance(y_test, yhat)

    return model_accuracy, class_accuracy


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
    # cols_dummies = dummies.columns.tolist()

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
    stepwise(X_train.drop(columns=dum_drop), X_test.drop(columns=dum_drop),
             y_train, y_test, categorical=cols_cat, core=[])


