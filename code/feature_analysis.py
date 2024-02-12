import os
import sys
import warnings
import numpy as np
import pandas as pd
import aux_feature_analysis as fa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# display options and data I/O directory
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')


def stepwise(X_train, X_test, y_train, y_test, categorical=None, core=None, output=True):
    """
    Performs a customized forward stepwise regression for model estimation
    and feature importance. The algorithm can handle dummy variables provided
    that they have a common prefix in their name.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :param categorical: list of categorical features (aggregated names)
    :param core: list of features that are enforced in every model iteration
    :param output: flag that controls the amount of generated output
    :return: model performance metrics
    """
    # print a message and initialize the minimum accuracy improvement threshold
    print('\nRunning the stepwise logistic regression classifier')
    min_accuracy = 0.01

    # re-validate the categoricals list, and initialize the lists of
    # granular features and (dummy-aggregated) candidate features
    granular_features = X_train.columns.tolist()
    if categorical:
        categorical = [f for f in categorical if any(f in prefix for prefix in granular_features)]
        numeric = [f for f in granular_features if not any(prefix in f for prefix in categorical)]
        candidate_feature_pool = numeric + categorical
    else:
        candidate_feature_pool = granular_features

    # initialize internal containers for the stepwise logic
    selected_features, accuracy_scores = [], []
    accuracy_with_candidate_features = []

    # remove from the candidate features those that are enforced in
    # every model (core features) and find their baseline performance
    if core:
        print('Core Features:', core)
        candidate_feature_pool = [f for f in candidate_feature_pool if f not in core]
    else:
        core = []

    # single run per candidate feature to restrict to those
    # that contribute individually above the min threshold
    for candidate_feature in candidate_feature_pool:
        prefixes = core + [candidate_feature]
        cols = [f for f in granular_features if any(prefix in f for prefix in prefixes)]
        oos_ma = fa.stepwise_model_accuracy(X_train[cols], X_test[cols], y_train, y_test)
        accuracy_with_candidate_features.append((oos_ma, candidate_feature))
    remaining_features = [tup[1] for tup in accuracy_with_candidate_features
                          if tup[0] >= min_accuracy]

    # perform stepwise regression
    current_oos_ma, best_new_oos_ma, delta_oos_ma = 0.0, 0.0, -1.0
    while (remaining_features and current_oos_ma == best_new_oos_ma and
           delta_oos_ma != 0.0):
        accuracy_with_candidate_features = []
        for candidate_feature in remaining_features:
            prefixes = core + selected_features + [candidate_feature]
            cols = [f for f in granular_features if any(prefix in f for prefix in prefixes)]
            oos_ma = fa.stepwise_model_accuracy(X_train[cols], X_test[cols], y_train, y_test)
            accuracy_with_candidate_features.append((oos_ma, candidate_feature))

        accuracy_with_candidate_features.sort()
        best_new_oos_ma, best_candidate = accuracy_with_candidate_features.pop()
        delta_oos_ma = best_new_oos_ma - current_oos_ma
        if delta_oos_ma > min_accuracy:
            current_oos_ma = best_new_oos_ma
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            accuracy_scores.append([delta_oos_ma, current_oos_ma])

    # get the best-performing basket of features with its OOS model accuracy
    if selected_features:
        prefixes = core + selected_features
        cols = [f for f in granular_features if any(prefix in f for prefix in prefixes)]
        accuracies = logistic_regression(X_train[cols], X_test[cols], y_train, y_test)
    elif core:
        cols = [f for f in granular_features if any(prefix in f for prefix in core)]
        accuracies = logistic_regression(X_train[cols], X_test[cols], y_train, y_test)
        selected_features, accuracy_scores = ['Core Features'], [[0.0, accuracies[0]]]
    else:
        accuracies = [0.0, 0.0, 0.0]
        selected_features, accuracy_scores = ['None'], [[0.0, 0.0]]

    if output:
        # print model performance
        print('Stepwise logistic regression OOS results')
        print('Model total accuracy: {}%'.format(round(accuracies[0] * 100, 2)))
        print('Class accuracies (%)')
        print(pd.Series(accuracies[1:], index=['CANCELLED', 'INFORCE']).round(4) * 100)

        # print feature selection results
        print('\nFeature importance for stepwise regression (top-to-bottom)')
        cols = ['Accuracy Improvement (%)', 'Cumulative Accuracy (%)']
        out = pd.DataFrame(accuracy_scores, index=selected_features, columns=cols)
        out = (out.round(4) * 100).rename_axis('Feature').reset_index()
        out.to_csv(os.path.join(DATA_DIR, 'stepwise_feature_importance.csv'), index=False)
        print(out, '\n')

    return accuracies


def logistic_regression(X_train, X_test, y_train, y_test, output=True):
    """
    Performs a logistic regression with the default L2 regularization,
    estimates model accuracy from the confusion matrix, plots more
    detailed performance metrics and estimates the churn probability
    across all active club members.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :param output: flag that controls the amount of generated output
    :return: list of total model and class-specific accuracies
    """
    # estimate model and get its predictions
    lr = LogisticRegression(max_iter=1e4, warm_start=True)
    lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)

    # OOS model/class accuracies from confusion matrix
    accuracies = fa.get_model_performance(y_test, yhat)

    if output:
        # plot the confusion matrix and the ROC curve
        fa.plot_model_performance(lr, X_test, y_test, filename='conf_mat_roc_curve_lr')

        # predict the probabilities of cancelling for all active members
        fa.churn_probabilities(lr, model='Stepwise Regression (%)', cols=X_test.columns.tolist())

    return accuracies


def random_forest_model(X_train, X_test, y_train, y_test, output=True):
    """
    Estimates a random forest classifier with model performance stemming
    from the confusion matrix. Model/class accuracies and ROC curve
    metrics are included. The feature importance analysis for this
    classifier is provided by Shapley values.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :param output: flag that controls the amount of generated output
    :return: model performance metrics
    """
    # print a message
    print('Running the random forest classifier')

    # estimate model and get its predictions
    rf = RandomForestClassifier(n_estimators=15, max_depth=2, ccp_alpha=0.02,
                                random_state=33, class_weight='balanced_subsample')
    rf.fit(X_train, y_train)
    yhat = rf.predict(X_test)

    # OOS model/class accuracies from confusion matrix
    accuracies = fa.get_model_performance(y_test, yhat)

    if output:
        # plot the confusion matrix and the ROC curve
        fa.plot_model_performance(rf, X_test, y_test, filename='conf_mat_roc_curve_rf')

        # print model performance
        print('Random forest OOS results')
        print('Model total accuracy: {}%'.format(round(accuracies[0] * 100, 2)))
        print('Class accuracies (%)')
        print(pd.Series(accuracies[1:], index=['CANCELLED', 'INFORCE']).round(4) * 100)

        # feature importance with Shapley values
        fa.shapley_feature_importance(rf, X_test)

        # predict the probabilities of cancelling for all active members
        fa.churn_probabilities(rf, model='Random Forest (%)')

    return accuracies


def feature_analysis():
    """
    Driver for the feature analysis (TBD)

    :return: None
    """
    # read the filtered data and save the categorical feature labels
    df = pd.read_csv(os.path.join(DATA_DIR, 'filtered_data.csv'))
    cols_cat = df.select_dtypes(exclude=np.number).drop(columns=['id', 'status']).columns.tolist()

    # create dummies for categorical features and merge with the data
    # dummies = fa.one_hot_encoding(df[cols_cat], drop='first')
    dummies = fa.one_hot_encoding(df[cols_cat])
    df = pd.concat([df.drop(columns=cols_cat), dummies], axis=1)

    # drop missing values, neither type of estimator can handle them
    df = df.dropna()

    # save a copy of the encoded feature data
    df.to_csv(os.path.join(DATA_DIR, 'encoded_features.csv'), index=False)

    # encode the dependent variable status
    df['y'] = df['status'].map({'CANCELLED': 0, 'INFORCE': 1})

    # stratified split into training and test sets
    X = df.drop(columns=['id', 'status', 'y'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, df['y'], stratify=df['y'], test_size=0.2, random_state=12)

    # initialize the model performance container with the naive benchmarks
    accuracies = fa.naive_benchmarks(y_test)

    # logistic regression
    res = stepwise(X_train, X_test, y_train, y_test, categorical=cols_cat)
    accuracies.append(res)

    # random forest
    res = random_forest_model(X_train, X_test, y_train, y_test)
    accuracies.append(res)

    # summarize and output main results
    fa.results_summary(accuracies)

    # backtest during cross-validation
    # core = ['marital_status', 'log_annual_fees']
    # dum_drop = [f for f in X_train.columns if f.startswith('term_years')
    #              or f.startswith('package') or f.startswith('duration')]
    # stepwise(X_train.drop(columns=dum_drop), X_test.drop(columns=dum_drop),
    #          y_train, y_test, categorical=cols_cat, core=core)


