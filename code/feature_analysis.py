import os
import sys
import shap
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


# display options and data I/O directory
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../data')
PLOT_DIR = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0]) or '.'), '../plots/results')

def one_hot_encoding(df, drop=None):
    """
    Transforms categorical variables to dummies.

    https://stackoverflow.com/questions/60153981/scikit-learn-one-hot-encoding-certain-columns-of-a-pandas-dataframe
    https://www.kaggle.com/code/marcinrutecki/one-hot-encoding-everything-you-need-to-know

    :param df: input data with categorical features only
    :param drop: sklearn drop argument for OneHotEncoder
    :return: dataframe with categorical features transformed to dummies
    """
    # transform the categorical features to dummies
    ohe = OneHotEncoder(drop=drop).fit(df)
    dummies = ohe.transform(df).toarray()

    # get the dummy labels
    dlabels = ohe.get_feature_names_out(df.columns)

    # return the output dataframe
    return pd.DataFrame(dummies, columns=dlabels, index=df.index).astype(int)


def stepwise(X_train, X_test, y_train, y_test, categorical=None, core=None):
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
        oos_ma = stepwise_model_accuracy(X_train[cols], X_test[cols], y_train, y_test)
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
            oos_ma = stepwise_model_accuracy(X_train[cols], X_test[cols], y_train, y_test)
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


def stepwise_model_accuracy(X_train, X_test, y_train, y_test):
    """
    Performs a logistic regression with the default L2 regularization
    and estimates model accuracy from the confusion matrix.

    Although a simplified version of the function logistic_regression,
    this function is required because the instantiated classifier object
    doesn't allow a varying number of features in the regression.
    Therefore, a single object instantiation in function stepwise and
    multiple calls to logistic_regression don't work, even though that
    practice results in cleaner code.

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
    estimates model accuracy from the confusion matrix, plots more
    detailed performance metrics and estimates the churn probability
    across all active club members.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
    :return: list of total model and class-specific accuracies
    """
    # estimate model and get its predictions
    lr = LogisticRegression(max_iter=1e4, warm_start=True)
    lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)

    # OOS model/class accuracies from confusion matrix
    accuracies = get_model_performance(y_test, yhat)

    # plot the confusion matrix and the ROC curve
    plot_model_performance(lr, X_test, y_test, filename='conf_mat_roc_curve_lr')

    # predict the probabilities of cancelling for all active members
    churn_probabilities(lr, model='Stepwise Regression (%)', cols=X_test.columns.tolist())

    return accuracies


def get_model_performance(y_test, y_pred, ca_flag=True):
    """
    Estimates the model and class accuracies from the confusion matrix.

    :param y_test: observed values of dependent variable in test set
    :param y_pred: predicted values of dependent variable in test set
    :param ca_flag: flag to estimate class accuracies
    :return: total model and class-specific accuracies
    """
    # estimate the confusion matrix
    conf = confusion_matrix(y_test, y_pred)

    # OOS model/class accuracies from confusion matrix
    accuracy = conf.diagonal().sum() / conf.sum()
    if ca_flag:
        class_accuracy = conf.diagonal() / conf.sum(axis=1)
        # class_accuracy = confusion_matrix(y_test, y_pred, normalize='true').diagonal()
        accuracy = [accuracy] + list(class_accuracy)

    return accuracy


def plot_model_performance(clf, X_test, y_test, filename='conf_mat_roc_curve'):
    """
    Visualizes the confusion matrix and ROC curve for the logistic regression models.

    :param clf: sklearn classifier object
    :param X_test: observed values of features in test set
    :param y_test: observed values of dependent variable in test set
    :param filename: name for the output pdf file
    :return: save the confusion matrix and ROC curve plots
    """
    # plot setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # create plots: confusion matrix (left), ROC curve (right)
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=['CANCELLED', 'INFORCE'], ax=ax1)
    RocCurveDisplay.from_estimator(clf, X_test, y_test, name='Membership Status ROC', ax=ax2)

    # add plot titles
    ax1.set_title('Confusion Matrix for Membership Status')
    ax2.set_title('ROC Curve for Membership Status')

    # save the plot
    fig.savefig(os.path.join(PLOT_DIR, '{}.pdf'.format(filename)), bbox_inches='tight', format='pdf')

    return None


def random_forest_model(X_train, X_test, y_train, y_test):
    """
    Estimates a random forest classifier with model performance stemming
    from the confusion matrix. Model/class accuracies and ROC curve
    metrics are included. The feature importance analysis for this
    classifier is provided by Shapley values.

    :param X_train: training data features
    :param X_test: test data features
    :param y_train: training data dependent variable
    :param y_test: test data dependent variable
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
    accuracies = get_model_performance(y_test, yhat)

    # plot the confusion matrix and the ROC curve
    plot_model_performance(rf, X_test, y_test, filename='conf_mat_roc_curve_rf')

    # print model performance
    print('Random forest OOS results')
    print('Model total accuracy: {}%'.format(round(accuracies[0] * 100, 2)))
    print('Class accuracies (%)')
    print(pd.Series(accuracies[1:], index=['CANCELLED', 'INFORCE']).round(4) * 100)

    # feature importance with Shapley values
    shapley_feature_importance(rf, X_test)

    # predict the probabilities of cancelling for all active members
    churn_probabilities(rf, model='Random Forest (%)')

    return accuracies


def shapley_feature_importance(clf, X_test):
    """
    Estimates and plots the mean Shapley value for each feature,
    which reflects the feature's average impact on model output
    magnitude.

    Long feature labels have smaller font size than the rest.
    https://stackoverflow.com/questions/62974417/change-font-size-of-single-tick-matplotlib-python

    :param clf: sklearn classifier object
    :param X_test: test data features
    :return: None
    """
    # estimate the Shapley values with dummies in the test set
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # visualize feature importance results with a bar plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10,
                      class_names=['CANCELLED', 'INFORCE'], show=False)
    title = 'Feature Importance Scores for Random Forest Classifier\n'\
            '(Average Impact on Model Output Magnitude)'
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
    ax.set_ylabel('Top-10 Features', fontsize=14)

    # reduce the font size of feature labels that exceed a threshold
    yticklabels = ax.get_yticklabels()
    fontsizes = [12 if len(label.get_text()) <= 20 else 9 for label in yticklabels]
    for tick, size in zip(yticklabels, fontsizes):
        tick.set_fontsize(size)

    # save the plot
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'shapley_rf.pdf'), bbox_inches='tight', format='pdf')


def naive_benchmarks(y_test):
    """
    Estimates the performance of two naive benchmark models.
    1) Nobody cancels: yhat = 1 for all members
    2) Flip a coin for each member: sample from uniform distribution for
       each member separately and map to binary decision (cancel/stay).

    Perform the analysis on the test data only for consistency with OOS
    predictions from the stepwise regression and random forest models.

    :param y_test: test data dependent variable
    :return: benchmark model/class performance metrics
    """
    # initialize container
    accuracies = []

    # model 1: nobody cancels
    yhat = np.ones_like(y_test)
    accuracies.append(get_model_performance(y_test, yhat))

    # model 2: flip a coin
    yhat = np.random.uniform(size=len(y_test))
    yhat = np.where(yhat < 0.5, 0, 1)
    accuracies.append(get_model_performance(y_test, yhat))

    # create the dataframe for churn probabilities among active members
    df = pd.read_csv(os.path.join(DATA_DIR, 'encoded_features.csv'), usecols=['id', 'status'])
    df = df[df['status'] == 'INFORCE'].drop(columns=['status'])
    df['Nobody Cancels (%)'], df['Flip a Coin (%)'] = 0, 50
    df.to_csv(os.path.join(DATA_DIR, 'churn_probabilities.csv'), index=False)

    return accuracies


def churn_probabilities(clf, model='Naive benchmark (%)', cols=None):
    """
    Predicts the probability of cancelling for each active member
    and updates the output file with the corresponding probabilities
    across different models.

    :param clf: sklearn classifier object
    :param model: model name
    :param cols: list of features used during the fitting process (default: all)
    :return: None
    """
    # read the data with dummy-encoded features and churn probabilities
    df = pd.read_csv(os.path.join(DATA_DIR, 'encoded_features.csv'))
    out = pd.read_csv(os.path.join(DATA_DIR, 'churn_probabilities.csv'))

    # list of features that were fitted (empty: use all features)
    if cols is None:
        drop_cols = ['status']
    else:
        drop_cols = [col for col in df.columns if col not in (['id'] + cols)]

    # restrict to active members only
    df = df[df['status'] == 'INFORCE'].drop(columns=drop_cols)

    # estimate the churn probabilities for each member
    probs = clf.predict_proba(df.drop(columns=['id']))[:, 0].round(4) * 100

    # save the probabilities for this model
    dft = pd.DataFrame({'id': df['id'], model: probs})
    out = pd.merge(out, dft, on='id')
    out.to_csv(os.path.join(DATA_DIR, 'churn_probabilities.csv'), index=False)


def results_summary(accuracies):
    """
    Summarizes and saves the main results from feature analysis.

    :param accuracies: model performance metrics
    :return: None
    """
    # save model performances
    index = ['Nobody Cancels', 'Flip a Coin', 'Stepwise Regression', 'Random Forest']
    cols = ['Total accuracy (%)', 'CANCELLED (%)', 'INFORCE (%)']
    out = pd.DataFrame(accuracies, index=index, columns=cols).round(4) * 100
    out = out.rename_axis('Model').reset_index()
    out.to_csv(os.path.join(DATA_DIR, 'model_performances.csv'), index=False)
    print('\nModel performances')
    print(out)

    # churn probability summary stat across models
    prc = [0.01, 0.25, 0.5, 0.75, 0.99]
    filename = os.path.join(DATA_DIR, 'churn_probabilities.csv')
    dft = pd.read_csv(filename).drop(columns=['id'])
    dfs = [dft.describe(percentiles=prc),
           pd.DataFrame(dft.skew()).T.rename(index={0: 'skewness'}),
           pd.DataFrame(dft.kurt()).T.rename(index={0: 'kurtosis'})]
    stat = pd.concat(dfs).T.rename_axis('Model').reset_index().drop(columns='count').round(2)
    stat.columns = [s if s.startswith('AR') else s.capitalize() for s in stat.columns]
    stat.to_csv(os.path.join(DATA_DIR, 'summ_stat_churn_prob.csv'), index=False)
    print('\nChurn probability summary statistics')
    print(stat, '\n')


def feature_analysis():
    """
    Driver for the feature analysis (TBD)

    :return: None
    """
    # read the filtered data and save the categorical feature labels
    df = pd.read_csv(os.path.join(DATA_DIR, 'filtered_data.csv'))
    cols_cat = df.select_dtypes(exclude=np.number).drop(columns=['id', 'status']).columns.tolist()

    # create dummies for categorical features and merge with the data
    # dummies = one_hot_encoding(df[cols_cat], drop='first')
    dummies = one_hot_encoding(df[cols_cat])
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
    accuracies = naive_benchmarks(y_test)

    # logistic regression
    res = stepwise(X_train, X_test, y_train, y_test, categorical=cols_cat)
    accuracies.append(res)

    # random forest
    res = random_forest_model(X_train, X_test, y_train, y_test)
    accuracies.append(res)

    # summarize and output main results
    results_summary(accuracies)

    # backtest during cross-validation
    # core = ['marital_status', 'log_annual_fees']
    # dum_drop = [f for f in X_train.columns if f.startswith('term_years')
    #              or f.startswith('package') or f.startswith('duration')]
    # stepwise(X_train.drop(columns=dum_drop), X_test.drop(columns=dum_drop),
    #          y_train, y_test, categorical=cols_cat, core=core)


