import os
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


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
    df = pd.read_csv(filename).drop(columns=['id'])
    dfs = [df.describe(percentiles=prc),
           pd.DataFrame(df.skew()).T.rename(index={0: 'skewness'}),
           pd.DataFrame(df.kurt()).T.rename(index={0: 'kurtosis'})]
    stat = pd.concat(dfs).T.rename_axis('Model').reset_index().drop(columns='count').round(2)
    stat.columns = [s if s.startswith('AR') else s.capitalize() for s in stat.columns]
    stat.to_csv(os.path.join(DATA_DIR, 'summ_stat_churn_prob.csv'), index=False)
    print('\nChurn probability summary statistics')
    print(stat, '\n')

    # churn probability histograms across models
    df = df[['Stepwise Regression (%)', 'Random Forest (%)']]
    fig, ax = plt.subplots()
    ax.hist(df, bins=15, histtype='bar', label=['Stepwise Regression', 'Random Forest'])
    ax.set_xlim([0, 100])
    ax.set_xticks(range(0, 110, 10))
    ax.set_ylim(top=ax.get_yticks().max() * 1.03)
    ax.set_title('Distribution of Churn Probabilities per Model')
    ax.set_xlabel('Probability to Cancel Membership (%)')
    ax.set_ylabel('Number of Club Members')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'hist_churn_prob.pdf'), bbox_inches='tight', format='pdf')

