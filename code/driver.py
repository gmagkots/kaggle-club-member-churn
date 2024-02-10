import time
from datetime import timedelta
from data_preprocess import raw_data_etl
from feature_analysis import feature_analysis


def main():
    """
    Controls the workflow for the analysis.

    :return: None
    """
    # set script start time
    start_time = time.time()

    # run functions
    # raw_data_etl()
    feature_analysis()

    # stop the clock
    elapsed = time.time() - start_time
    print('Execution time: {}'.format(str(timedelta(seconds=elapsed))))


if __name__ == '__main__':
    main()
