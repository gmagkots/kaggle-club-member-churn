# kaggle-club-member-churn
 Club membership churn prediction with forward stepwise logistic regression and
 random forest estimators. The raw data stem from 
 [Kaggle](https://www.kaggle.com/datasets/sonannguyenngoc/club-data-set).

Stepwise logistic regression main results summary
-------------------------------------------------
Model total accuracy: 87.75%  
Class accuracies:  
| MEMBERSHIP_STATUS | Accuracy (%) |   
| :---------------- | :----------: |  
| CANCELLED         |    76.10     |  
| INFORCE           |    92.86     |   

Feature importance:  
1. MEMBERSHIP_TERM_YEARS
2. MEMBERSHIP_DURATION (CURRENT_DATE/END_DATE - START_DATE)
3. MEMBERSHIP_PACKAGE

Random forest main results summary
----------------------------------
Model total accuracy: 88.74%  
Class accuracies (%)  
| MEMBERSHIP_STATUS | Accuracy (%) |   
| :---------------- | :----------: |  
| CANCELLED         |    93.31     |  
| INFORCE           |    86.73     | 

Feature importance:  
1. MEMBERSHIP_DURATION
2. START_YEAR
3. MEMBERSHIP_PACKAGE
4. MEMBERSHIP_TERM_YEARS
5. OCCUPATION_CODE
6. MEMBERSHIP_AGE_AT_ISSUE

