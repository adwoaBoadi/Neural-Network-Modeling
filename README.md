# Neural-Network-Modeling

Build a neural network based classifier that can determine whether a customer will leave the bank or not in the next 6 months.

## Context of Problem

Businesses like banks which provide service have to worry about problem of 'Customer Churn' i.e. customers leaving and joining another service provider. It is important to understand which aspects of the service influence a customer's decision in this regard. Management can concentrate efforts on improvement of service, keeping in mind these priorities.

## Goal
The goal is to build a model that will help management concerntrate efforts on improvement of services.

## Data Dictionary
CustomerId: Unique ID which is assigned to each customer.

Surname: Last name of the customer

CreditScore: It defines the credit history of the customer.

Geography: A customer’s location

Gender: It defines the Gender of the customer

Age: Age of the customer

Tenure: Number of years for which the customer has been with the bank

NumOfProducts: refers to the number of products that a customer has purchased through the bank.

Balance: Account balance

HasCrCard: It is a categorical variable which decides whether the customer has credit card or not.

EstimatedSalary: Estimated salary

isActiveMember: Is is a categorical variable which decides whether the customer is active member of the bank or not ( Active member in the sense, using bank products regularly, making transactions etc )

Exited : whether or not the customer left the bank within six month. It can take two values
0=No ( Customer did not leave the bank )
1=Yes ( Customer left the bank )

## Files for Project
churn.csv: csv file with data

data_modeling.ipynb: python file for modeling

EDA.ipynb: python file for exploratory data analysis

utilities.py: python file for functiones needed  

## Actionable Insights and Business Recommendations

In analyzing individual variables, we observe that customer tenure across the bank is consistent with no extremes and displays an even distribution, indicating a stable customer base regarding service duration. The balance variable reveals that most customers maintain near zero, indicating minimal outliers and a slight preference towards lower balance accounts, with a distribution leans marginally to the left. Our examination of customer salaries reveals a clientele with a solid financial standing. The average wage among our bank customers is around 100,000, with the higher salary range peaking at approximately $200,000. This suggests a potential for offering higher-value services and products. The age distribution among our clients, which averages around 37 years, shows no significant outliers and a distribution that is slightly right-skewed. This means the majority of our customers are younger than the mean age, with a few older clients pushing the average higher. The credit score average stands around 650, which also demonstrates a slight deviation from a perfectly normal distribution. Our categorical analysis underscores the bank's success in attracting and retaining a diverse customer base. We find that 50% of our customers are located in France, 25.1% in Germany, and the remaining 24% in Spain, indicating a broad geographical reach. The gender distribution shows a slight male majority at 54.6%, suggesting a balanced customer base. A substantial 70.5% of our clients possess a credit card from the bank, and 51.5% regularly engage with our offerings, indicating a high level of customer involvement. The retention rate is a testament to our customer-centric approach, with 79.6% of our clients choosing to stay with us while 20.4% have departed. 

Further analysis exploring the relationship between variables (bivariate analysis) using a correlation matrix suggests a minimal direct correlation between most features and the likelihood of a customer leaving the bank. However, age is the most correlated variable to customer churn, with a correlation coefficient of 0.29, hinting that customer retention strategies must be adjusted with age demographics in mind. Interestingly, among those who have left, there is a trend of higher median balances compared to those who stayed, suggesting that financial factors play a role in the decision to leave. Despite this, the estimated salary and the number of products purchased are similar between those who stayed and those who left, indicating these factors may not be primary influencers in a customer's decision to stay with or leave the bank. This analysis provides critical insights for the bank to tailor its customer retention strategies, focusing on age-related preferences and potentially re-evaluating services for customers with higher balances to enhance loyalty and reduce churn rates. 

Final Model: Neural Network (Adam, No Regularization) Reasoning:
Best Recall score on the training set (0.86), indicating strong performance in learning from the training data. Best F1 score on the validation set (0.85), demonstrating good performance in generalizing to unseen data.
When considering validation scores only, Model 2 (that is the model with an Adam optimizer and no regularization) outperforms others, suggesting its effectiveness in real-world applications and new data scenarios. Recommend buidling other models utilizing other regularization techniques suchs as weight initialization, changing hyperparameters of optimizers, batch normalization, and increasing the size of hidden layers.
