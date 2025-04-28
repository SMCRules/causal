# Import necessary packages
# pip install gurobi-machinelearning
import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gurobi_ml import add_predictor_constr
from causaldata import thornton_hiv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""
The dataset comes from Thornton (2008). Individuals in rural Malawi were offered 
random monetary incentives to learn their HIV test results. 
The question is: Given a limited incentive budget, which people should we pay 
(and how much) to maximize the number of individuals who return for their HIV results?

By formulating a decision as an optimization problem, we can directly compute 
the best treatment plan under constraints. The Gurobis Machine Learning integration 
embeds a causal prediction model (a logistic regression) into an optimization model. 
This approach lets us find the optimal set of incentives to offer to maximize 
the number of positive outcomes, given budget limits.
"""
# data in causaldata package.
data = thornton_hiv.load_pandas().data
# Limit the sample size to Gurobi's free license
data = data.sample(n=1000, ignore_index=True)
# print(data.head())

# Replace NaN with appropriate values
data = data.fillna({
    'got': 0.0,
    'tinc': 0.0,
    'any': 0.0,
    'age': data['age'].mean(),
})

data.head()
# print(data.head())
# Describe the overview of the data
# print(data.describe())

# Individuals offered "any" incentives have higher "got" = got HIV results
data.groupby('any')['got'].plot(
    kind='hist',
    bins=2,
    alpha=0.6,
    density=True,
    legend=True
)

plt.xlabel('got')      
plt.ylabel('Frequency')  
plt.title('Histogram of HIV results grouped by any incentives')  
plt.legend()  
plt.tight_layout()
plt.show() 

# Systematic difference in "distvct" = distance in kms between control 
# and any incentives treated?
data.groupby('any')['distvct'].plot(
    kind='hist',
    alpha=0.6,
    density=True,
    legend=True
)
plt.xlabel('distvct')      
plt.ylabel('Frequency')  
plt.title('Histogram of distvct grouped by any incentive treatment')  
plt.legend()  
plt.tight_layout()
plt.show() 

# There are some biases in "age" between control and treated in the dataset,
# which means that the samples are not fully randomized and we must be careful interpreting the result
data.groupby('any')['age'].plot(
    kind='hist', 
    density=True, 
    alpha=0.6, 
    legend=True
    )
plt.xlabel('age')      
plt.ylabel('Frequency')  
plt.title('Histogram of age grouped by any incentive treatment')  
plt.legend()  
plt.tight_layout()
plt.show()

