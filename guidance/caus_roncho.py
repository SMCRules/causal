import graphviz
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from functools import reduce
from io import StringIO
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import roc_auc_score
from pandas.api.types import CategoricalDtype

np.random.seed(100)

# Only for the sake of cleaner output in this example
logging.getLogger("pgmpy").setLevel(logging.ERROR) 

### Functions
def get_OR_for_treatment(df_a, df_b, outcome_col='lung_cancer_death'):
    # Calculates the odds ratio given an outcome column, for the case where all of the
    # data in df_a is of the treated group and all of the data in df_b is the untreated group.
    a = df_a[outcome_col].sum()
    b = len(df_a) - a

    c = df_b[outcome_col].sum()
    d = len(df_b) - c

    OR = (a / b) / (c / d)

    return OR


def get_coef_from_statmodels_model(model, coef_name=None):
    # A convenience function to get the coefficients from a fitted statsmodels model.
    summary_coefs = pd.read_html(StringIO(model.summary().tables[1].as_html()), header=0, index_col=0)[0]['coef']
    return summary_coefs.to_frame() if coef_name is None else summary_coefs[coef_name]

def clean_simulated_data(df):
    """
    Cleans a simulated dataframe by converting Categorical columns:
    - Converts binary 0/1 categories to int
    - Converts multi-class categories to string
    Returns a copy of the cleaned dataframe.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        if isinstance(df_clean[col].dtype, CategoricalDtype):
            categories = set(df_clean[col].cat.categories)
            if categories <= {0, 1}:  # Only 0/1 values
                df_clean[col] = df_clean[col].astype(int)
            else:
                df_clean[col] = df_clean[col].astype(str)
    
    return df_clean

### Conditional Probability Distributions (CPDs)

# Define the conditional probability distribution for residential location
cpd_residential_location = TabularCPD(
    variable='residential_location',
    variable_card=2,
    values=[[0.8], [0.2]],
    state_names={'residential_location': ['Urban', 'Rural']}
).normalize(inplace=False)

# Define the conditional probability distribution for smoking, 
# depending on residential location
cpd_smoking = TabularCPD(
    variable='smoking',
    variable_card=2,
    values=[
        [0.1, 0.9],
        [0.9, 0.1]
    ],
    evidence=['residential_location'],
    evidence_card=[2],
    state_names={
        'smoking': ['No', 'Yes'],
        'residential_location': ['Urban', 'Rural']
    }
).normalize(inplace=False)

# Define the conditional probability distribution for lung function,
# depending on smoking and residential location
cpd_lung_function = TabularCPD(
    variable='lung_function',
    variable_card=3,
    values=[
        [0.3, 0.3, 0.5, 0.4],
        [0.5, 0.6, 0.1, 0.1],
        [0.2, 0.2, 0.6, 0.4]
    ],
    evidence=['smoking', 'residential_location'],
    evidence_card=[2, 2],
    state_names={
        'lung_function': ['impaired', 'normal', 'severely_impaired'],
        'smoking': ['No', 'Yes'],
        'residential_location': ['Urban', 'Rural']
    }
).normalize(inplace=False)

# Define the conditional probability distribution for lung cancer death,
# depending on smoking and lung function
cpd_lung_cancer_death = TabularCPD(
    variable='lung_cancer_death',
    variable_card=2,
    values=[
        [0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5, 0.7, 0.7, 0.7]
    ],
    evidence=['smoking', 'lung_function'],
    evidence_card=[2, 3],
    state_names={
        'smoking': ['No', 'Yes'],
        'lung_function': ['impaired', 'normal', 'severely_impaired'],
        'lung_cancer_death': [0, 1]
    }
).normalize(inplace=False)

# Create a Bayesian Network which will be the DAG we will work with
model = DiscreteBayesianNetwork([
    ('residential_location', 'smoking'),
    ('residential_location', 'lung_function'),
    ('smoking', 'lung_function'),
    ('smoking', 'lung_cancer_death'),
    ('lung_function', 'lung_cancer_death')
    ])

# Add the CPDs to the model.
# The CPDs are the conditional probability distribution tables previously defined, that 
# determine the probabilities of each node
model.add_cpds(
    cpd_residential_location, 
    cpd_smoking, 
    cpd_lung_function, 
    cpd_lung_cancer_death
    )

# Plot our DAG to make sure it looks as intended
dot = model.to_graphviz()
dot.graph_attr.update(ratio=0.5)
graphviz.Source(dot.to_string())
# Write to file and layout
# dot.draw('dag_output.pdf', prog='dot')
# import os
# os.system('xdg-open dag_output.pdf')

# data simulation, N observations based on the provided CPDs and the DAG
N = 10_000
df = model.simulate(n_samples=N, show_progress=False)
# print(df.sample(10))

### Clean df after simulation for modeling
df_clean = clean_simulated_data(df)

# set an explicit reference category (important for modeling!)
df_clean['lung_function'] = pd.Categorical(
    df_clean['lung_function'], categories=['normal', 'impaired', 'severely_impaired']
)

### CPD Inspection
print(df_clean['residential_location'].value_counts(normalize=True).to_frame().round(2))

print(df_clean.groupby(
    'residential_location'
)['smoking'].value_counts(normalize=True).to_frame().round(2).sort_index())

print(df_clean.groupby(
    ['residential_location', 'smoking']
)['lung_function'].value_counts(normalize=True).to_frame().round(2).sort_index())

print(df_clean.groupby(
    'lung_function', observed=False
)['lung_cancer_death'].value_counts(normalize=True).to_frame().round(2).sort_index())

# tiny diagnostic to run
print(df_clean['lung_cancer_death'].dtype)
print(df_clean['lung_cancer_death'].unique())

### Fit a Model With All Covariates
model_full = smf.glm(
    formula='lung_cancer_death ~ smoking + lung_function + residential_location',
    data=df_clean,
    family=sm.families.Binomial()
).fit()

print(model_full.summary())

y_pred_probs = model_full.predict(df_clean)
y_true = df_clean['lung_cancer_death']
roc_auc = roc_auc_score(y_true, y_pred_probs)
print(f"\nROC AUC: {roc_auc:.2f}")

# full_model_OR = get_coef_from_statmodels_model(model_full)
# full_model_OR = np.exp(full_model_OR).rename(columns={'coef': 'full_OR'})
# print(full_model_OR.round(2))

"""
#  Marginal Odds-Ratios

marginal_OR = {}
marginal_OR['residential_location'] = get_OR_for_treatment(
    df[df['residential_location'] == 'Urban'],
    df[df['residential_location'] == 'Rural']
)

marginal_OR['smoking'] = get_OR_for_treatment(
    df[df['smoking'] == 'Yes'],
    df[df['smoking'] == 'No']
)

marginal_OR['lung_function_sev_imp'] = get_OR_for_treatment(
    df[df['lung_function'] == 'severely_impaired'],
    df[df['lung_function'] == 'normal']
)

marginal_OR['lung_function_imp'] = get_OR_for_treatment(
    df[df['lung_function'] == 'impaired'],
    df[df['lung_function'] == 'normal']
)

marginal_OR = pd.Series(marginal_OR, name='marginal_OR').to_frame()
print(marginal_OR.round(2))
"""




