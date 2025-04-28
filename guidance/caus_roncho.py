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
# dot.draw('dag_cpds_output.pdf', prog='dot')
# import os
# os.system('xdg-open dag_cpds_output.pdf')

# data simulation, N observations based on the provided CPDs and the DAG
N = 10_000
df = model.simulate(n_samples=N, show_progress=False, seed=42)
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

full_model_OR = get_coef_from_statmodels_model(model_full)
full_model_OR = np.exp(full_model_OR).rename(columns={'coef': 'full_OR'})
print(full_model_OR.round(2))

###  Marginal Odds-Ratios

marginal_OR = {}
marginal_OR['residential_location'] = get_OR_for_treatment(
    df_clean[df_clean['residential_location'] == 'Urban'],
    df_clean[df_clean['residential_location'] == 'Rural']
)

marginal_OR['smoking'] = get_OR_for_treatment(
    df_clean[df_clean['smoking'] == 'Yes'],
    df_clean[df_clean['smoking'] == 'No']
)

marginal_OR['lung_function_sev_imp'] = get_OR_for_treatment(
    df_clean[df_clean['lung_function'] == 'severely_impaired'],
    df_clean[df_clean['lung_function'] == 'normal']
)

marginal_OR['lung_function_imp'] = get_OR_for_treatment(
    df_clean[df_clean['lung_function'] == 'impaired'],
    df_clean[df_clean['lung_function'] == 'normal']
)

marginal_OR = pd.Series(marginal_OR, name='marginal_OR').to_frame()
print(marginal_OR.round(2))

### Adjusted Models
adjusted_OR = {}
# residential Location
model_adj = smf.glm(
    formula='lung_cancer_death ~ residential_location',
    data=df_clean,
    family=sm.families.Binomial()
).fit()

adjusted_OR['residential_location'] = get_coef_from_statmodels_model(
    model_adj, 'residential_location[T.Urban]'
    )
print(model_adj.summary())


# smoking
model_adj = smf.glm(
    formula='lung_cancer_death ~ residential_location + smoking',
    data=df_clean,
    family=sm.families.Binomial()
).fit()
adjusted_OR['smoking'] = get_coef_from_statmodels_model(model_adj, 'smoking[T.Yes]')

print(model_adj.summary())


# lung Function
model_adj = smf.glm(
    formula='lung_cancer_death ~ lung_function + smoking',
    data=df_clean,
    family=sm.families.Binomial()
).fit()

adjusted_OR['lung_function_imp'] = get_coef_from_statmodels_model(
    model_adj, 'lung_function[T.impaired]'
    )
adjusted_OR['lung_function_sev_imp'] = get_coef_from_statmodels_model(
    model_adj, 'lung_function[T.severely_impaired]'
    )

print(model_adj.summary())

# summarize adjusted odds ratios
adjusted_OR = np.exp(pd.Series(adjusted_OR)).to_frame().rename(columns={0: 'adjusted_OR'}).round(2)
print(adjusted_OR)

### True Causal Effects via Simulation
# Residential Location, fixing seed and cleaning simulated data fixes ORs
samples_urban = model.simulate(
    n_samples=N, 
    do={'residential_location': 'Urban'}, 
    show_progress=False,
    seed=100
    )
samples_urban_clean = clean_simulated_data(samples_urban)

samples_rural = model.simulate(
    n_samples=N, 
    do={'residential_location': 'Rural'}, 
    show_progress=False,
    seed=100
    )
samples_rural_clean = clean_simulated_data(samples_rural)

residential_location_OR = get_OR_for_treatment(
    samples_urban_clean, 
    samples_rural_clean
    )   
print(f'True odds ratio (Urban vs. Rural): {residential_location_OR:.4f}\nTrue log odds ratio (Urban vs. Rural): {np.log(residential_location_OR):.4f}')

# Smoking
samples_smoke_no = model.simulate(
    n_samples=N, 
    do={'smoking': 'No'}, 
    show_progress=False,
    seed=100
    )
samples_smoke_no_clean = clean_simulated_data(samples_smoke_no)

samples_smoke_yes = model.simulate(
    n_samples=N, 
    do={'smoking': 'Yes'}, 
    show_progress=False,
    seed=100
    )

samples_smoke_yes_clean = clean_simulated_data(samples_smoke_yes)

smoking_OR = get_OR_for_treatment(
    samples_smoke_yes_clean, 
    samples_smoke_no_clean
    )  
print(f'True odds ratio (Smoking vs. Non-Smoking): {smoking_OR:.4f}\nTrue log odds ratio (Smoking vs. Non-Smoking): {np.log(smoking_OR):.4f}')

# Lung Function
samples_lung_fun_ref = model.simulate(
    n_samples=N, 
    do={'lung_function': 'impaired'}, 
    show_progress=False,
    seed=100
    )
samples_lung_fun_ref_clean = clean_simulated_data(samples_lung_fun_ref)

samples_lung_fun_imp = model.simulate(
    n_samples=N, 
    do={'lung_function': 'normal'}, 
    show_progress=False,
    seed=100
    )

samples_lung_fun_imp_clean = clean_simulated_data(samples_lung_fun_imp)

samples_lung_fun_sev_imp = model.simulate(
    n_samples=N, 
    do={'lung_function': 'severely_impaired'}, 
    show_progress=False,
    seed=100
    )

samples_lung_fun_sev_imp_clean = clean_simulated_data(samples_lung_fun_sev_imp)

lung_fun_imp_OR = get_OR_for_treatment(
    samples_lung_fun_imp_clean, 
    samples_lung_fun_ref_clean
    )
lung_fun_sev_imp_OR = get_OR_for_treatment(
    samples_lung_fun_sev_imp_clean, 
    samples_lung_fun_ref_clean
    )

print(f'True odds ratio (Impaired vs. Normal): {lung_fun_imp_OR:.4f}\nTrue log odds ratio (Impaired vs. Normal): {np.log(lung_fun_imp_OR):.4f}')
print(f'\nTrue odds ratio (Severely Impaired vs. Normal): {lung_fun_sev_imp_OR:.4f}\nTrue log odds ratio (Severely Impaired vs. Normal): {np.log(lung_fun_sev_imp_OR):.4f}')

# Summarize true odds ratios
true_OR = pd.Series({
    'residential_location': residential_location_OR,
    'smoking': smoking_OR,
    'lung_function_imp': lung_fun_imp_OR,
    'lung_function_sev_imp': lung_fun_sev_imp_OR
}, name='true_OR').to_frame()
print(true_OR.round(2))


### SHAP values
X = pd.get_dummies(df_clean.drop("lung_cancer_death", axis=1), drop_first=True)
y = df["lung_cancer_death"]

xgb_model = xgb.XGBClassifier(random_state=1)
xgb_model.fit(X, y)

roc_auc = roc_auc_score(y, xgb_model.predict_proba(X)[:, 1])
print(f"ROC AUC: {roc_auc:.2f}")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_summary_df = pd.DataFrame({
    "Feature": X.columns,
    "mean_abs_SHAP": mean_abs_shap
}).sort_values("mean_abs_SHAP", ascending=False)

print(shap_summary_df.round(3))

shap.summary_plot(shap_values, X, plot_type="bar", show=False, rng=1)
ax = plt.gca()
ax.set_title("Mean Absolute SHAP Values by Feature", fontsize=10)
ax.set_xlabel('Mean Absolute SHAP Value', fontsize=10)
ax.tick_params(axis='y', labelsize=11)
plt.tight_layout()  
plt.show()  


### Results
full_model_OR.drop(index=['Intercept'], inplace=True)
full_model_OR.rename(
    index={
        'smoking[T.Yes]': 'smoking', 'lung_function[T.impaired]': 'lung_function_imp',
        'lung_function[T.severely_impaired]': 'lung_function_sev_imp', 
        'residential_location[T.Urban]': 'residential_location'
        }, 
    inplace=True
)

shap_summary_df = shap_summary_df.set_index('Feature').rename(
    index={
        'smoking_Yes': 'smoking', 'lung_function_severely_impaired': 'lung_function_sev_imp',
        'lung_function_impaired': 'lung_function_imp', 'residential_location_Urban': 'residential_location'
        }
)

dfs = [full_model_OR, true_OR, marginal_OR, shap_summary_df, adjusted_OR]
results = reduce(lambda left, right: left.merge(right, left_index=True, right_index=True), dfs)

print(results.round(2))

plot_cols = ['full_OR', 'true_OR', 'marginal_OR']
ax = results.loc[:, plot_cols].plot(kind='bar', figsize=(8, 5))
ax.set_ylabel("Odds Ratio")
ax.set_title("Comparison of Different Odds Ratios Estimates")

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

new_labels = ["Full Model", "True", "Marginal"]
plt.legend(labels=new_labels)
plt.tight_layout()
plt.show() 

# Normalize "true_OR" so it sums to 1
results["true_OR_norm"] = results["true_OR"] / results["true_OR"].sum()

# Normalize "Mean Absolute SHAP Value" so it sums to 1
results["mean_abs_shap_norm"] = (
    results["mean_abs_SHAP"] / results["mean_abs_SHAP"].sum()
)

ax = results[["true_OR_norm", "mean_abs_shap_norm"]].plot(kind="bar", figsize=(8, 5))
ax.set_ylabel("Normalized Value")
ax.set_title("Normalized True Odds Ratio vs. Normalized Mean Absolute SHAP Value")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
new_labels = ["True effect (norm)", "Mean abs. SHAP (norm)"]
plt.legend(labels=new_labels)
plt.tight_layout()
plt.show()


# Full Model Assumptions DAG

model = DiscreteBayesianNetwork([
    ('residential_location', 'lung_cancer_death'),
    ('smoking', 'lung_cancer_death'),
    ('lung_function', 'lung_cancer_death')
])

dot = model.to_graphviz()
dot.graph_attr.update(ratio=0.4)
graphviz.Source(dot.to_string())
# Write to file and layout
# dot.draw('dag_full_model_output.pdf', prog='dot')







