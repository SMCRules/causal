
from catboost import CatBoostClassifier, CatBoostRegressor
from econml.dml import CausalForestDML
from graphviz import Digraph
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, roc_auc_score
from IPython.display import display
import numpy as np
import pandas as pd

import time
  
# Create datasets and train models

models = {

    "CausalForestDML": CausalForestDML(
        model_t=CatBoostClassifier(verbose=False),
        model_y=CatBoostRegressor(verbose=False),
        discrete_treatment=True),

    "CatBoost": CatBoostRegressor(verbose=False),

}
     

## %%time
start_time = time.time()

n_non_decision_variables = 20
n_clusters = 20
decision_variable = "is_called"
target_variable = "y"

results = pd.DataFrame()
i = 0

for dataset_size, n_obs in zip(["small", "medium"], [10_000, 100_000]):
    for decision_predictability, std_epsilon in zip(["small", "medium", "large"], [50, 10, 3]):
        for decision_effect_size, avg_effect in zip(["small", "medium", "large"], [.5, 1, 2]):

            print(f"dataset_size: {dataset_size}, decision_predictability: {decision_predictability}, decision_effect: {decision_effect_size}")

            ix_trn = range(int(n_obs / 2))
            ix_tst = range(int(n_obs / 2), n_obs)
            non_decision_variables = [str(f) for f in range(n_non_decision_variables)]

            df = pd.DataFrame(np.random.normal(size=(n_obs, n_non_decision_variables)), columns=non_decision_variables)

            df["kmeans_label"] = KMeans(n_clusters=n_clusters, random_state=123).fit_predict(df[non_decision_variables])

            df[decision_variable] = (
                (df[non_decision_variables] * np.repeat([-1,1], n_non_decision_variables)[:n_non_decision_variables]).sum(axis=1)
                + np.random.normal(0, std_epsilon, size=(len(df),)) > 0).astype(int)

            df["decision_effect"] = df["kmeans_label"].replace({
                k: e for k, e in zip(range(n_clusters), np.linspace(2-avg_effect*2, 2+avg_effect*2, n_clusters))})

            df[target_variable] = (
                df[non_decision_variables].sum(axis=1)
                + df["decision_effect"] * df[decision_variable]
                + np.random.normal(size=(len(df)))
            )

            df["y_if_treated"] = (
                df[non_decision_variables].sum(axis=1)
                + df["decision_effect"]
                + np.random.normal(size=(len(df)))
            )

            roc_is_treated = roc_auc_score(
                df.loc[ix_tst, decision_variable],
                CatBoostClassifier(verbose=False).fit(
                    X=df.loc[ix_trn, non_decision_variables],
                    y=df.loc[ix_trn, decision_variable]).predict_proba(df.loc[ix_tst, non_decision_variables])[:, 1])

            for model_name, model in models.items():

                if "DML" in model_name:
                    model = model.fit(
                        X=df.loc[ix_trn, non_decision_variables],
                        T=df.loc[ix_trn, decision_variable],
                        Y=df.loc[ix_trn, target_variable])

                    decision_effect = model.effect(df.loc[ix_tst, non_decision_variables])

                    X_effect = np.mean([
                        model.models_y[0][i].predict(df.loc[ix_tst, non_decision_variables]) for i in range(2)], axis=0)

                    y_pred = X_effect + decision_effect * df.loc[ix_tst, decision_variable]

                    y_pred_if_treated = X_effect + decision_effect

                else:
                    model = model.fit(
                        X=df.loc[ix_trn, non_decision_variables + [decision_variable]],
                        y=df.loc[ix_trn, target_variable])

                    decision_effect = (
                        model.predict(df.assign(**{decision_variable: 1}).loc[ix_tst, non_decision_variables + [decision_variable]])
                        - model.predict(df.assign(**{decision_variable: 0}).loc[ix_tst, non_decision_variables + [decision_variable]]))

                    y_pred = model.predict(df.loc[ix_tst, non_decision_variables + [decision_variable]])

                    y_pred_if_treated = model.predict(df.assign(**{decision_variable: 1}).loc[ix_tst, non_decision_variables + [decision_variable]])

                results.loc[i, "dataset size"] = dataset_size
                results.loc[i, "decision predictability"] = decision_predictability
                results.loc[i, "decision effect"] = decision_effect_size
                results.loc[i, "std epsilon"] = std_epsilon
                results.loc[i, "avg effect"] = avg_effect
                results.loc[i, "roc decision"] = roc_is_treated
                results.loc[i, "model"] = model_name
                results.loc[i, "std effect"] = np.std(decision_effect)
                results.loc[i, "r2 target"] = r2_score(df.loc[ix_tst, target_variable], y_pred)
                results.loc[i, "r2 effect"] = r2_score(df.loc[ix_tst, "decision_effect"], decision_effect)

                i += 1

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

# Display results
print("--- R2 of decision effect ---\n\n")
for dataset_size in ["small", "medium"]:
    results_agg = (
        results
            [results["dataset size"] == dataset_size]
            .sort_values("r2 effect", ascending=False)
            .groupby(["decision predictability", "decision effect"])
            .apply(lambda d: f"{d.iloc[0,:]['model']} ({d.iloc[0,:]['r2 effect']:.0%})", include_groups=False)
            .rename("best model")
            .reset_index()
    )

    print(f"dataset size: {dataset_size}\n")

    display(
        pd.crosstab(
            results_agg["decision predictability"],
            results_agg["decision effect"],
            values=results_agg["best model"],
            aggfunc="first"
        ).sort_index(ascending=False).sort_index(ascending=False, axis=1)
    )
    print("\n\n")


# Theory
dot = Digraph()
dot.attr(rankdir='LR', ordering='out')
dot.node("X", "non-decision variables")
dot.node("T", "decision variables")
dot.node("y", "target variable")
dot.edge("X", "y")
dot.edge("X", "T")
dot.edge("T", "y")
dot.render("causal_graph_01", format="png")
#     
dot = Digraph()
dot.attr(rankdir='LR', ordering='out')
dot.node("X", "non-decision variables\n(customer tenure,\nlast month spend)")
dot.node("T", "decision variables\n(call)")
dot.node("y", "target variable\n(next month spend)")
dot.edge("X", "y")
dot.edge("X", "T")
dot.edge("T", "y")
dot.render("causal_graph_02", format="png")
#
dot = Digraph()
dot.attr(rankdir='LR', ordering='out')
dot.node("X", "non-decision variables")
dot.node("T", "decision variables")
dot.node("y", "target variable")
dot.edge("X", "y", label="model_y")
dot.edge("X", "T", label="model_t")
dot.edge("T", "y", label="causal_forest")
dot.render("causal_graph_03", format="png")
#
dot = Digraph()
dot.attr(rankdir='LR', ordering='out')
dot.node("X", "non-decision variables")
dot.node("T", "decision variables")
dot.node("y", "target variable")
dot.edge("X", "y")
dot.edge("X", "T", label="decision\npredictability")
dot.edge("T", "y", label="decision\neffect")
dot.render("causal_graph_04", format="png")     
#
dot = Digraph()
dot.attr(rankdir="LR", ordering="out")
dot.node("X", "non-decision variables")
dot.node("T", "decision variables")
dot.node("y", "target variable")
dot.edge("X", "y")
dot.edge("X", "T", label="decision\npredictability", penwidth=".25", color="red")
dot.edge("T", "y", label="decision\neffect", penwidth="4", color="red")

dot.node("X2", "non-decision variables")
dot.node("T2", "decision variables")
dot.node("y2", "target variable")
dot.edge("X2", "y2", penwidth=".50")
dot.edge("X2", "T2", label="decision\npredictability", penwidth=".25", color="red")
dot.edge("T2", "y2", label="decision\neffect", penwidth="4", color="red")

dot.render("causal_graph_05", format="png")
     

     

     

