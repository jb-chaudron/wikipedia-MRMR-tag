from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from matplotlib.pyplot import axes
import pandas as pd 

import matplotlib.pyplot as plt 

def get_figure_ts(node_numb : int, 
                  columns : list, 
                  X_in : pd.DataFrame, 
                  y_in : pd.DataFrame, 
                  model : Pipeline, 
                  ax : axes, 
                  in_out : str = "in", 
                  legend : bool =False, 
                  predict : bool =False) -> axes:
    
    X_in = X_in.sort_index(level=0)
    y_in = y_in.sort_index(level=0)

    i = node_numb

    if len(columns) == 0:
        columns = X_in.columns
        color_columns = "none"
    else:
        color_columns = "not none"
        columns = [c for c in columns if c in X_in.columns]

    if predict:
        intercept = model.intercept_
        ceofs = model.coef_

        X_aug = intercept+X_in*ceofs
        X_to_pred = X_in.loc[((X_in.index.get_level_values(1)==i) & (X_in.index.get_level_values(2) == in_out)),:]
        ax.plot(model.predict(X_to_pred), linestyle="--", color="Red")
        #ax.plot(search_elastic.best_estimator_["model"].predict(X_to_pred), linestyle="--", color="Red")
    else:
        ax.plot(
            X_in.loc[((X_in.index.get_level_values(1)==i) & (X_in.index.get_level_values(2) == in_out)),:].sum(1).to_numpy(), linestyle="--", color="Red")
        X_aug = X_in
    
    if color_columns == "none":
        X_aug.loc[((X_aug.index.get_level_values(1)==i) & (X_aug.index.get_level_values(2) == in_out)),columns].plot(legend=legend,ax=ax, linestyle="dotted",color="Gray")
    else:
        X_aug.loc[((X_aug.index.get_level_values(1)==i) & (X_aug.index.get_level_values(2) == in_out)),columns].plot(legend=legend,ax=ax, linestyle="-.", colormap="tab20")
    plt.xticks(rotation=45, ha="right")
    
    ax.plot(y_in.loc[((y_in.index.get_level_values(1) == i) & (y_in.index.get_level_values(2) == in_out))].to_numpy(), linestyle="-", color="Green")
