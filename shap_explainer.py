import shap
import matplotlib.pyplot as plt
import os

# Bar plot of feature importance
def shap_bar_plot(best_model, X_sample, feature_names=None, save_path="Figs/shap_plot.png"):
    print("\nBar Plot for Feature Importance:")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    estimator = best_model.estimators_[0]
    
    # Create SHAP KernelExplainer
    explainer = shap.KernelExplainer(estimator.predict, X_sample)
    shap_values = explainer.shap_values(X_sample)
    
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar",show=False)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_predictions(best_model, X, y, num_outputs=4, save_path="Figs/Prediction_plot.png"):
    print("\nPredict vs. Reality:")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    y_pred = best_model.predict(X)
    
    _, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs,12))
    
    for i in range(num_outputs):
        # Predict vs. Reality
        ax = axes[0, i]
        r2 = r2_score(y.iloc[:, i], y_pred[:, i]) 
        ax.scatter(y.iloc[:, i], y_pred[:, i], alpha=0.7, color="blue")
        ax.plot([y.iloc[:, i].min(), y.iloc[:, i].max()],
                [y.iloc[:, i].min(), y.iloc[:, i].max()],
                color="red", linestyle="--")
        ax.set_title(f"Predict vs. Reality (Output {i+1})\n$R^2 = {r2:.3f}$")
        ax.set_xlabel("Reality")
        ax.set_ylabel("Predict")
        ax.grid(True)
        
        # Residual Plot
        residuals = y.iloc[:, i] - y_pred[:, i]
        ax_res = axes[1, i]
        ax_res.scatter(y.iloc[:, i], residuals, alpha=0.7, color="green")
        ax_res.axhline(0, color="red", linestyle="--")
        ax_res.set_title(f"Residual Plot (Output {i+1})")
        ax_res.set_xlabel("Reality")
        ax_res.set_ylabel("Residuals")
        ax_res.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


