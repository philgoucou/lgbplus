"""
LGB^A+: Hybrid Tree + Linear Boosting

A gradient boosting variant that alternates between:
1. Fitting a block of trees on residuals
2. Fitting a single linear term (greedy feature selection) on residuals

This captures both nonlinear patterns (via trees) and linear structure 
(via repeated linear updates), making it robust to unknown DGPs.

Author: Philippe Goulet Coulombe (UQAM)
License: MIT
"""

import numpy as np
from typing import Optional, List, Tuple
import warnings


class LGBPlusA:
    """
    LGB^A+ regressor.
    
    Alternates between gradient boosting trees and greedy linear updates.
    Effective when the true DGP has both linear and nonlinear components.
    
    Parameters
    ----------
    n_cycles : int, default=25
        Number of alternating cycles. Each cycle fits one block of trees
        followed by one linear update.
        
    trees_per_cycle : int, default=10
        Number of trees to fit per cycle. Total trees = n_cycles × trees_per_cycle.
        
    lr_tree : float, default=0.02
        Learning rate (shrinkage) for tree predictions.
        
    lr_linear : float, default=0.1
        Learning rate for linear updates. Higher values let linear
        component build up faster.
        
    num_leaves : int, default=15
        Maximum number of leaves per tree. Lower = more regularization.
        
    min_child_samples : int, default=20
        Minimum samples required in a leaf.
        
    random_state : int or None, default=None
        Random seed for reproducibility.
        
    verbose : bool, default=False
        If True, print progress during fitting.
    
    Attributes
    ----------
    init_ : float
        Initial prediction (mean of training target).
        
    trees_ : list
        Fitted tree boosters, one per cycle.
        
    linear_steps_ : list of tuples
        Each tuple is (feature_index, coefficient, intercept) for the
        linear update in that cycle.
        
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances from trees (sum of gains).
        
    linear_feature_counts_ : ndarray of shape (n_features,)
        How many times each feature was selected for linear updates.
    
    Examples
    --------
    >>> from lgb_plus_A import LGBPlusA
    >>> import numpy as np
    >>> X = np.random.randn(1000, 5)
    >>> y = 0.5 * X[:, 0] + 0.3 * np.tanh(X[:, 1]) + np.random.randn(1000) * 0.1
    >>> model = LGBPlusA(n_cycles=25, trees_per_cycle=10)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """
    
    def __init__(
        self,
        n_cycles: int = 25,
        trees_per_cycle: int = 10,
        lr_tree: float = 0.02,
        lr_linear: float = 0.1,
        num_leaves: int = 15,
        min_child_samples: int = 20,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_cycles = n_cycles
        self.trees_per_cycle = trees_per_cycle
        self.lr_tree = lr_tree
        self.lr_linear = lr_linear
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """
        Fit the alternating boost model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
            
        y : array-like of shape (n_samples,)
            Training target.
            
        feature_names : list of str, optional
            Names for features (used in verbose output).
            
        Returns
        -------
        self : LGBPlusA
            Fitted model.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        # Initialize
        self.init_ = float(y.mean())
        pred = np.full(n_samples, self.init_)
        
        self.trees_ = []
        self.linear_steps_ = []
        self.feature_importances_ = np.zeros(n_features)
        self.linear_feature_counts_ = np.zeros(n_features, dtype=int)
        
        # LightGBM parameters
        lgb_params = dict(
            objective="regression",
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            verbosity=-1,
            force_col_wise=True,
        )
        if self.random_state is not None:
            lgb_params["seed"] = self.random_state
        
        for cycle in range(self.n_cycles):
            # === Tree step ===
            resid = y - pred
            dtrain = lgb.Dataset(X, label=resid, free_raw_data=False)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(lgb_params, dtrain, self.trees_per_cycle)
            
            tree_pred = booster.predict(X)
            pred = pred + self.lr_tree * tree_pred
            self.trees_.append(booster)
            
            # Accumulate feature importance
            importance = booster.feature_importance(importance_type='gain')
            if len(importance) == n_features:
                self.feature_importances_ += importance
            
            # === Linear step ===
            resid = y - pred
            
            # Find feature most correlated with residual
            best_j, best_corr = 0, 0.0
            for j in range(n_features):
                corr = np.corrcoef(X[:, j], resid)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_j = j
            
            # Fit simple linear regression on that feature
            x_j = X[:, best_j]
            x_mean, x_var = x_j.mean(), x_j.var()
            if x_var > 1e-10:
                coef = np.cov(x_j, resid)[0, 1] / x_var
                intercept = resid.mean() - coef * x_mean
            else:
                coef, intercept = 0.0, resid.mean()
            
            linear_pred = coef * x_j + intercept
            pred = pred + self.lr_linear * linear_pred
            
            self.linear_steps_.append((best_j, coef, intercept))
            self.linear_feature_counts_[best_j] += 1
            
            if self.verbose and (cycle + 1) % 5 == 0:
                train_rmse = np.sqrt(np.mean((y - pred) ** 2))
                print(f"Cycle {cycle+1}/{self.n_cycles}: "
                      f"RMSE={train_rmse:.4f}, "
                      f"linear feature={feature_names[best_j]}")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.init_)
        
        for booster, (j, coef, intercept) in zip(self.trees_, self.linear_steps_):
            pred = pred + self.lr_tree * booster.predict(X)
            pred = pred + self.lr_linear * (coef * X[:, j] + intercept)
        
        return pred
    
    def get_total_trees(self) -> int:
        """Return total number of trees in the ensemble."""
        return self.n_cycles * self.trees_per_cycle
    
    def summary(self) -> str:
        """Return a summary of the fitted model."""
        if not hasattr(self, 'trees_'):
            return "Model not fitted yet."
        
        lines = [
            "LGBPlusA Summary",
            "=" * 40,
            f"Total cycles: {self.n_cycles}",
            f"Trees per cycle: {self.trees_per_cycle}",
            f"Total trees: {self.get_total_trees()}",
            f"Learning rates: tree={self.lr_tree}, linear={self.lr_linear}",
            "",
            "Linear feature selection frequency:",
        ]
        
        # Sort by count
        sorted_idx = np.argsort(self.linear_feature_counts_)[::-1]
        for idx in sorted_idx:
            if self.linear_feature_counts_[idx] > 0:
                lines.append(f"  Feature {idx}: {self.linear_feature_counts_[idx]} times")
        
        return "\n".join(lines)


# Convenience function for quick fitting
def lgb_plus_A(
    X_train, y_train, X_test=None,
    n_cycles: int = 25,
    trees_per_cycle: int = 10,
    lr_tree: float = 0.02,
    lr_linear: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[LGBPlusA, np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to fit LGBPlusA and get predictions.
    
    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    X_test : array-like, optional
        Test features. If provided, returns test predictions.
    n_cycles : int
        Number of alternating cycles.
    trees_per_cycle : int
        Trees per cycle.
    lr_tree : float
        Tree learning rate.
    lr_linear : float
        Linear learning rate.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    model : LGBPlusA
        Fitted model.
    train_pred : ndarray
        Training predictions.
    test_pred : ndarray or None
        Test predictions (if X_test provided).
    """
    model = LGBPlusA(
        n_cycles=n_cycles,
        trees_per_cycle=trees_per_cycle,
        lr_tree=lr_tree,
        lr_linear=lr_linear,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None
    
    return model, train_pred, test_pred


if __name__ == "__main__":
    # Quick demo
    print("LGBPlusA Demo")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Generate hybrid DGP data
    n = 2000
    X = np.random.randn(n, 4)
    
    # y = linear + nonlinear + noise
    y = (0.5 * X[:, 0] +                    # linear
         0.3 * np.tanh(2 * X[:, 1]) +       # nonlinear
         0.2 * X[:, 0] * (X[:, 1] > 0) +    # interaction
         np.random.randn(n) * 0.3)          # noise
    
    # Train/test split
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]
    
    # Fit model
    model = LGBPlusA(n_cycles=25, trees_per_cycle=10, verbose=True)
    model.fit(X_train, y_train, feature_names=['x0', 'x1', 'x2', 'x3'])
    
    # Evaluate
    from sklearn.metrics import root_mean_squared_error
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    print(f"\nTrain RMSE: {root_mean_squared_error(y_train, pred_train):.4f}")
    print(f"Test RMSE:  {root_mean_squared_error(y_test, pred_test):.4f}")
    print(f"\n{model.summary()}")
