"""
LGB+ (Competition): Iterative Tree vs Linear Boosting with Ensemble

An iterative boosting model where each step chooses between:
1. A small tree update (LightGBM with tiny trees)
2. A single-feature linear update (pick predictor most correlated with residual)

At each step, computes both candidates and applies whichever reduces loss more.
Runs an ensemble of B independent runs with row subsampling (70%) per run.

Selection methods for choosing between tree/linear:
- "oob": Out-of-bag evaluation (recommended for macro). Uses samples NOT in
         the current subsample S_t. Different "judges" each step.
- "validation": Fixed held-out validation set (original implementation).
- "training": Evaluate on training subsample S_t (not recommended, tree has
              structural advantage).

Algorithm Spec:
- B = 10 ensemble members
- T = configurable boosting steps per run
- eta = shared learning rate for both updates
- subsample = 0.7 row sampling per step
- Tree: num_leaves ~5-10, min_data_in_leaf ~20, one tree per step
- Linear: OLS on single best-correlated feature
- Decision: choose update with lower loss on selection set

Author: Philippe Goulet Coulombe (UQAM)
License: MIT
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Literal
import warnings


class LGBPlus:
    """
    LGB+ (Competition) regressor.

    At each boosting step, competes a small tree update against a single-feature
    linear update, choosing whichever reduces loss more. Runs B independent
    ensemble members with row subsampling.

    Parameters
    ----------
    n_ensemble : int, default=10
        Number of ensemble members (B). Each member is an independent boosting run.

    n_steps : int, default=200
        Number of boosting steps per ensemble member (T).

    learning_rate : float, default=0.05
        Learning rate (eta) applied to both tree and linear updates.

    subsample : float, default=0.7
        Row subsampling rate for each step. Rows are sampled without replacement.

    num_leaves : int, default=5
        Maximum number of leaves per tree. Keep small (5-10) for regularization.

    min_data_in_leaf : int, default=20
        Minimum samples required in a leaf.

    lambda_l2 : float, default=0.1
        L2 regularization for trees.

    selection_method : str, default='oob'
        Method for choosing between tree/linear updates:
        - 'oob': Out-of-bag evaluation (recommended). Uses samples NOT in
                 the current subsample. Different "judges" each step.
        - 'validation': Fixed held-out validation set.
        - 'training': Evaluate on training subsample (not recommended).

    val_fraction : float, default=0.2
        Fraction of training data held out for step-selection decisions.
        Only used if selection_method='validation'. If 0, uses training subset.

    early_stop_patience : int or None, default=50
        Stop if validation loss hasn't improved for this many steps.
        Set to None to disable early stopping.

    aggregation : str, default='mean'
        How to aggregate ensemble predictions: 'mean' or 'median'.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        If True, print progress during fitting.

    Attributes
    ----------
    ensemble_ : list
        List of fitted ensemble members. Each member is a dict containing:
        - 'steps': list of step dicts (type, tree/linear params)
        - 'init': initial prediction value

    step_type_counts_ : ndarray of shape (n_ensemble, 2)
        Counts of [tree_steps, linear_steps] per ensemble member.

    linear_feature_counts_ : ndarray of shape (n_features,)
        Aggregated counts of how often each feature was selected for linear steps.

    feature_importances_ : ndarray of shape (n_features,)
        Feature importances from trees (sum of gains across ensemble).

    val_losses_ : list
        Validation loss history for each ensemble member.

    Examples
    --------
    >>> from lgb_plus import LGBPlus
    >>> import numpy as np
    >>> X = np.random.randn(1000, 5)
    >>> y = 0.5 * X[:, 0] + 0.3 * np.tanh(X[:, 1]) + np.random.randn(1000) * 0.1
    >>> model = LGBPlus(n_steps=100, n_ensemble=10)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        n_ensemble: int = 10,
        n_steps: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.7,
        num_leaves: int = 5,
        min_data_in_leaf: int = 20,
        lambda_l2: float = 0.1,
        selection_method: Literal['oob', 'validation', 'training'] = 'oob',
        val_fraction: float = 0.2,
        early_stop_patience: Optional[int] = 50,
        aggregation: Literal['mean', 'median'] = 'mean',
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_ensemble = n_ensemble
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.selection_method = selection_method
        self.val_fraction = val_fraction
        self.early_stop_patience = early_stop_patience
        self.aggregation = aggregation
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """
        Fit the LGB+ (Competition) model.

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
        self : LGBPlus
            Fitted model.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        # Pre-standardize columns for fast correlation computation
        X_centered = X - X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std[X_std < 1e-10] = 1.0
        X_standardized = X_centered / X_std

        # Initialize storage
        self.ensemble_ = []
        self.step_type_counts_ = np.zeros((self.n_ensemble, 2), dtype=int)
        self.linear_feature_counts_ = np.zeros(n_features, dtype=int)
        self.feature_importances_ = np.zeros(n_features)
        self.val_losses_ = []

        # LightGBM base parameters
        lgb_params = dict(
            objective="regression",
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            lambda_l2=self.lambda_l2,
            learning_rate=1.0,  # We apply learning_rate manually
            verbosity=-1,
            force_col_wise=True,
            num_threads=1,  # Single thread per tree for stability
        )

        # Validate selection_method
        if self.selection_method not in ['oob', 'validation', 'training']:
            raise ValueError(f"selection_method must be 'oob', 'validation', or 'training', got {self.selection_method}")

        # Create fixed validation set only if selection_method='validation' and val_fraction > 0
        if self.selection_method == 'validation' and self.val_fraction > 0:
            rng_val = np.random.default_rng(self.random_state)
            n_val = int(n_samples * self.val_fraction)
            val_idx = rng_val.choice(n_samples, size=n_val, replace=False)
            train_pool_idx = np.setdiff1d(np.arange(n_samples), val_idx)
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            # For OOB and training methods, use all samples for training
            train_pool_idx = np.arange(n_samples)
            X_val, y_val = None, None

        if self.verbose:
            print(f"Selection method: {self.selection_method}")
            if self.selection_method == 'oob':
                oob_size = int(n_samples * (1 - self.subsample))
                print(f"OOB samples per step: ~{oob_size} ({100*(1-self.subsample):.0f}% of training)")

        # Train each ensemble member
        for b in range(self.n_ensemble):
            if self.verbose:
                print(f"Training ensemble member {b+1}/{self.n_ensemble}")

            member = self._fit_one_member(
                X, y, X_standardized, X_val, y_val, train_pool_idx,
                lgb_params, b
            )
            self.ensemble_.append(member)

            # Accumulate stats
            self.step_type_counts_[b] = member['step_type_counts']
            self.linear_feature_counts_ += member['linear_feature_counts']
            self.feature_importances_ += member['feature_importances']
            self.val_losses_.append(member['val_loss_history'])

        return self

    def _fit_one_member(
        self, X, y, X_standardized, X_val, y_val, train_pool_idx,
        lgb_params, member_idx
    ):
        """Fit a single ensemble member."""
        import lightgbm as lgb

        n_samples, n_features = X.shape
        n_train_pool = len(train_pool_idx)
        subsample_size = int(n_train_pool * self.subsample)

        # Set seed for this member
        seed = None if self.random_state is None else self.random_state + member_idx
        rng = np.random.default_rng(seed)

        if seed is not None:
            lgb_params_run = {**lgb_params, 'seed': seed}
        else:
            lgb_params_run = lgb_params.copy()

        # Initialize
        init_pred = float(y.mean())
        pred = np.full(n_samples, init_pred)

        if X_val is not None:
            pred_val = np.full(len(y_val), init_pred)

        steps = []
        step_type_counts = np.array([0, 0], dtype=int)  # [tree, linear]
        linear_feature_counts = np.zeros(n_features, dtype=int)
        feature_importances = np.zeros(n_features)
        val_loss_history = []

        best_val_loss = np.inf
        steps_without_improvement = 0

        for t in range(self.n_steps):
            # Step A: Sample rows
            S_t = rng.choice(train_pool_idx, size=subsample_size, replace=False)
            X_S, y_S = X[S_t], y[S_t]
            resid_S = y_S - pred[S_t]

            # Step B: Build candidate TREE update
            dtrain = lgb.Dataset(X_S, label=resid_S, free_raw_data=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(lgb_params_run, dtrain, num_boost_round=1)

            g_tree_all = booster.predict(X)
            f_tree = pred + self.learning_rate * g_tree_all

            # Step C: Build candidate LINEAR update
            # Compute correlations on subset using pre-standardized data
            resid_S_centered = resid_S - resid_S.mean()
            resid_S_std = resid_S.std()
            if resid_S_std < 1e-10:
                resid_S_std = 1.0
            resid_S_standardized = resid_S_centered / resid_S_std

            # Correlation = dot product of standardized vectors / n
            corrs = (X_standardized[S_t].T @ resid_S_standardized) / len(S_t)
            corrs = np.nan_to_num(corrs, nan=0.0)

            j_star = np.argmax(np.abs(corrs))

            # OLS coefficient on subset
            x_j_S = X_S[:, j_star]
            x_j_S_sq_sum = (x_j_S ** 2).sum() + 1e-10
            b = (x_j_S @ resid_S) / x_j_S_sq_sum

            g_lin_all = b * X[:, j_star]
            f_lin = pred + self.learning_rate * g_lin_all

            # Step D: Choose winner based on selection method
            if self.selection_method == 'oob':
                # Out-of-bag: evaluate on samples NOT in current subsample
                oob_idx = np.setdiff1d(train_pool_idx, S_t)
                X_oob = X[oob_idx]
                y_oob = y[oob_idx]
                pred_oob = pred[oob_idx]

                f_tree_oob = pred_oob + self.learning_rate * booster.predict(X_oob)
                f_lin_oob = pred_oob + self.learning_rate * b * X_oob[:, j_star]
                L_tree = np.mean((y_oob - f_tree_oob) ** 2)
                L_lin = np.mean((y_oob - f_lin_oob) ** 2)

            elif self.selection_method == 'validation' and X_val is not None:
                # Validation set: evaluate on fixed held-out set
                f_tree_val = pred_val + self.learning_rate * booster.predict(X_val)
                f_lin_val = pred_val + self.learning_rate * b * X_val[:, j_star]
                L_tree = np.mean((y_val - f_tree_val) ** 2)
                L_lin = np.mean((y_val - f_lin_val) ** 2)

            else:
                # Training: evaluate on current subsample (not recommended)
                L_tree = np.mean((y_S - f_tree[S_t]) ** 2)
                L_lin = np.mean((y_S - f_lin[S_t]) ** 2)

            # Choose and apply update
            if L_tree <= L_lin:
                # Accept tree step
                pred = f_tree
                if self.selection_method == 'validation' and X_val is not None:
                    pred_val = f_tree_val

                steps.append({
                    'type': 'tree',
                    'booster': booster
                })
                step_type_counts[0] += 1

                # Accumulate feature importance
                try:
                    importance = booster.feature_importance(importance_type='gain')
                    if len(importance) == n_features:
                        feature_importances += importance
                except:
                    pass

                chosen_type = 'tree'
                current_val_loss = L_tree
            else:
                # Accept linear step
                pred = f_lin
                if self.selection_method == 'validation' and X_val is not None:
                    pred_val = f_lin_val

                steps.append({
                    'type': 'linear',
                    'feature_idx': j_star,
                    'coef': b
                })
                step_type_counts[1] += 1
                linear_feature_counts[j_star] += 1

                chosen_type = 'linear'
                current_val_loss = L_lin

            # For validation method, update current_val_loss using full validation set
            if self.selection_method == 'validation' and X_val is not None:
                current_val_loss = np.mean((y_val - pred_val) ** 2)

            val_loss_history.append(current_val_loss)

            # Early stopping check
            if self.early_stop_patience is not None:
                if current_val_loss < best_val_loss - 1e-10:
                    best_val_loss = current_val_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if steps_without_improvement >= self.early_stop_patience:
                    if self.verbose:
                        print(f"  Early stopping at step {t+1}")
                    break

            # Verbose logging
            if self.verbose and (t + 1) % 50 == 0:
                train_rmse = np.sqrt(np.mean((y - pred) ** 2))
                tree_pct = 100 * step_type_counts[0] / (t + 1)
                print(f"  Step {t+1}/{self.n_steps}: RMSE={train_rmse:.4f}, "
                      f"tree={tree_pct:.0f}%, last={chosen_type}")

        return {
            'init': init_pred,
            'steps': steps,
            'step_type_counts': step_type_counts,
            'linear_feature_counts': linear_feature_counts,
            'feature_importances': feature_importances,
            'val_loss_history': val_loss_history
        }

    def predict(self, X) -> np.ndarray:
        """
        Predict using the fitted ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (ensemble aggregation).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        all_preds = np.zeros((self.n_ensemble, n_samples))

        for b, member in enumerate(self.ensemble_):
            pred = np.full(n_samples, member['init'])

            for step in member['steps']:
                if step['type'] == 'tree':
                    pred = pred + self.learning_rate * step['booster'].predict(X)
                else:  # linear
                    j = step['feature_idx']
                    pred = pred + self.learning_rate * step['coef'] * X[:, j]

            all_preds[b] = pred

        if self.aggregation == 'median':
            return np.median(all_preds, axis=0)
        else:
            return np.mean(all_preds, axis=0)

    def predict_individual(self, X) -> np.ndarray:
        """
        Get predictions from each ensemble member.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        predictions : ndarray of shape (n_ensemble, n_samples)
            Predictions from each ensemble member.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        all_preds = np.zeros((self.n_ensemble, n_samples))

        for b, member in enumerate(self.ensemble_):
            pred = np.full(n_samples, member['init'])

            for step in member['steps']:
                if step['type'] == 'tree':
                    pred = pred + self.learning_rate * step['booster'].predict(X)
                else:
                    j = step['feature_idx']
                    pred = pred + self.learning_rate * step['coef'] * X[:, j]

            all_preds[b] = pred

        return all_preds

    def get_step_type_summary(self) -> Dict[str, Any]:
        """
        Get summary of step type choices across ensemble.

        Returns
        -------
        summary : dict
            Contains 'total_tree', 'total_linear', 'tree_fraction',
            'per_member' arrays.
        """
        total_tree = self.step_type_counts_[:, 0].sum()
        total_linear = self.step_type_counts_[:, 1].sum()
        total = total_tree + total_linear

        return {
            'total_tree': int(total_tree),
            'total_linear': int(total_linear),
            'tree_fraction': total_tree / total if total > 0 else 0,
            'per_member_tree': self.step_type_counts_[:, 0],
            'per_member_linear': self.step_type_counts_[:, 1]
        }

    def get_linear_feature_summary(self) -> Dict[str, int]:
        """
        Get summary of linear feature selection frequency.

        Returns
        -------
        summary : dict
            Feature name -> selection count, sorted by count.
        """
        sorted_idx = np.argsort(self.linear_feature_counts_)[::-1]
        return {
            self.feature_names_[i]: int(self.linear_feature_counts_[i])
            for i in sorted_idx
            if self.linear_feature_counts_[i] > 0
        }

    def summary(self) -> str:
        """Return a summary of the fitted model."""
        if not hasattr(self, 'ensemble_'):
            return "Model not fitted yet."

        step_summary = self.get_step_type_summary()

        lines = [
            "LGBPlus Summary",
            "=" * 50,
            f"Ensemble members: {self.n_ensemble}",
            f"Steps per member: {self.n_steps} (max)",
            f"Learning rate: {self.learning_rate}",
            f"Selection method: {self.selection_method}",
            f"Row subsample: {self.subsample}",
            f"Tree leaves: {self.num_leaves}",
            "",
            "Step type distribution:",
            f"  Tree steps: {step_summary['total_tree']} ({100*step_summary['tree_fraction']:.1f}%)",
            f"  Linear steps: {step_summary['total_linear']} ({100*(1-step_summary['tree_fraction']):.1f}%)",
            "",
            "Linear feature selection frequency:",
        ]

        for fname, count in self.get_linear_feature_summary().items():
            lines.append(f"  {fname}: {count} times")

        return "\n".join(lines)


# Convenience function
def lgb_plus(
    X_train, y_train, X_test=None,
    n_ensemble: int = 10,
    n_steps: int = 200,
    learning_rate: float = 0.05,
    subsample: float = 0.7,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Tuple['LGBPlus', np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to fit LGBPlus and get predictions.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    X_test : array-like, optional
        Test features. If provided, returns test predictions.
    n_ensemble : int
        Number of ensemble members.
    n_steps : int
        Boosting steps per member.
    learning_rate : float
        Learning rate.
    subsample : float
        Row subsampling rate.
    random_state : int, optional
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    model : LGBPlus
        Fitted model.
    train_pred : ndarray
        Training predictions.
    test_pred : ndarray or None
        Test predictions (if X_test provided).
    """
    model = LGBPlus(
        n_ensemble=n_ensemble,
        n_steps=n_steps,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
        verbose=verbose
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) if X_test is not None else None

    return model, train_pred, test_pred


if __name__ == "__main__":
    # Quick demo
    print("LGBPlus Demo")
    print("=" * 50)

    np.random.seed(42)

    # Generate hybrid DGP data (AR-type with both linear and nonlinear)
    n = 2000
    p = 6
    X = np.random.randn(n, p)

    # DGP: y = linear + nonlinear + interaction + noise
    y = (0.6 * X[:, 0] +                         # strong linear
         0.3 * X[:, 1] +                         # moderate linear
         0.4 * np.tanh(2 * X[:, 2]) +            # nonlinear
         0.2 * X[:, 0] * (X[:, 3] > 0) +         # interaction
         np.random.randn(n) * 0.3)               # noise

    # Train/test split
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

    # Fit model with OOB selection (recommended)
    print("\nFitting LGBPlus with OOB selection...")
    model = LGBPlus(
        n_ensemble=10,
        n_steps=200,
        learning_rate=0.05,
        subsample=0.7,
        num_leaves=5,
        selection_method='oob',  # Recommended for macro
        early_stop_patience=50,
        random_state=42,
        verbose=True
    )
    model.fit(X_train, y_train, feature_names=[f'x{i}' for i in range(p)])

    # Evaluate
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_rmse = np.sqrt(np.mean((y_train - pred_train) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - pred_test) ** 2))

    print(f"\nTrain RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"\n{model.summary()}")

    # Compare with vanilla LightGBM
    print("\n" + "=" * 50)
    print("Comparison with vanilla LightGBM:")

    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred_test = lgb_model.predict(X_test)
    lgb_rmse = np.sqrt(np.mean((y_test - lgb_pred_test) ** 2))
    print(f"Vanilla LightGBM Test RMSE: {lgb_rmse:.4f}")
