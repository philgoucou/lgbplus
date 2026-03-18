#' =============================================================================
#' LGB+ (Competition): Per-Step Tree vs Linear Competition
#' =============================================================================
#'
#' This implements the COMPETITION-BASED algorithm where at each step:
#' 1. Build a candidate tree update
#' 2. Build a candidate linear update
#' 3. Choose whichever reduces loss more
#'
#' Selection methods for choosing between tree/linear:
#' - "oob": Out-of-bag evaluation (recommended for macro). Uses samples NOT in
#'          the current subsample S_t. Different "judges" each step.
#' - "validation": Fixed held-out validation set (original implementation).
#' - "training": Evaluate on training subsample S_t (not recommended, tree has
#'               structural advantage).
#'
#' This is different from LGB^A+ (`lgb_plus_A.R`) which always applies BOTH.
#'
#' Use this if you want the algorithm that adapts its tree/linear ratio
#' based on the data (can go 100% trees or 100% linear if optimal).
#'
#' @author Philippe Goulet Coulombe (UQAM)
#' =============================================================================

# Check and load required packages
.check_lightgbm_plus <- function() {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("Package 'lightgbm' is required. Install with: install.packages('lightgbm')")
  }
}


#' Fit an LGB+ model (competition-based)
#'
#' At each step, competes a tree update against a linear update and chooses
#' whichever reduces loss more.
#'
#' @param X Matrix or data.frame of features (n_samples x n_features)
#' @param y Numeric vector of target values
#' @param n_steps Number of boosting steps (default: 200)
#' @param learning_rate Learning rate for both updates (default: 0.05)
#' @param subsample Row subsampling rate per step (default: 0.7)
#' @param num_leaves Max leaves per tree (default: 5)
#' @param min_data Minimum samples in leaf (default: 20)
#' @param lambda_l2 L2 regularization for trees (default: 0.1)
#' @param linear_candidate_fraction Fraction of features to consider as candidates for
#'        the linear step (default: 0.5). At each step, randomly sample this fraction
#'        of features, compute correlations only on those, and pick the best one.
#'        Similar to RF's mtry but for the linear step.
#' @param selection_method Method for choosing between tree/linear updates:
#'        "oob" (recommended) = evaluate on out-of-bag samples (not in current subsample)
#'        "validation" = evaluate on fixed held-out validation set
#'        "training" = evaluate on training subsample (not recommended)
#' @param val_fraction Fraction for validation set (only used if selection_method="validation",
#'        default: 0.2)
#' @param early_stop_patience Early stopping patience (default: 50, NULL to disable)
#' @param seed Random seed (default: NULL)
#' @param verbose Print progress (default: FALSE)
#'
#' @return A lgb_plus model object
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000 * 5), ncol = 5)
#' y <- 0.5 * X[,1] + 0.3 * tanh(X[,2]) + rnorm(1000) * 0.1
#' # Recommended: OOB selection
#' model <- lgb_plus(X, y, selection_method = "oob")
#' pred <- predict(model, X)
#' print(model)  # Shows tree vs linear step distribution
#' }
#'
#' @export
lgb_plus <- function(
    X, y,
    n_steps = 200,
    learning_rate = 0.05,
    subsample = 0.7,
    num_leaves = 5,
    min_data = 20,
    lambda_l2 = 0.1,
    linear_candidate_fraction = 0.5,
    selection_method = "oob",
    val_fraction = 0.2,
    early_stop_patience = 50,
    seed = NULL,
    verbose = FALSE
) {
  .check_lightgbm_plus()

  # Convert inputs
  X <- as.matrix(X)
  y <- as.numeric(y)

  n_samples <- nrow(X)
  n_features <- ncol(X)

  feature_names <- colnames(X)
  if (is.null(feature_names)) {
    feature_names <- paste0("V", seq_len(n_features))
  }

  # Pre-standardize for fast correlation computation
  X_means <- colMeans(X, na.rm = TRUE)
  X_sds <- apply(X, 2, sd, na.rm = TRUE)
  X_sds[X_sds < 1e-10] <- 1.0
  X_standardized <- scale(X, center = X_means, scale = X_sds)

  # Set seed
  if (!is.null(seed)) set.seed(seed)

  # Validate selection_method
  selection_method <- match.arg(selection_method, c("oob", "validation", "training"))

  # Create validation set (only for selection_method = "validation")
  if (selection_method == "validation" && val_fraction > 0) {
    n_val <- floor(n_samples * val_fraction)
    val_idx <- sample(n_samples, n_val, replace = FALSE)
    train_pool_idx <- setdiff(seq_len(n_samples), val_idx)
    X_val <- X[val_idx, , drop = FALSE]
    y_val <- y[val_idx]
  } else {
    # For OOB and training methods, use all samples for training
    train_pool_idx <- seq_len(n_samples)
    X_val <- NULL
    y_val <- NULL
  }

  n_train_pool <- length(train_pool_idx)
  subsample_size <- floor(n_train_pool * subsample)

  if (verbose) {
    cat(sprintf("Selection method: %s\n", selection_method))
    if (selection_method == "oob") {
      cat(sprintf("OOB samples per step: ~%d (%.0f%% of training)\n",
                  n_train_pool - subsample_size, 100 * (1 - subsample)))
    } else if (selection_method == "validation") {
      cat(sprintf("Validation set size: %d (%.0f%% of data)\n",
                  length(y_val), 100 * val_fraction))
    }
  }

  # LightGBM parameters
  lgb_params <- list(
    objective = "regression",
    num_leaves = num_leaves,
    min_data_in_leaf = min_data,
    lambda_l2 = lambda_l2,
    learning_rate = 1.0,  # We apply learning_rate manually
    verbosity = -1,
    force_col_wise = TRUE,
    num_threads = 1
  )
  if (!is.null(seed)) {
    lgb_params$seed <- seed
  }

  # Initialize
  init_pred <- mean(y)
  pred <- rep(init_pred, n_samples)

  if (!is.null(X_val)) {
    pred_val <- rep(init_pred, length(y_val))
  }

  steps <- list()
  step_type_counts <- c(tree = 0L, linear = 0L)
  linear_feature_counts <- rep(0L, n_features)
  names(linear_feature_counts) <- feature_names

  best_val_loss <- Inf
  steps_without_improvement <- 0

  # Storage for OOB/validation loss tracking
  oob_loss_history <- numeric(n_steps)
  train_loss_history <- numeric(n_steps)

  # Main loop
  for (t in seq_len(n_steps)) {
    # Step A: Sample rows
    S_t <- sample(train_pool_idx, subsample_size, replace = FALSE)
    X_S <- X[S_t, , drop = FALSE]
    y_S <- y[S_t]
    resid_S <- y_S - pred[S_t]

    # Step B: Build candidate TREE update
    dtrain <- lightgbm::lgb.Dataset(
      data = X_S,
      label = resid_S,
      free_raw_data = FALSE
    )

    booster <- lightgbm::lgb.train(
      params = lgb_params,
      data = dtrain,
      nrounds = 1,
      verbose = -1
    )

    g_tree_all <- predict(booster, X)
    f_tree <- pred + learning_rate * g_tree_all

    # Step C: Build candidate LINEAR update
    # Randomly sample candidate features (like RF's mtry)
    n_candidates <- max(1L, floor(n_features * linear_candidate_fraction))
    candidate_idx <- sample(n_features, n_candidates, replace = FALSE)

    # Compute correlations only on candidate features
    resid_S_centered <- resid_S - mean(resid_S)
    resid_S_sd <- sd(resid_S)
    if (resid_S_sd < 1e-10) resid_S_sd <- 1.0
    resid_S_standardized <- resid_S_centered / resid_S_sd

    # Correlation only on candidates
    corrs_candidates <- as.vector(crossprod(X_standardized[S_t, candidate_idx, drop = FALSE], resid_S_standardized)) / length(S_t)
    corrs_candidates[is.na(corrs_candidates)] <- 0

    # Pick best among candidates
    best_candidate <- which.max(abs(corrs_candidates))
    j_star <- candidate_idx[best_candidate]

    # OLS coefficient on subset
    x_j_S <- X_S[, j_star]
    x_j_S_sq_sum <- sum(x_j_S^2) + 1e-10
    b <- sum(x_j_S * resid_S) / x_j_S_sq_sum

    g_lin_all <- b * X[, j_star]
    f_lin <- pred + learning_rate * g_lin_all

    # Step D: Choose winner based on selection method
    if (selection_method == "oob") {
      # Out-of-bag: evaluate on samples NOT in current subsample
      oob_idx <- setdiff(train_pool_idx, S_t)
      X_oob <- X[oob_idx, , drop = FALSE]
      y_oob <- y[oob_idx]
      pred_oob <- pred[oob_idx]

      f_tree_oob <- pred_oob + learning_rate * predict(booster, X_oob)
      f_lin_oob <- pred_oob + learning_rate * b * X_oob[, j_star]
      L_tree <- mean((y_oob - f_tree_oob)^2)
      L_lin <- mean((y_oob - f_lin_oob)^2)

    } else if (selection_method == "validation" && !is.null(X_val)) {
      # Validation set: evaluate on fixed held-out set
      f_tree_val <- pred_val + learning_rate * predict(booster, X_val)
      f_lin_val <- pred_val + learning_rate * b * X_val[, j_star]
      L_tree <- mean((y_val - f_tree_val)^2)
      L_lin <- mean((y_val - f_lin_val)^2)

    } else {
      # Training: evaluate on current subsample (not recommended)
      L_tree <- mean((y_S - f_tree[S_t])^2)
      L_lin <- mean((y_S - f_lin[S_t])^2)
    }

    # Choose and apply update
    if (L_tree <= L_lin) {
      # Accept tree step
      pred <- f_tree
      if (!is.null(X_val)) {
        pred_val <- f_tree_val
      }

      steps[[t]] <- list(
        type = "tree",
        booster = booster
      )
      step_type_counts["tree"] <- step_type_counts["tree"] + 1L

      chosen_type <- "tree"
    } else {
      # Accept linear step
      pred <- f_lin
      if (!is.null(X_val)) {
        pred_val <- f_lin_val
      }

      steps[[t]] <- list(
        type = "linear",
        feature_idx = j_star,
        coef = b
      )
      step_type_counts["linear"] <- step_type_counts["linear"] + 1L
      linear_feature_counts[j_star] <- linear_feature_counts[j_star] + 1L

      chosen_type <- "linear"
    }

    # Track loss for early stopping
    if (selection_method == "oob") {
      # For OOB, use the OOB loss from this step
      current_val_loss <- L_tree  # Use whichever was chosen (will be min of the two)
      if (L_lin < L_tree) current_val_loss <- L_lin
    } else if (selection_method == "validation" && !is.null(X_val)) {
      current_val_loss <- mean((y_val - pred_val)^2)
    } else {
      current_val_loss <- mean((y - pred)^2)
    }

    # Store loss history
    oob_loss_history[t] <- current_val_loss
    train_loss_history[t] <- mean((y - pred)^2)

    # Early stopping check
    if (!is.null(early_stop_patience)) {
      if (current_val_loss < best_val_loss - 1e-10) {
        best_val_loss <- current_val_loss
        steps_without_improvement <- 0
      } else {
        steps_without_improvement <- steps_without_improvement + 1
      }

      if (steps_without_improvement >= early_stop_patience) {
        if (verbose) {
          cat(sprintf("  Early stopping at step %d\n", t))
        }
        break
      }
    }

    # Verbose logging
    if (verbose && t %% 50 == 0) {
      train_rmse <- sqrt(mean((y - pred)^2))
      tree_pct <- 100 * step_type_counts["tree"] / t
      cat(sprintf("  Step %d/%d: RMSE=%.4f, tree=%.0f%%, last=%s\n",
                  t, n_steps, train_rmse, tree_pct, chosen_type))
    }
  }

  # Return model object
  actual_steps <- length(steps)  # May be < n_steps if early stopped
  model <- list(
    init = init_pred,
    steps = steps,
    n_steps = n_steps,
    actual_steps = actual_steps,
    learning_rate = learning_rate,
    n_features = n_features,
    feature_names = feature_names,
    step_type_counts = step_type_counts,
    linear_feature_counts = linear_feature_counts,
    selection_method = selection_method,
    linear_candidate_fraction = linear_candidate_fraction,
    # Loss tracking
    oob_loss_history = oob_loss_history[1:actual_steps],
    train_loss_history = train_loss_history[1:actual_steps],
    final_oob_loss = current_val_loss,
    final_train_loss = mean((y - pred)^2)
  )

  class(model) <- "lgb_plus"
  return(model)
}


#' Predict method for lgb_plus
#'
#' @param object A lgb_plus model
#' @param newdata Matrix or data.frame of features
#' @param ... Additional arguments (ignored)
#'
#' @return Numeric vector of predictions
#'
#' @export
predict.lgb_plus <- function(object, newdata, ...) {
  .check_lightgbm_plus()

  X <- as.matrix(newdata)
  n_samples <- nrow(X)

  pred <- rep(object$init, n_samples)

  for (step in object$steps) {
    if (step$type == "tree") {
      pred <- pred + object$learning_rate * predict(step$booster, X)
    } else {
      # Linear step (single feature)
      j <- step$feature_idx
      pred <- pred + object$learning_rate * step$coef * X[, j]
    }
  }

  return(pred)
}


#' Get linear component only
#'
#' @param object A lgb_plus model
#' @param newdata Matrix or data.frame of features
#'
#' @return Numeric vector of linear component predictions
#'
#' @export
predict_linear_component.lgb_plus <- function(object, newdata) {
  X <- as.matrix(newdata)
  n_samples <- nrow(X)

  pred <- rep(0, n_samples)

  for (step in object$steps) {
    if (step$type == "linear") {
      j <- step$feature_idx
      pred <- pred + object$learning_rate * step$coef * X[, j]
    }
  }

  return(pred)
}


#' Get trees component only
#'
#' @param object A lgb_plus model
#' @param newdata Matrix or data.frame of features
#'
#' @return Numeric vector of trees component predictions
#'
#' @export
predict_trees_component.lgb_plus <- function(object, newdata) {
  .check_lightgbm_plus()

  X <- as.matrix(newdata)
  n_samples <- nrow(X)

  pred <- rep(0, n_samples)

  for (step in object$steps) {
    if (step$type == "tree") {
      pred <- pred + object$learning_rate * predict(step$booster, X)
    }
  }

  return(pred)
}


#' Print method for lgb_plus
#'
#' @param x A lgb_plus model
#' @param ... Additional arguments (ignored)
#'
#' @export
print.lgb_plus <- function(x, ...) {
  total_steps <- sum(x$step_type_counts)
  tree_pct <- 100 * x$step_type_counts["tree"] / total_steps

  cat("LGB+ (Competition) Model\n")
  cat("============================\n")
  cat(sprintf("Total steps: %d (max: %d)\n", total_steps, x$n_steps))
  cat(sprintf("Learning rate: %.3f\n", x$learning_rate))
  cat(sprintf("Selection method: %s\n", x$selection_method))
  cat(sprintf("Linear candidate fraction: %.0f%%\n", 100 * x$linear_candidate_fraction))
  cat(sprintf("Features: %d\n", x$n_features))
  cat("\nStep type distribution:\n")
  cat(sprintf("  Tree steps: %d (%.1f%%)\n", x$step_type_counts["tree"], tree_pct))
  cat(sprintf("  Linear steps: %d (%.1f%%)\n", x$step_type_counts["linear"], 100 - tree_pct))
  cat("\nLinear feature selection frequency:\n")

  sorted_counts <- sort(x$linear_feature_counts, decreasing = TRUE)
  for (i in seq_along(sorted_counts)) {
    if (sorted_counts[i] > 0) {
      cat(sprintf("  %s: %d times\n", names(sorted_counts)[i], sorted_counts[i]))
    }
  }

  invisible(x)
}


#' Fit an ensemble of competition-based models
#'
#' @param X Matrix or data.frame of features
#' @param y Numeric vector of target values
#' @param n_ensemble Number of ensemble members (default: 5)
#' @param selection_method Method for choosing between tree/linear updates:
#'        "oob" (recommended), "validation", or "training"
#' @param ... Additional arguments passed to lgb_plus
#'
#' @return A lgb_plus_ensemble model object
#'
#' @export
lgb_plus_ensemble <- function(
    X, y,
    n_ensemble = 5,
    selection_method = "oob",
    base_seed = 123,
    ...
) {
  models <- list()
  step_type_totals <- c(tree = 0L, linear = 0L)

  for (i in seq_len(n_ensemble)) {
    model_seed <- base_seed + i - 1

    models[[i]] <- lgb_plus(
      X = X, y = y,
      selection_method = selection_method,
      seed = model_seed,
      ...
    )

    step_type_totals <- step_type_totals + models[[i]]$step_type_counts
  }

  ensemble <- list(
    models = models,
    n_ensemble = n_ensemble,
    step_type_totals = step_type_totals,
    n_features = models[[1]]$n_features,
    feature_names = models[[1]]$feature_names,
    learning_rate = models[[1]]$learning_rate,
    selection_method = selection_method,
    linear_candidate_fraction = models[[1]]$linear_candidate_fraction
  )

  class(ensemble) <- "lgb_plus_ensemble"
  return(ensemble)
}


#' Predict method for lgb_plus_ensemble
#'
#' @param object A lgb_plus_ensemble model
#' @param newdata Matrix or data.frame of features
#' @param ... Additional arguments (ignored)
#'
#' @return Numeric vector of ensemble-averaged predictions
#'
#' @export
predict.lgb_plus_ensemble <- function(object, newdata, ...) {
  preds <- sapply(object$models, function(m) predict(m, newdata))
  rowMeans(preds)
}


#' Get linear component for ensemble
#'
#' @param object A lgb_plus_ensemble model
#' @param newdata Matrix or data.frame of features
#'
#' @return Numeric vector of ensemble-averaged linear component
#'
#' @export
predict_linear_component.lgb_plus_ensemble <- function(object, newdata) {
  preds <- sapply(object$models, function(m) {
    predict_linear_component.lgb_plus(m, newdata)
  })
  rowMeans(preds)
}


#' Get trees component for ensemble
#'
#' @param object A lgb_plus_ensemble model
#' @param newdata Matrix or data.frame of features
#'
#' @return Numeric vector of ensemble-averaged trees component
#'
#' @export
predict_trees_component.lgb_plus_ensemble <- function(object, newdata) {
  preds <- sapply(object$models, function(m) {
    predict_trees_component.lgb_plus(m, newdata)
  })
  rowMeans(preds)
}


#' Print method for lgb_plus_ensemble
#'
#' @param x A lgb_plus_ensemble model
#' @param ... Additional arguments (ignored)
#'
#' @export
print.lgb_plus_ensemble <- function(x, ...) {
  total_steps <- sum(x$step_type_totals)
  tree_pct <- 100 * x$step_type_totals["tree"] / total_steps

  cat("LGB+ Ensemble\n")
  cat("================================\n")
  cat(sprintf("Ensemble members: %d\n", x$n_ensemble))
  cat(sprintf("Learning rate: %.3f\n", x$learning_rate))
  cat(sprintf("Selection method: %s\n", x$selection_method))
  cat(sprintf("Linear candidate fraction: %.0f%%\n", 100 * x$linear_candidate_fraction))
  cat(sprintf("Features: %d\n", x$n_features))
  cat("\nAggregate step distribution (all members):\n")
  cat(sprintf("  Tree steps: %d (%.1f%%)\n", x$step_type_totals["tree"], tree_pct))
  cat(sprintf("  Linear steps: %d (%.1f%%)\n", x$step_type_totals["linear"], 100 - tree_pct))

  invisible(x)
}


# =============================================================================
# Loss History Utilities
# =============================================================================

#' Get loss history from lgb_plus model
#'
#' @param object A lgb_plus model
#' @return Data frame with step, oob_loss, train_loss columns
#'
#' @export
get_loss_history <- function(object) {
  if (is.null(object$oob_loss_history)) {
    stop("Model does not contain loss history. Was it trained with an older version?")
  }
  data.frame(
    step = seq_along(object$oob_loss_history),
    oob_loss = object$oob_loss_history,
    train_loss = object$train_loss_history
  )
}

#' Plot loss curves for lgb_plus model
#'
#' @param object A lgb_plus model
#' @param ... Additional arguments passed to plot()
#'
#' @export
plot_loss_history <- function(object, ...) {
  if (is.null(object$oob_loss_history)) {
    stop("Model does not contain loss history.")
  }

  steps <- seq_along(object$oob_loss_history)
  ylim <- range(c(object$train_loss_history, object$oob_loss_history))

  plot(steps, object$train_loss_history, type = "l", col = "blue",
       xlab = "Boosting Step", ylab = "MSE Loss", ylim = ylim,
       main = "LGB+C Boosting Curves", ...)
  lines(steps, object$oob_loss_history, col = "red")
  legend("topright", legend = c("Train", "OOB"), col = c("blue", "red"), lty = 1)
}


# =============================================================================
# Demo
# =============================================================================

if (FALSE) {  # Set to TRUE to run demo

  set.seed(42)

  # Generate hybrid DGP data
  n <- 2000
  X <- matrix(rnorm(n * 4), ncol = 4)
  colnames(X) <- c("x1", "x2", "x3", "x4")

  # y = linear + nonlinear + noise
  y <- 0.5 * X[, 1] +                      # linear
    0.3 * tanh(2 * X[, 2]) +               # nonlinear
    0.2 * X[, 1] * (X[, 2] > 0) +          # interaction
    rnorm(n) * 0.3                         # noise

  # Train/test split
  train_idx <- 1:1500
  test_idx <- 1501:2000

  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[test_idx, ]
  y_test <- y[test_idx]

  # Fit single model with OOB selection
  cat("Fitting single competition model (OOB selection)...\n")
  model <- lgb_plus(X_train, y_train, selection_method = "oob", verbose = TRUE)
  print(model)

  pred_test <- predict(model, X_test)
  cat(sprintf("\nTest RMSE: %.4f\n", sqrt(mean((y_test - pred_test)^2))))

  # Fit ensemble with OOB selection
  cat("\nFitting competition ensemble (OOB selection)...\n")
  ensemble <- lgb_plus_ensemble(
    X_train, y_train,
    n_ensemble = 5,
    selection_method = "oob",
    n_steps = 200,
    learning_rate = 0.05,
    verbose = FALSE
  )
  print(ensemble)

  pred_test_ens <- predict(ensemble, X_test)
  cat(sprintf("Ensemble Test RMSE: %.4f\n", sqrt(mean((y_test - pred_test_ens)^2))))

  # Decomposition
  linear_comp <- predict_linear_component.lgb_plus_ensemble(ensemble, X_test)
  trees_comp <- predict_trees_component.lgb_plus_ensemble(ensemble, X_test)
  cat(sprintf("\nLinear component variance: %.4f\n", var(linear_comp)))
  cat(sprintf("Trees component variance: %.4f\n", var(trees_comp)))
}
