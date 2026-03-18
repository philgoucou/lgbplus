#' LGB^A+: Hybrid Tree + Linear Boosting
#'
#' A gradient boosting variant that alternates between:
#' 1. Fitting a block of trees on residuals
#' 2. Fitting a single linear term (greedy feature selection) on residuals
#'
#' This captures both nonlinear patterns (via trees) and linear structure
#' (via repeated linear updates), making it robust to unknown DGPs.
#'
#' @author Philippe Goulet Coulombe (UQAM)
#' @license MIT

# Check and load required packages
.check_lightgbm <- function() {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("Package 'lightgbm' is required. Install with: install.packages('lightgbm')")
  }
}

#' Fit an LGB^A+ model
#'
#' @param X Matrix or data.frame of features (n_samples x n_features)
#' @param y Numeric vector of target values
#' @param n_cycles Number of alternating cycles (default: 25)
#' @param trees_per_cycle Trees to fit per cycle (default: 10)
#' @param lr_tree Learning rate for trees (default: 0.02)
#' @param lr_linear Learning rate for linear updates (default: 0.1)
#' @param num_leaves Max leaves per tree (default: 15)
#' @param min_data Minimum samples in leaf (default: 20)
#' @param subsample Row subsampling fraction for trees (default: 1.0, no subsampling)
#' @param seed Random seed (default: NULL)
#' @param verbose Print progress (default: FALSE)
#'
#' @return An lgb_plus_A model object (list)
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000 * 4), ncol = 4)
#' y <- 0.5 * X[,1] + 0.3 * tanh(X[,2]) + rnorm(1000) * 0.1
#' model <- lgb_plus_A(X, y, subsample = 0.75)
#' pred <- predict(model, X)
#' }
#'
#' @export
lgb_plus_A <- function(
    X, y,
    n_cycles = 25,
    trees_per_cycle = 10,
    lr_tree = 0.02,
    lr_linear = 0.1,
    num_leaves = 15,
    min_data = 20,
    subsample = 1.0,
    seed = NULL,
    verbose = FALSE
) {
  .check_lightgbm()
  
  # Convert inputs
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  feature_names <- colnames(X)
  if (is.null(feature_names)) {
    feature_names <- paste0("V", seq_len(n_features))
  }
  
  # Initialize
  init_pred <- mean(y)
  pred <- rep(init_pred, n_samples)
  
  trees <- list()
  linear_steps <- list()
  linear_feature_counts <- rep(0, n_features)
  names(linear_feature_counts) <- feature_names
  
  # LightGBM parameters
  lgb_params <- list(
    objective = "regression",
    num_leaves = num_leaves,
    min_data_in_leaf = min_data,
    learning_rate = 1.0,  # We apply lr_tree manually
    verbosity = -1,
    force_col_wise = TRUE
  )

  # Add subsampling if < 1.0
  if (subsample < 1.0) {
    lgb_params$bagging_fraction <- subsample
    lgb_params$bagging_freq <- 1  # Subsample every iteration
  }

  if (!is.null(seed)) {
    lgb_params$seed <- seed
    lgb_params$bagging_seed <- seed  # For reproducible subsampling
  }
  
  # Main loop
  for (cycle in seq_len(n_cycles)) {
    # === Tree step ===
    resid <- y - pred
    
    dtrain <- lightgbm::lgb.Dataset(
      data = X,
      label = resid,
      free_raw_data = FALSE
    )
    
    booster <- lightgbm::lgb.train(
      params = lgb_params,
      data = dtrain,
      nrounds = trees_per_cycle,
      verbose = -1
    )
    
    tree_pred <- predict(booster, X)
    pred <- pred + lr_tree * tree_pred
    trees[[cycle]] <- booster
    
    # === Linear step ===
    resid <- y - pred
    
    # Find feature most correlated with residual
    correlations <- sapply(seq_len(n_features), function(j) {
      cor(X[, j], resid, use = "complete.obs")
    })
    correlations[is.na(correlations)] <- 0
    
    best_j <- which.max(abs(correlations))
    
    # Fit simple linear regression
    x_j <- X[, best_j]
    x_mean <- mean(x_j)
    x_var <- var(x_j)
    
    if (x_var > 1e-10) {
      coef <- cov(x_j, resid) / x_var
      intercept <- mean(resid) - coef * x_mean
    } else {
      coef <- 0
      intercept <- mean(resid)
    }
    
    linear_pred <- coef * x_j + intercept
    pred <- pred + lr_linear * linear_pred
    
    linear_steps[[cycle]] <- list(
      feature = best_j,
      coef = coef,
      intercept = intercept
    )
    linear_feature_counts[best_j] <- linear_feature_counts[best_j] + 1
    
    if (verbose && cycle %% 5 == 0) {
      train_rmse <- sqrt(mean((y - pred)^2))
      cat(sprintf("Cycle %d/%d: RMSE=%.4f, linear feature=%s\n",
                  cycle, n_cycles, train_rmse, feature_names[best_j]))
    }
  }
  
  # Return model object
  model <- list(
    init = init_pred,
    trees = trees,
    linear_steps = linear_steps,
    n_cycles = n_cycles,
    trees_per_cycle = trees_per_cycle,
    lr_tree = lr_tree,
    lr_linear = lr_linear,
    subsample = subsample,
    n_features = n_features,
    feature_names = feature_names,
    linear_feature_counts = linear_feature_counts
  )
  
  class(model) <- "lgb_plus_A"
  return(model)
}


#' Predict method for lgb_plus_A
#'
#' @param object An lgb_plus_A model
#' @param newdata Matrix or data.frame of features
#' @param ... Additional arguments (ignored)
#'
#' @return Numeric vector of predictions
#'
#' @export
predict.lgb_plus_A <- function(object, newdata, ...) {
  .check_lightgbm()
  
  X <- as.matrix(newdata)
  n_samples <- nrow(X)
  
  pred <- rep(object$init, n_samples)
  
  for (cycle in seq_along(object$trees)) {
    booster <- object$trees[[cycle]]
    lin <- object$linear_steps[[cycle]]
    
    # Tree prediction
    pred <- pred + object$lr_tree * predict(booster, X)
    
    # Linear prediction
    j <- lin$feature
    pred <- pred + object$lr_linear * (lin$coef * X[, j] + lin$intercept)
  }
  
  return(pred)
}


#' Print method for lgb_plus_A
#'
#' @param x An lgb_plus_A model
#' @param ... Additional arguments (ignored)
#'
#' @export
print.lgb_plus_A <- function(x, ...) {
  cat("LGB^A+ Model\n")
  cat("=======================\n")
  cat(sprintf("Cycles: %d\n", x$n_cycles))
  cat(sprintf("Trees per cycle: %d\n", x$trees_per_cycle))
  cat(sprintf("Total trees: %d\n", x$n_cycles * x$trees_per_cycle))
  cat(sprintf("Learning rates: tree=%.3f, linear=%.3f\n", x$lr_tree, x$lr_linear))
  cat(sprintf("Features: %d\n", x$n_features))
  cat("\nLinear feature selection frequency:\n")
  
  sorted_counts <- sort(x$linear_feature_counts, decreasing = TRUE)
  for (i in seq_along(sorted_counts)) {
    if (sorted_counts[i] > 0) {
      cat(sprintf("  %s: %d times\n", names(sorted_counts)[i], sorted_counts[i]))
    }
  }
  
  invisible(x)
}


#' Summary method for lgb_plus_A
#'
#' @param object An lgb_plus_A model
#' @param ... Additional arguments (ignored)
#'
#' @return Summary statistics (invisibly)
#'
#' @export
summary.lgb_plus_A <- function(object, ...) {
  print(object)
  
  cat("\nLinear coefficients (cumulative, scaled by lr_linear):\n")
  
  # Aggregate linear contributions by feature
  linear_contrib <- rep(0, object$n_features)
  names(linear_contrib) <- object$feature_names
  
  for (lin in object$linear_steps) {
    linear_contrib[lin$feature] <- linear_contrib[lin$feature] + 
      object$lr_linear * lin$coef
  }
  
  sorted_contrib <- sort(abs(linear_contrib), decreasing = TRUE)
  for (i in seq_along(sorted_contrib)) {
    fname <- names(sorted_contrib)[i]
    val <- linear_contrib[fname]
    if (abs(val) > 1e-6) {
      cat(sprintf("  %s: %.4f\n", fname, val))
    }
  }
  
  invisible(list(
    linear_feature_counts = object$linear_feature_counts,
    linear_contributions = linear_contrib
  ))
}


#' Cross-validation for lgb_plus_A
#'
#' @param X Matrix or data.frame of features
#' @param y Numeric vector of target values
#' @param n_folds Number of CV folds (default: 5)
#' @param ... Additional arguments passed to lgb_plus_A
#'
#' @return List with CV results
#'
#' @export
cv_lgb_plus_A <- function(X, y, n_folds = 5, ...) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X)
  
  # Create folds
  fold_ids <- sample(rep(1:n_folds, length.out = n))
  
  rmse_folds <- numeric(n_folds)
  
  for (fold in seq_len(n_folds)) {
    test_idx <- which(fold_ids == fold)
    train_idx <- which(fold_ids != fold)
    
    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    y_test <- y[test_idx]
    
    model <- lgb_plus_A(X_train, y_train, ...)
    pred <- predict(model, X_test)
    
    rmse_folds[fold] <- sqrt(mean((y_test - pred)^2))
  }
  
  list(
    rmse_mean = mean(rmse_folds),
    rmse_sd = sd(rmse_folds),
    rmse_folds = rmse_folds
  )
}


# ============================================================
# Demo / Test
# ============================================================

#' Fit an Ensembled LGB^A+ model
#'
#' Runs multiple lgb_plus_A models with tree subsampling and averages predictions.
#' The ensemble helps reduce variance while the subsampling prevents overfitting.
#'
#' @param X Matrix or data.frame of features (n_samples x n_features)
#' @param y Numeric vector of target values
#' @param n_runs Number of ensemble members (default: 10)
#' @param subsample_rate Subsampling rate for trees (default: 0.7)
#' @param n_cycles Number of alternating cycles (default: 25)
#' @param trees_per_cycle Trees to fit per cycle (default: 10)
#' @param lr_tree Learning rate for trees (default: 0.02)
#' @param lr_linear Learning rate for linear updates (default: 0.1)
#' @param num_leaves Max leaves per tree (default: 15)
#' @param min_data Minimum samples in leaf (default: 20)
#' @param seed Random seed (default: NULL)
#' @param verbose Print progress (default: FALSE)
#'
#' @return An lgb_plus_A_ensemble model object (list)
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000 * 4), ncol = 4)
#' y <- 0.5 * X[,1] + 0.3 * tanh(X[,2]) + rnorm(1000) * 0.1
#' model <- lgb_plus_A_ensemble(X, y, n_runs = 10, subsample_rate = 0.7)
#' pred <- predict(model, X)
#' }
#'
#' @export
lgb_plus_A_ensemble <- function(
    X, y,
    n_runs = 10,
    subsample_rate = 0.7,
    n_cycles = 25,
    trees_per_cycle = 10,
    lr_tree = 0.02,
    lr_linear = 0.1,
    num_leaves = 15,
    min_data = 20,
    seed = NULL,
    verbose = FALSE
) {
  .check_lightgbm()

  # Convert inputs
  X <- as.matrix(X)
  y <- as.numeric(y)

  n_samples <- nrow(X)
  n_features <- ncol(X)

  feature_names <- colnames(X)
  if (is.null(feature_names)) {
    feature_names <- paste0("V", seq_len(n_features))
  }

  # Storage for ensemble models
  ensemble_models <- list()

  # Aggregate linear feature counts across ensemble
  linear_feature_counts_total <- rep(0, n_features)
  names(linear_feature_counts_total) <- feature_names

  # LightGBM parameters with subsampling for trees
  lgb_params <- list(
    objective = "regression",
    num_leaves = num_leaves,
    min_data_in_leaf = min_data,
    learning_rate = 1.0,  # We apply lr_tree manually
    verbosity = -1,
    force_col_wise = TRUE,
    bagging_fraction = subsample_rate,  # Subsample rate for trees
    bagging_freq = 1                     # Apply bagging every iteration
  )

  # Main ensemble loop
  for (run in seq_len(n_runs)) {
    if (!is.null(seed)) {
      run_seed <- seed + run - 1
    } else {
      run_seed <- NULL
    }

    if (verbose) {
      cat(sprintf("Fitting ensemble member %d/%d\n", run, n_runs))
    }

    # Initialize for this run
    init_pred <- mean(y)
    pred <- rep(init_pred, n_samples)

    trees <- list()
    linear_steps <- list()
    linear_feature_counts <- rep(0, n_features)
    names(linear_feature_counts) <- feature_names

    # Set seed for this run
    if (!is.null(run_seed)) {
      lgb_params$seed <- run_seed
      lgb_params$bagging_seed <- run_seed
    }

    # Cycle loop
    for (cycle in seq_len(n_cycles)) {
      # === Tree step ===
      resid <- y - pred

      dtrain <- lightgbm::lgb.Dataset(
        data = X,
        label = resid,
        free_raw_data = FALSE
      )

      booster <- lightgbm::lgb.train(
        params = lgb_params,
        data = dtrain,
        nrounds = trees_per_cycle,
        verbose = -1
      )

      tree_pred <- predict(booster, X)
      pred <- pred + lr_tree * tree_pred
      trees[[cycle]] <- booster

      # === Linear step ===
      resid <- y - pred

      # Find feature most correlated with residual
      correlations <- sapply(seq_len(n_features), function(j) {
        cor(X[, j], resid, use = "complete.obs")
      })
      correlations[is.na(correlations)] <- 0

      best_j <- which.max(abs(correlations))

      # Fit simple linear regression
      x_j <- X[, best_j]
      x_mean <- mean(x_j)
      x_var <- var(x_j)

      if (x_var > 1e-10) {
        coef <- cov(x_j, resid) / x_var
        intercept <- mean(resid) - coef * x_mean
      } else {
        coef <- 0
        intercept <- mean(resid)
      }

      linear_pred <- coef * x_j + intercept
      pred <- pred + lr_linear * linear_pred

      linear_steps[[cycle]] <- list(
        feature = best_j,
        coef = coef,
        intercept = intercept
      )
      linear_feature_counts[best_j] <- linear_feature_counts[best_j] + 1
    }

    # Store this run's model
    ensemble_models[[run]] <- list(
      init = init_pred,
      trees = trees,
      linear_steps = linear_steps,
      linear_feature_counts = linear_feature_counts
    )

    # Accumulate feature counts
    linear_feature_counts_total <- linear_feature_counts_total + linear_feature_counts
  }

  # Return ensemble model object
  model <- list(
    ensemble_models = ensemble_models,
    n_runs = n_runs,
    subsample_rate = subsample_rate,
    n_cycles = n_cycles,
    trees_per_cycle = trees_per_cycle,
    lr_tree = lr_tree,
    lr_linear = lr_linear,
    n_features = n_features,
    feature_names = feature_names,
    linear_feature_counts = linear_feature_counts_total / n_runs  # Average counts
  )

  class(model) <- "lgb_plus_A_ensemble"
  return(model)
}


#' Predict method for lgb_plus_A_ensemble
#'
#' @param object An lgb_plus_A_ensemble model
#' @param newdata Matrix or data.frame of features
#' @param ... Additional arguments (ignored)
#'
#' @return Numeric vector of predictions (averaged across ensemble)
#'
#' @export
predict.lgb_plus_A_ensemble <- function(object, newdata, ...) {
  .check_lightgbm()

  X <- as.matrix(newdata)
  n_samples <- nrow(X)

  # Accumulate predictions from all ensemble members
  pred_sum <- rep(0, n_samples)

  for (run in seq_along(object$ensemble_models)) {
    member <- object$ensemble_models[[run]]
    pred <- rep(member$init, n_samples)

    for (cycle in seq_along(member$trees)) {
      booster <- member$trees[[cycle]]
      lin <- member$linear_steps[[cycle]]

      # Tree prediction
      pred <- pred + object$lr_tree * predict(booster, X)

      # Linear prediction
      j <- lin$feature
      pred <- pred + object$lr_linear * (lin$coef * X[, j] + lin$intercept)
    }

    pred_sum <- pred_sum + pred
  }

  # Average predictions
  return(pred_sum / object$n_runs)
}


#' Print method for lgb_plus_A_ensemble
#'
#' @param x An lgb_plus_A_ensemble model
#' @param ... Additional arguments (ignored)
#'
#' @export
print.lgb_plus_A_ensemble <- function(x, ...) {
  cat("LGB^A+ Ensemble Model\n")
  cat("=================================\n")
  cat(sprintf("Ensemble members: %d\n", x$n_runs))
  cat(sprintf("Tree subsample rate: %.2f\n", x$subsample_rate))
  cat(sprintf("Cycles per member: %d\n", x$n_cycles))
  cat(sprintf("Trees per cycle: %d\n", x$trees_per_cycle))
  cat(sprintf("Total trees per member: %d\n", x$n_cycles * x$trees_per_cycle))
  cat(sprintf("Learning rates: tree=%.3f, linear=%.3f\n", x$lr_tree, x$lr_linear))
  cat(sprintf("Features: %d\n", x$n_features))
  cat("\nAvg linear feature selection frequency:\n")

  sorted_counts <- sort(x$linear_feature_counts, decreasing = TRUE)
  for (i in seq_along(sorted_counts)) {
    if (sorted_counts[i] > 0) {
      cat(sprintf("  %s: %.1f times\n", names(sorted_counts)[i], sorted_counts[i]))
    }
  }

  invisible(x)
}


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
  
  # Fit model
  model <- lgb_plus_A(X_train, y_train, verbose = TRUE)
  
  # Predict
  pred_train <- predict(model, X_train)
  pred_test <- predict(model, X_test)
  
  # Evaluate
  cat("\nResults:\n")
  cat(sprintf("Train RMSE: %.4f\n", sqrt(mean((y_train - pred_train)^2))))
  cat(sprintf("Test RMSE:  %.4f\n", sqrt(mean((y_test - pred_test)^2))))
  
  # Summary
  cat("\n")
  print(model)
  
  # Cross-validation
  cat("\nCross-validation:\n")
  cv_result <- cv_lgb_plus_A(X, y, n_folds = 5)
  cat(sprintf("CV RMSE: %.4f (+/- %.4f)\n", cv_result$rmse_mean, cv_result$rmse_sd))
}
