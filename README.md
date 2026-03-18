# lgbplus

R and Python implementations of **LGB+** and **LGB<sup>A</sup>+**, hybrid boosting algorithms that interleave tree-based and linear updates within a single boosting loop.

**Paper**: Philippe Goulet Coulombe, "LGB+: A Macroeconomic Forecasting Road Test"

## Files

### R

| File | Algorithm | Description |
|------|-----------|-------------|
| `R/lgb_plus.R` | **LGB+** | Per-step competition: tree vs. linear candidate, winner chosen via OOB evaluation |
| `R/lgb_plus_A.R` | **LGB<sup>A</sup>+** | Fixed alternating schedule: block of tree updates, then one linear correction, repeat |

### Python

| File | Algorithm | Description |
|------|-----------|-------------|
| `python/lgb_plus.py` | **LGB+** | Competition variant with optional ensembling |
| `python/lgb_plus_A.py` | **LGB<sup>A</sup>+** | Alternating variant |

## Dependencies

- **R**: `lightgbm`
- **Python**: `numpy`, `lightgbm`

## Quick start (R)

```r
source("R/lgb_plus_A.R")

model <- lgb_plus_A(X_train, y_train,
    n_cycles = 25, trees_per_cycle = 10,
    lr_tree = 0.02, lr_linear = 0.1,
    num_leaves = 5, min_data = 15)

preds <- predict(model, X_test)
```

```r
source("R/lgb_plus.R")

model <- lgb_plus(X_train, y_train,
    n_steps = 600, lr = 0.02,
    num_leaves = 5, min_data = 15,
    subsample = 0.7, selection_method = "oob")

preds <- predict(model, X_test)
```

## Quick start (Python)

```python
from lgb_plus_A import LGBPlusA

model = LGBPlusA(n_cycles=25, trees_per_cycle=10,
    lr_tree=0.02, lr_linear=0.1,
    num_leaves=5, min_data=15)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

```python
from lgb_plus import LGBPlus

model = LGBPlus(n_steps=600, lr=0.02,
    num_leaves=5, min_data=15,
    subsample=0.7, selection_method="oob")
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## License

MIT
