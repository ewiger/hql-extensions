# hql-ext-ml

The `ml` package for [HQL](https://github.com/ewiger/hql): datasets, fitted
models as values, prediction and metrics, over
[linfa](https://github.com/rust-ml/linfa).

```sh
hql package install ml          # into the project, .hql/packages/ml/
hql package install --home ml   # into the HQL home, for every run
```

```hql
import ml
import ml.metrics

target = [3.0f, 5.0f, 7.0f]
d = Dataset([[1.0f], [2.0f], [3.0f]], target = target)
model = d.fit_linear()
metrics.rmse(model.predict(d), target)
```

| Module | Provides |
| --- | --- |
| `ml` | `Dataset`, `dataset`, `split`, `fit_linear`, `fit_logistic`, `fit_kmeans`, `predict` |
| `ml.metrics` | `rmse`, `mae`, `r2`, `accuracy`, `confusion` |

A model is a record: `Linear` and `Logistic` hold `coefficients` and an
`intercept`, `KMeans` holds `centroids`. Every number is a `Float`, written
with the suffix `f`.

## What this crate is

A crate named `hql-ext-<package>` whose library is an HQL extension. `hql`
does not link it: `hql package install` builds it on your machine against the
`hql` you run and loads the library when a program imports `ml`.

- `modules/` holds the contracts: each function declared in HQL without a
  body, with its doc string.
- `src/core/` holds the computations over plain Rust values.
- `src/natives/` binds the two: one Rust function per declared name, which
  converts the HQL values of a call, runs the computation, and converts the
  result.
- `src/lib.rs` states the extension with `hql::extension!`.
- `examples/` holds checked HQL programs, and is a project: its
  `.hql/config.toml` loads the package from this checkout.

The crate is on crates.io so that `hql package` and `cargo` can fetch a
pinned version of it. Its Rust API is the package's implementation, not a
library to depend on.
