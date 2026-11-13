//! The computations of `ml` and `ml.metrics` over plain Rust values,
//! independent of HQL's runtime: datasets, the three model kinds, and the
//! metrics. The natives that bind them to the contracts under `modules/`
//! convert HQL values to and from these.

mod dataset;
mod error;
mod kmeans;
mod linear;
mod logistic;
pub mod metrics;

pub use dataset::Dataset;
pub use error::{MlError, Result};
pub use kmeans::{KMeans, fit_kmeans};
pub use linear::{Linear, fit_linear};
pub use logistic::{Logistic, fit_logistic};
