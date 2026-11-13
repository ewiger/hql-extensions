//! The refusals of the `ml` package. Each message is the text a program
//! sees when a call is refused.

/// Why an `ml` operation refused its input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MlError {
    /// A row of features whose length differs from the first row's.
    #[error("row {row} has {found} features; row 0 has {expected}")]
    UnequalRows {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// A target whose length differs from the number of rows.
    #[error("the dataset has {rows} rows and {targets} targets")]
    TargetLength { rows: usize, targets: usize },
    /// A fit that needs a target over an unlabelled dataset.
    #[error("`{operation}` needs a labelled dataset; this one has no target")]
    Unlabelled { operation: &'static str },
    /// A fit over a dataset with no rows.
    #[error("`{operation}` needs at least one row; the dataset is empty")]
    Empty { operation: &'static str },
    /// A logistic target holding a value other than 0 or 1.
    #[error("row {row} has target {value}; a logistic target is 0 or 1")]
    NonBinaryTarget { row: usize, value: f64 },
    /// A row to predict whose length differs from the fitted feature count.
    #[error("row {row} has {found} features; the model was fitted on {expected}")]
    FeatureCount {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// Two lists compared element by element whose lengths differ.
    #[error("{predicted} predicted values against {actual} actual values")]
    UnequalLengths { predicted: usize, actual: usize },
    /// A test fraction outside 0 to 1.
    #[error("the test fraction is {0}; it must be between 0 and 1")]
    Fraction(f64),
    /// A cluster count below 1 or above the number of rows.
    #[error("{clusters} clusters over {rows} rows; between 1 and {rows} are possible")]
    Clusters { clusters: i64, rows: usize },
    /// An iteration count below 1.
    #[error("max_iterations is {0}; it must be at least 1")]
    Iterations(i64),
    /// A failure the fitting library reports.
    #[error("{operation} failed: {reason}")]
    Fit {
        operation: &'static str,
        reason: String,
    },
}

/// The result of an `ml` operation.
pub type Result<T> = std::result::Result<T, MlError>;
