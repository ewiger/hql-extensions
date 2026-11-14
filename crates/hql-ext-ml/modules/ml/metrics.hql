// The contract of module `ml.metrics`: every function is declared without
// a body, and the `ml` package's extension implements it. HQL-0057 part 2.

/// How far predictions are from what was observed. Each function refuses
/// lists of unequal length. Imported by name: `import ml.metrics`.
module ml.metrics

import core

/// The root of the mean squared difference.
fn rmse(predicted: List<Float>, actual: List<Float>) : Float

/// The mean absolute difference.
fn mae(predicted: List<Float>, actual: List<Float>) : Float

/// The coefficient of determination: 1 for a perfect fit, 0 for the mean,
/// below 0 for worse than the mean.
fn r2(predicted: List<Float>, actual: List<Float>) : Float

/// The fraction of rows classified as observed, a value above `threshold`
/// read as the positive class on both sides.
fn accuracy(predicted: List<Float>, actual: List<Float>, *, threshold: Float = 0.5) : Float

/// The counts of a classification against what was observed.
type Confusion = { true_positive : Int, false_positive : Int, true_negative : Int, false_negative : Int }

/// The counts of true and false positives and negatives at `threshold`.
fn confusion(predicted: List<Float>, actual: List<Float>, *, threshold: Float = 0.5) : Confusion
