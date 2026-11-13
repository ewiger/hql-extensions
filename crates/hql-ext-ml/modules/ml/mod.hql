//! Module `ml`, provided by the `ml` package: a dataset built from records
//! or from rows of features, a model fitted over it and kept as a value,
//! and prediction. The metrics are `ml.metrics`. HQL-0057 part 2.

/// A dataset: one row of `Float` features per observation, and the target
/// per row where the dataset is labelled.
type Dataset = { features : List<List<Float>>, target? : List<Float> }

/// A labelled dataset from rows of features and a target per row. Rows of
/// unequal length are refused, naming the first that differs.
fn Dataset(features: List<List<Float>>, *, target: List<Float>) : Dataset

/// An unlabelled dataset from rows of features.
fn Dataset(features: List<List<Float>>) : Dataset

/// A labelled dataset read from values: each function in `features`
/// gives one feature of a value, in order, and `target` gives its target:
/// `houses.dataset([h => h.area, h => h.rooms], target = h => h.price)`.
fn dataset<T>(self: Collection<T>, features: List<T -> Float>, *, target: T -> Float) : Dataset

/// An unlabelled dataset read from values.
fn dataset<T>(self: Collection<T>, features: List<T -> Float>) : Dataset

/// The two parts of a split dataset.
type Split = { train : Dataset, test : Dataset }

/// `self` split into a training and a test part after a shuffle: the last
/// `test` fraction of rows is the test part. `seed` fixes the shuffle;
/// absent, the program's `random` seed is read.
fn split(self: Dataset, *, test: Float = 0.2, seed: Int) : Split

/// A fitted model. Each kind is a record whose fields are readable and
/// printable as any record's are.
type Model : {Debug}

/// A linear regression: a coefficient per feature and an intercept.
type Linear : Model { coefficients : List<Float>, intercept : Float }

/// A logistic regression: a coefficient per feature and an intercept;
/// `predict` gives a probability per row.
type Logistic : Model { coefficients : List<Float>, intercept : Float }

/// K-means clustering: a centroid per cluster.
type KMeans : Model { centroids : List<List<Float>> }

/// Ordinary least squares over a labelled dataset; an unlabelled one is
/// refused.
fn fit_linear(self: Dataset) : Linear

/// Logistic regression over a labelled dataset whose target is 0 or 1;
/// an unlabelled one is refused.
fn fit_logistic(self: Dataset, *, max_iterations: Int = 100) : Logistic

/// K-means with `clusters` centroids over the features; a target is
/// ignored. `seed` fixes the initial centroids.
fn fit_kmeans(self: Dataset, *, clusters: Int, seed: Int) : KMeans

/// The predicted target per row. A row whose length differs from the
/// fitted feature count is refused.
fn predict(self: Linear, features: List<List<Float>>) : List<Float>

/// The probability of the positive class per row.
fn predict(self: Logistic, features: List<List<Float>>) : List<Float>

/// The index of the nearest centroid per row.
fn predict(self: KMeans, features: List<List<Float>>) : List<Int>

/// `predict` over a dataset's features.
fn predict(self: Linear, dataset: Dataset) : List<Float>

/// `predict` over a dataset's features.
fn predict(self: Logistic, dataset: Dataset) : List<Float>

/// `predict` over a dataset's features.
fn predict(self: KMeans, dataset: Dataset) : List<Int>
