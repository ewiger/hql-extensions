// status: valid-now
// feature: a linear model fitted over a labelled dataset predicts a new row, and rmse judges it
// implementation: implemented
// proposal: doc/proposals/HQL-0057/parts/02-ml.hmd
// environment: pure
// expected-type: List<List<Float>>
// expected: [[9.0], [0.0]]

import ml
import ml.metrics

/// `x` to three decimal places: a fit is exact only up to rounding.
fn near(x: Float) : Float = (x * 1000.0f).round() / 1000.0f

target = [3.0f, 5.0f, 7.0f]
d = Dataset([[1.0f], [2.0f], [3.0f]], target = target)
model = d.fit_linear()
[model.predict([[4.0f]]) | map(near), [near(metrics.rmse(model.predict(d), target))]]
