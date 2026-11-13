// status: proposed
// feature: a linear model fitted over a labelled dataset predicts a new row, and rmse judges it
// implementation: pending
// proposal: doc/proposals/HQL-0057/parts/02-ml.md
// environment: pure
// expected-type: List<Float>
// expected: [7.0, 0.0]

import ml
import ml.metrics

d = Dataset([[1.0], [2.0], [3.0]], target = [3.0, 5.0, 7.0])
model = d.fit_linear()
[model.predict([[3.0]]).get(0), metrics.rmse(model.predict(d), d.target)]
