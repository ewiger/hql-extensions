// status: valid-now
// feature: a logistic model gives a probability per row, which accuracy and confusion read at a threshold
// implementation: implemented
// proposal: doc/proposals/HQL-0057/parts/02-ml.hmd
// environment: pure
// expected-type: List<Float>
// expected: [1.0, 3.0, 3.0]

import ml
import ml.metrics

observed = [0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f]
d = Dataset([[0.0f], [1.0f], [2.0f], [3.0f], [4.0f], [5.0f]], target = observed)
probabilities = d.fit_logistic(max_iterations = 200).predict(d)
counts = metrics.confusion(probabilities, observed)
[metrics.accuracy(probabilities, observed), counts.true_positive.float(), counts.true_negative.float()]
