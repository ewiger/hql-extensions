// status: valid-now
// feature: k-means over an unlabelled dataset gives the index of the nearest centroid per row
// implementation: implemented
// proposal: doc/proposals/HQL-0057/parts/02-ml.hmd
// environment: pure
// expected-type: List<Bool>
// expected: [true, true, true]

import ml

d = Dataset([[1.0f, 1.0f], [1.2f, 0.8f], [8.0f, 8.0f], [8.2f, 7.9f]])
model = d.fit_kmeans(clusters = 2, seed = 1)
nearest = model.predict(d)
// Which cluster is numbered 0 follows the seed; which rows share one does not.
[nearest[0] == nearest[1], nearest[2] == nearest[3], nearest[0] != nearest[2]]
