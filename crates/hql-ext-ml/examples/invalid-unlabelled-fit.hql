// status: invalid
// feature: a fit that needs a target refuses an unlabelled dataset, naming the function
// implementation: implemented
// proposal: doc/proposals/HQL-0057/parts/02-ml.hmd
// environment: pure
// phase: execute
// kind: invalid-argument
// severity: failure

import ml

Dataset([[1.0f], [2.0f]]).fit_linear()
