// status: valid-now
// feature: a dataset is read from records by one function per feature, and split by a seed
// implementation: implemented
// proposal: doc/proposals/HQL-0057/parts/02-ml.hmd
// environment: pure
// expected-type: List<Int>
// expected: [4, 2, 3, 1]

import ml

type House { area : Float, rooms : Float, price : Float }

houses = [
    House(area = 50.0f, rooms = 2.0f, price = 200.0f),
    House(area = 80.0f, rooms = 3.0f, price = 310.0f),
    House(area = 120.0f, rooms = 4.0f, price = 450.0f),
    House(area = 65.0f, rooms = 2.0f, price = 240.0f),
]
d = houses.dataset([(h: House) => h.area, (h: House) => h.rooms], target = h => h.price)
parts = d.split(test = 0.25f, seed = 7)
[d.features.count(), d.features[0].count(), parts.train.features.count(), parts.test.features.count()]
