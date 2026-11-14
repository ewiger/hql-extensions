//! The operations of module `ml`, one per name `modules/ml/mod.hql`
//! declares. A name declared in several forms is one operation, which reads
//! which form the call took from what it was given.

use super::convert::{
    self, Result, argument, dataset, dataset_value, float, int, refused, required,
};
use crate::core::{self, Dataset};
use hql::ext::{self, EvalCx, Operation, Span, TypedArg, Value};
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::rand_core::{RngCore, SeedableRng};

/// `fn Dataset(features, *, target)` and `fn Dataset(features)`.
fn construct(
    cx: &mut dyn EvalCx,
    input: Value,
    arguments: &[TypedArg],
    span: Span,
) -> Result<Value> {
    let features = convert::rows(&input, &span)?;
    let target = argument(cx, arguments, "target", None)?
        .map(|target| convert::floats(&target, &span))
        .transpose()?;
    let built = Dataset::new(features, target).map_err(|error| refused(&span, &error))?;
    Ok(dataset_value(&built))
}

/// `fn dataset<T>(self: Collection<T>, features: List<T -> Float>, *, target:
/// T -> Float)` and the form without `target`.
fn read(cx: &mut dyn EvalCx, input: Value, arguments: &[TypedArg], span: Span) -> Result<Value> {
    let values = convert::elements(&input, &span)?;
    let readers = required(cx, arguments, "features", Some(0), &span)?;
    let readers = convert::elements(&readers, &span)?;
    let mut features = Vec::with_capacity(values.len());
    for value in values {
        let mut row = Vec::with_capacity(readers.len());
        for reader in readers {
            // A function of the program: the running hql applies it.
            let feature = cx.call(reader, vec![value.clone()], &span)?;
            row.push(float(&feature, &span)?);
        }
        features.push(row);
    }
    let target = match ext::named(arguments, "target") {
        None => None,
        Some(reader) => {
            let mut target = Vec::with_capacity(values.len());
            for value in values {
                target.push(float(&cx.apply_lambda(reader, value.clone())?, &span)?);
            }
            Some(target)
        }
    };
    let built = Dataset::new(features, target).map_err(|error| refused(&span, &error))?;
    Ok(dataset_value(&built))
}

/// A shuffle of `0..rows` that a seed fixes.
fn shuffled(rows: usize, seed: u64) -> Vec<usize> {
    let mut generator = Xoshiro256Plus::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..rows).collect();
    for last in (1..rows).rev() {
        // A row count fits in 64 bits, and the remainder is below it.
        #[allow(clippy::cast_possible_truncation)]
        let chosen = (generator.next_u64() % (last as u64 + 1)) as usize;
        order.swap(last, chosen);
    }
    order
}

/// A seed argument as the generator takes it: the bits of the `Int`.
fn seed_bits(seed: i64) -> u64 {
    u64::from_ne_bytes(seed.to_ne_bytes())
}

/// `fn split(self: Dataset, *, test: Float = 0.2, seed: Int) : Split` and
/// the effectful form without `seed`.
fn split(cx: &mut dyn EvalCx, input: Value, arguments: &[TypedArg], span: Span) -> Result<Value> {
    let whole = dataset(&input, &span)?;
    let test = match argument(cx, arguments, "test", None)? {
        Some(test) => float(&test, &span)?,
        None => 0.2,
    };
    // Without a seed of its own the shuffle follows the program's.
    let seed = match argument(cx, arguments, "seed", None)? {
        Some(seed) => seed_bits(int(&seed, &span)?),
        None => cx.draw(&span),
    };
    let (train, held) = whole
        .split(test, &shuffled(whole.rows(), seed))
        .map_err(|error| refused(&span, &error))?;
    Ok(ext::record(
        ext::ty("Split", []),
        [
            ("train", dataset_value(&train)),
            ("test", dataset_value(&held)),
        ],
    ))
}

/// `fn fit_linear(self: Dataset) : Linear`
fn fit_linear(
    _cx: &mut dyn EvalCx,
    input: Value,
    _arguments: &[TypedArg],
    span: Span,
) -> Result<Value> {
    let model =
        core::fit_linear(&dataset(&input, &span)?).map_err(|error| refused(&span, &error))?;
    Ok(convert::linear_value(&model))
}

/// `fn fit_logistic(self: Dataset, *, max_iterations: Int = 100) : Logistic`
fn fit_logistic(
    cx: &mut dyn EvalCx,
    input: Value,
    arguments: &[TypedArg],
    span: Span,
) -> Result<Value> {
    let iterations = match argument(cx, arguments, "max_iterations", None)? {
        Some(iterations) => int(&iterations, &span)?,
        None => 100,
    };
    let model = core::fit_logistic(&dataset(&input, &span)?, iterations)
        .map_err(|error| refused(&span, &error))?;
    Ok(convert::logistic_value(&model))
}

/// `fn fit_kmeans(self: Dataset, *, clusters: Int, seed: Int) : KMeans`
fn fit_kmeans(
    cx: &mut dyn EvalCx,
    input: Value,
    arguments: &[TypedArg],
    span: Span,
) -> Result<Value> {
    let clusters = int(&required(cx, arguments, "clusters", None, &span)?, &span)?;
    let seed = seed_bits(int(&required(cx, arguments, "seed", None, &span)?, &span)?);
    let model = core::fit_kmeans(&dataset(&input, &span)?, clusters, seed)
        .map_err(|error| refused(&span, &error))?;
    Ok(convert::kmeans_value(&model))
}

/// The six forms of `predict`: a model of one of three kinds, over rows of
/// features or over a dataset's.
fn predict(cx: &mut dyn EvalCx, input: Value, arguments: &[TypedArg], span: Span) -> Result<Value> {
    // The second parameter is `features` in three forms and `dataset` in
    // three: a call writes either name, or neither.
    let given = match argument(cx, arguments, "features", Some(0))? {
        Some(given) => given,
        None => required(cx, arguments, "dataset", None, &span)?,
    };
    let features = if convert::kind(&given) == "Dataset" {
        dataset(&given, &span)?.features().to_vec()
    } else {
        convert::rows(&given, &span)?
    };
    match convert::kind(&input) {
        "Linear" => convert::linear(&input, &span)?
            .predict(&features)
            .map(|predicted| convert::float_list(&predicted)),
        "Logistic" => convert::logistic(&input, &span)?
            .predict(&features)
            .map(|predicted| convert::float_list(&predicted)),
        "KMeans" => {
            let nearest = convert::kmeans(&input, &span)?
                .predict(&features)
                .map_err(|error| refused(&span, &error))?;
            let indices = nearest
                .into_iter()
                .map(|index| convert::count(index, &span))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Value::list(indices, hql::ext::TypeRef::INT));
        }
        _ => return Err(ext::failure(span, "expected a fitted model")),
    }
    .map_err(|error| refused(&span, &error))
}

/// The operations of module `ml`.
pub static STEPS: &[Operation] = &[
    hql::native!("Dataset", construct),
    hql::native!("dataset", read),
    // One form draws from the evaluation seed, so the name is effectful.
    hql::native!("split", split).effectful(),
    hql::native!("fit_linear", fit_linear),
    hql::native!("fit_logistic", fit_logistic),
    hql::native!("fit_kmeans", fit_kmeans),
    hql::native!("predict", predict),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shuffle_is_a_permutation_a_seed_fixes() {
        let order = shuffled(50, 7);
        assert_eq!(order, shuffled(50, 7));
        assert_ne!(order, shuffled(50, 8));
        let mut sorted = order;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert!(shuffled(0, 1).is_empty());
    }
}
