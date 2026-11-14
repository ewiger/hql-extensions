//! The operations of module `ml.metrics`, one per name
//! `modules/ml/metrics.hql` declares.

use super::convert::{self, Result, argument, float, refused, required};
use crate::core::metrics;
use hql::ext::{self, EvalCx, Operation, Span, TypedArg, Value};

/// The two lists every metric compares: `predicted` is the call's input,
/// `actual` its first argument.
fn compared(
    cx: &mut dyn EvalCx,
    input: &Value,
    arguments: &[TypedArg],
    span: &Span,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let actual = required(cx, arguments, "actual", Some(0), span)?;
    Ok((
        convert::floats(input, span)?,
        convert::floats(&actual, span)?,
    ))
}

/// The `threshold` of a call, 0.5 where it writes none.
fn threshold(cx: &mut dyn EvalCx, arguments: &[TypedArg], span: &Span) -> Result<f64> {
    match argument(cx, arguments, "threshold", None)? {
        Some(threshold) => float(&threshold, span),
        None => Ok(0.5),
    }
}

/// `fn rmse(predicted: List<Float>, actual: List<Float>) : Float`
fn rmse(cx: &mut dyn EvalCx, input: Value, arguments: &[TypedArg], span: Span) -> Result<Value> {
    let (predicted, actual) = compared(cx, &input, arguments, &span)?;
    metrics::rmse(&predicted, &actual)
        .map(Value::Float)
        .map_err(|error| refused(&span, &error))
}

/// `fn mae(predicted: List<Float>, actual: List<Float>) : Float`
fn mae(cx: &mut dyn EvalCx, input: Value, arguments: &[TypedArg], span: Span) -> Result<Value> {
    let (predicted, actual) = compared(cx, &input, arguments, &span)?;
    metrics::mae(&predicted, &actual)
        .map(Value::Float)
        .map_err(|error| refused(&span, &error))
}

/// `fn r2(predicted: List<Float>, actual: List<Float>) : Float`
fn r2(cx: &mut dyn EvalCx, input: Value, arguments: &[TypedArg], span: Span) -> Result<Value> {
    let (predicted, actual) = compared(cx, &input, arguments, &span)?;
    metrics::r2(&predicted, &actual)
        .map(Value::Float)
        .map_err(|error| refused(&span, &error))
}

/// `fn accuracy(predicted, actual, *, threshold: Float = 0.5) : Float`
fn accuracy(
    cx: &mut dyn EvalCx,
    input: Value,
    arguments: &[TypedArg],
    span: Span,
) -> Result<Value> {
    let (predicted, actual) = compared(cx, &input, arguments, &span)?;
    let threshold = threshold(cx, arguments, &span)?;
    metrics::accuracy(&predicted, &actual, threshold)
        .map(Value::Float)
        .map_err(|error| refused(&span, &error))
}

/// `fn confusion(predicted, actual, *, threshold: Float = 0.5) : Confusion`
fn confusion(
    cx: &mut dyn EvalCx,
    input: Value,
    arguments: &[TypedArg],
    span: Span,
) -> Result<Value> {
    let (predicted, actual) = compared(cx, &input, arguments, &span)?;
    let threshold = threshold(cx, arguments, &span)?;
    let counts = metrics::confusion(&predicted, &actual, threshold)
        .map_err(|error| refused(&span, &error))?;
    Ok(ext::record(
        ext::ty("Confusion", []),
        [
            (
                "true_positive",
                convert::count(counts.true_positive, &span)?,
            ),
            (
                "false_positive",
                convert::count(counts.false_positive, &span)?,
            ),
            (
                "true_negative",
                convert::count(counts.true_negative, &span)?,
            ),
            (
                "false_negative",
                convert::count(counts.false_negative, &span)?,
            ),
        ],
    ))
}

/// The operations of module `ml.metrics`.
pub static STEPS: &[Operation] = &[
    hql::native!("rmse", rmse),
    hql::native!("mae", mae),
    hql::native!("r2", r2),
    hql::native!("accuracy", accuracy),
    hql::native!("confusion", confusion),
];
