//! HQL values to and from the plain Rust values `core` computes over.

use crate::core::{Dataset, KMeans, Linear, Logistic, MlError};
use hql::ext::{self, Diagnostic, EvalCx, Span, TypeRef, TypedArg, Value};
use std::sync::Arc;

/// What a native of this package returns.
pub type Result<T> = std::result::Result<T, Diagnostic>;

/// A refusal of `core`, as a failure of the call.
pub fn refused(span: &Span, error: &MlError) -> Diagnostic {
    ext::failure(span.clone(), error.to_string())
}

/// A value the declared types rule out: a defect of the package, said
/// plainly rather than panicked over.
fn unexpected(span: &Span, expected: &str) -> Diagnostic {
    ext::failure(span.clone(), format!("expected {expected}"))
}

/// The type `List<Float>`.
fn floats_type() -> TypeRef {
    TypeRef::list(TypeRef::FLOAT)
}

/// A `Float`.
pub fn float(value: &Value, span: &Span) -> Result<f64> {
    match value {
        Value::Float(number) => Ok(*number),
        _ => Err(unexpected(span, "a `Float`")),
    }
}

/// An `Int`.
pub fn int(value: &Value, span: &Span) -> Result<i64> {
    match value {
        Value::Int(number) => Ok(*number),
        _ => Err(unexpected(span, "an `Int`")),
    }
}

/// The elements of a `List` or a `Set`.
pub fn elements<'a>(value: &'a Value, span: &Span) -> Result<&'a [Value]> {
    match value {
        Value::List(held, _) | Value::Set(held, _) => Ok(held.as_slice()),
        _ => Err(unexpected(span, "a `List` or a `Set`")),
    }
}

/// A `List<Float>`.
pub fn floats(value: &Value, span: &Span) -> Result<Vec<f64>> {
    elements(value, span)?
        .iter()
        .map(|element| float(element, span))
        .collect()
}

/// A `List<List<Float>>`.
pub fn rows(value: &Value, span: &Span) -> Result<Vec<Vec<f64>>> {
    elements(value, span)?
        .iter()
        .map(|row| floats(row, span))
        .collect()
}

/// A `List<Float>` value.
pub fn float_list(numbers: &[f64]) -> Value {
    Value::list(
        numbers.iter().copied().map(Value::Float).collect(),
        TypeRef::FLOAT,
    )
}

/// A `List<List<Float>>` value.
pub fn row_list(rows: &[Vec<f64>]) -> Value {
    Value::list(
        rows.iter().map(|row| float_list(row)).collect(),
        floats_type(),
    )
}

/// A count as an `Int`.
pub fn count(number: usize, span: &Span) -> Result<Value> {
    i64::try_from(number)
        .map(Value::Int)
        .map_err(|_| ext::failure(span.clone(), "numeric overflow"))
}

/// The field of a record of this package.
fn field<'a>(value: &'a Value, name: &str, span: &Span) -> Result<&'a Value> {
    match value {
        Value::Record(record) => record
            .field(name)
            .ok_or_else(|| unexpected(span, &format!("a record with `{name}`"))),
        _ => Err(unexpected(span, "a record")),
    }
}

/// The name of the record type a value is of.
pub fn kind(value: &Value) -> &'static str {
    match value {
        Value::Record(record) => record.ty.constructor.0,
        _ => "",
    }
}

/// A `Dataset` value: its target present where the dataset is labelled.
pub fn dataset_value(dataset: &Dataset) -> Value {
    let target = match dataset.target() {
        Some(target) => Value::Present(Arc::new(float_list(target))),
        None => Value::Absent(floats_type()),
    };
    ext::record(
        ext::ty("Dataset", []),
        [
            ("features", row_list(dataset.features())),
            ("target", target),
        ],
    )
}

/// The dataset a `Dataset` value holds.
pub fn dataset(value: &Value, span: &Span) -> Result<Dataset> {
    let features = rows(field(value, "features", span)?, span)?;
    let target = match field(value, "target", span)? {
        Value::Absent(_) => None,
        Value::Present(held) => Some(floats(held, span)?),
        held => Some(floats(held, span)?),
    };
    Dataset::new(features, target).map_err(|error| refused(span, &error))
}

/// A `Linear` value.
pub fn linear_value(model: &Linear) -> Value {
    ext::record(
        ext::ty("Linear", []),
        [
            ("coefficients", float_list(&model.coefficients)),
            ("intercept", Value::Float(model.intercept)),
        ],
    )
}

/// The model a `Linear` value holds.
pub fn linear(value: &Value, span: &Span) -> Result<Linear> {
    Ok(Linear {
        coefficients: floats(field(value, "coefficients", span)?, span)?,
        intercept: float(field(value, "intercept", span)?, span)?,
    })
}

/// A `Logistic` value.
pub fn logistic_value(model: &Logistic) -> Value {
    ext::record(
        ext::ty("Logistic", []),
        [
            ("coefficients", float_list(&model.coefficients)),
            ("intercept", Value::Float(model.intercept)),
        ],
    )
}

/// The model a `Logistic` value holds.
pub fn logistic(value: &Value, span: &Span) -> Result<Logistic> {
    Ok(Logistic {
        coefficients: floats(field(value, "coefficients", span)?, span)?,
        intercept: float(field(value, "intercept", span)?, span)?,
    })
}

/// A `KMeans` value.
pub fn kmeans_value(model: &KMeans) -> Value {
    ext::record(
        ext::ty("KMeans", []),
        [("centroids", row_list(&model.centroids))],
    )
}

/// The model a `KMeans` value holds.
pub fn kmeans(value: &Value, span: &Span) -> Result<KMeans> {
    Ok(KMeans {
        centroids: rows(field(value, "centroids", span)?, span)?,
    })
}

/// The value of an argument a call may write by name or, where the
/// declaration allows it, at a position among the unnamed ones. Absent
/// where the call does not write it.
pub fn argument(
    cx: &mut dyn EvalCx,
    arguments: &[TypedArg],
    name: &str,
    position: Option<usize>,
) -> Result<Option<Value>> {
    let written = ext::named(arguments, name)
        .or_else(|| position.and_then(|at| ext::positional(arguments, at)));
    written
        .map(|argument| ext::value_of(cx, argument))
        .transpose()
}

/// The value of an argument the declaration requires.
pub fn required(
    cx: &mut dyn EvalCx,
    arguments: &[TypedArg],
    name: &str,
    position: Option<usize>,
    span: &Span,
) -> Result<Value> {
    argument(cx, arguments, name, position)?
        .ok_or_else(|| ext::failure(span.clone(), format!("`{name}` is not given")))
}
