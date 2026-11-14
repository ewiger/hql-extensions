//! The `ml` package: the implementations of the external declarations of
//! `ml` and `ml.metrics`, whose contracts are under `modules/`.
//!
//! [`core`] holds the computations over plain Rust values, and [`natives`]
//! binds them to the contracts: one operation per declared name. The
//! extension below states the two modules and is what `hql` loads, through
//! the `hql_extension_v1` symbol `hql::extension!` exports.

#![deny(unsafe_code)]

pub mod core;
pub mod natives;

hql::extension!(hql::ext::extension(
    "ml",
    env!("CARGO_PKG_VERSION"),
    &[
        hql::ext::provided("ml", natives::ml::STEPS),
        hql::ext::provided("ml.metrics", natives::metrics::STEPS),
    ],
));
