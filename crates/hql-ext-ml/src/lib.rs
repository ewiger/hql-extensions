//! The `ml` package: the implementations of the external declarations of
//! `ml` and `ml.metrics`, whose contracts are under `modules/`.
//!
//! [`core`] holds the computations over plain Rust values. The extension
//! entry point, `hql_extension_v1`, is written by `hql::extension!` once
//! hql exposes `hql::ext` (HQL-0057 part 1); the natives that bind `core`
//! to the contracts come with it.

#![forbid(unsafe_code)]

pub mod core;
