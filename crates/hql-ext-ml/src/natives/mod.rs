//! The natives: what binds `core` to the contracts under `modules/`. Each
//! converts the HQL values of a call to plain Rust values, runs the
//! computation, and converts the result back.

mod convert;
pub mod metrics;
pub mod ml;
