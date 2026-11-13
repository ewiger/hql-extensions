//! A dataset: rows of `Float` features, and a target per row where it is
//! labelled.

use ndarray::{Array1, Array2};

use super::error::{MlError, Result};

/// Rows of features, every row of one length, and an optional target per
/// row.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    features: Vec<Vec<f64>>,
    target: Option<Vec<f64>>,
}

impl Dataset {
    /// A dataset from rows of features and, where labelled, a target per
    /// row. Rows of unequal length are refused, naming the first that
    /// differs from row 0, as is a target whose length is not the row count.
    pub fn new(features: Vec<Vec<f64>>, target: Option<Vec<f64>>) -> Result<Self> {
        check_rows(&features)?;
        if let Some(target) = &target
            && target.len() != features.len()
        {
            return Err(MlError::TargetLength {
                rows: features.len(),
                targets: target.len(),
            });
        }
        Ok(Self { features, target })
    }

    /// The rows of features.
    pub fn features(&self) -> &[Vec<f64>] {
        &self.features
    }

    /// The target per row, if the dataset is labelled.
    pub fn target(&self) -> Option<&[f64]> {
        self.target.as_deref()
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.features.len()
    }

    /// The number of features per row; 0 for an empty dataset.
    pub fn width(&self) -> usize {
        self.features.first().map_or(0, Vec::len)
    }

    /// The dataset split in two: rows taken in `order`, the last `test`
    /// fraction of them (rounded up) forming the test part. `order` is a
    /// permutation of the row indices, which the caller draws so the
    /// shuffle follows the program's seed.
    ///
    /// # Panics
    ///
    /// When `order` is not a permutation of `0..rows()`; the caller makes it.
    pub fn split(&self, test: f64, order: &[usize]) -> Result<(Dataset, Dataset)> {
        if !(0.0..=1.0).contains(&test) {
            return Err(MlError::Fraction(test));
        }
        assert_eq!(order.len(), self.rows(), "order must permute the rows");
        // A fraction in 0..=1 of a row count is a non-negative count no
        // larger than the row count.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let test_rows = ((self.rows() as f64) * test).ceil() as usize;
        let (train, held) = order.split_at(self.rows() - test_rows);
        Ok((self.select(train), self.select(held)))
    }

    fn select(&self, rows: &[usize]) -> Dataset {
        Dataset {
            features: rows.iter().map(|&i| self.features[i].clone()).collect(),
            target: self
                .target
                .as_ref()
                .map(|target| rows.iter().map(|&i| target[i]).collect()),
        }
    }

    /// The features as a matrix, one row per observation.
    pub(crate) fn matrix(&self) -> Array2<f64> {
        matrix(&self.features, self.width())
    }

    /// The target as a vector, or the refusal naming `operation`.
    pub(crate) fn labels(&self, operation: &'static str) -> Result<Array1<f64>> {
        let target = self
            .target
            .as_ref()
            .ok_or(MlError::Unlabelled { operation })?;
        if target.is_empty() {
            return Err(MlError::Empty { operation });
        }
        Ok(Array1::from(target.clone()))
    }
}

/// Refuses rows of unequal length, naming the first that differs from row 0.
fn check_rows(rows: &[Vec<f64>]) -> Result<()> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    match rows.iter().position(|row| row.len() != first.len()) {
        Some(row) => Err(MlError::UnequalRows {
            row,
            expected: first.len(),
            found: rows[row].len(),
        }),
        None => Ok(()),
    }
}

/// Refuses a row whose length is not the fitted feature count.
pub(crate) fn check_width(rows: &[Vec<f64>], expected: usize) -> Result<()> {
    match rows.iter().position(|row| row.len() != expected) {
        Some(row) => Err(MlError::FeatureCount {
            row,
            expected,
            found: rows[row].len(),
        }),
        None => Ok(()),
    }
}

/// Rows already checked to have `width` features each, as a matrix.
pub(crate) fn matrix(rows: &[Vec<f64>], width: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), width), |(i, j)| rows[i][j])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unequal_rows_name_the_first_that_differs() {
        let refused = Dataset::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0]], None);
        assert_eq!(
            refused,
            Err(MlError::UnequalRows {
                row: 2,
                expected: 2,
                found: 1
            })
        );
    }

    #[test]
    fn a_target_of_another_length_is_refused() {
        let refused = Dataset::new(vec![vec![1.0], vec![2.0]], Some(vec![1.0]));
        assert_eq!(
            refused,
            Err(MlError::TargetLength {
                rows: 2,
                targets: 1
            })
        );
    }

    #[test]
    fn split_takes_the_last_fraction_in_order() -> Result<()> {
        let d = Dataset::new(
            (0..5).map(|i| vec![f64::from(i)]).collect(),
            Some((0..5).map(f64::from).collect()),
        )?;
        let (train, test) = d.split(0.4, &[4, 3, 2, 1, 0])?;
        assert_eq!(train.features(), &[vec![4.0], vec![3.0], vec![2.0]]);
        assert_eq!(test.features(), &[vec![1.0], vec![0.0]]);
        assert_eq!(test.target(), Some(&[1.0, 0.0][..]));
        Ok(())
    }

    #[test]
    fn split_rounds_the_test_part_up() -> Result<()> {
        let d = Dataset::new((0..3).map(|i| vec![f64::from(i)]).collect(), None)?;
        let (train, test) = d.split(0.2, &[0, 1, 2])?;
        assert_eq!((train.rows(), test.rows()), (2, 1));
        Ok(())
    }

    #[test]
    fn a_fraction_outside_zero_to_one_is_refused() -> Result<()> {
        let d = Dataset::new(vec![vec![1.0]], None)?;
        assert_eq!(d.split(1.5, &[0]), Err(MlError::Fraction(1.5)));
        Ok(())
    }
}
