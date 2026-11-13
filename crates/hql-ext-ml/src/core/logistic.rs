//! Logistic regression over a target of 0 and 1.

use linfa::traits::Fit;
use linfa_logistic::LogisticRegression;

use super::dataset::{Dataset, check_width};
use super::error::{MlError, Result};
use super::linear::dot;

/// A fitted logistic regression: a coefficient per feature and an
/// intercept, giving the probability of the class 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Logistic {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}

/// Logistic regression over a labelled dataset whose target is 0 or 1.
/// An unlabelled or empty dataset, a target value other than 0 or 1, and
/// `max_iterations` below 1 are refused.
pub fn fit_logistic(dataset: &Dataset, max_iterations: i64) -> Result<Logistic> {
    const OPERATION: &str = "fit_logistic";
    let iterations = u64::try_from(max_iterations)
        .ok()
        .filter(|&n| n > 0)
        .ok_or(MlError::Iterations(max_iterations))?;
    let target = dataset.labels(OPERATION)?;
    let classes = target
        .iter()
        .enumerate()
        .map(|(row, &value)| match value {
            0.0 => Ok(0_u8),
            1.0 => Ok(1_u8),
            _ => Err(MlError::NonBinaryTarget { row, value }),
        })
        .collect::<Result<ndarray::Array1<u8>>>()?;
    let data = linfa::Dataset::new(dataset.matrix(), classes);
    let fitted = LogisticRegression::default()
        .max_iterations(iterations)
        .fit(&data)
        .map_err(|e| MlError::Fit {
            operation: OPERATION,
            reason: e.to_string(),
        })?;
    // linfa models the probability of its positive class, the more
    // frequent one; the negated parameters model the other, since
    // sigmoid(-z) = 1 - sigmoid(z). Kept as the probability of 1.
    let sign = if fitted.labels().pos.class == 1 {
        1.0
    } else {
        -1.0
    };
    Ok(Logistic {
        coefficients: fitted.params().iter().map(|c| sign * c).collect(),
        intercept: sign * fitted.intercept(),
    })
}

impl Logistic {
    /// The probability of the class 1 per row. A row whose length differs
    /// from the coefficient count is refused.
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>> {
        check_width(rows, self.coefficients.len())?;
        Ok(rows
            .iter()
            .map(|row| sigmoid(self.intercept + dot(&self.coefficients, row)))
            .collect())
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn separable(ones_first: bool) -> Result<Dataset> {
        let rows: Vec<Vec<f64>> = [-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| vec![x])
            .collect();
        let target = rows
            .iter()
            .map(|r| if (r[0] > 0.0) == ones_first { 1.0 } else { 0.0 })
            .collect();
        Dataset::new(rows, Some(target))
    }

    #[test]
    fn probabilities_fall_on_the_right_side() -> Result<()> {
        // Five rows of class 1, four of class 0: 1 is linfa's positive class.
        let model = fit_logistic(&separable(true)?, 100)?;
        let p = model.predict(&[vec![-3.0], vec![3.0]])?;
        assert!(p[0] < 0.5 && p[1] > 0.5, "{p:?}");
        Ok(())
    }

    #[test]
    fn the_probability_is_of_class_one_when_zero_is_more_frequent() -> Result<()> {
        // Five rows of class 0: linfa's positive class is 0, and the model
        // is flipped so it still gives the probability of 1.
        let model = fit_logistic(&separable(false)?, 100)?;
        let p = model.predict(&[vec![-3.0], vec![3.0]])?;
        assert!(p[0] > 0.5 && p[1] < 0.5, "{p:?}");
        Ok(())
    }

    #[test]
    fn a_target_other_than_zero_or_one_is_refused() -> Result<()> {
        let d = Dataset::new(vec![vec![1.0], vec![2.0]], Some(vec![0.0, 2.0]))?;
        assert_eq!(
            fit_logistic(&d, 100),
            Err(MlError::NonBinaryTarget { row: 1, value: 2.0 })
        );
        Ok(())
    }

    #[test]
    fn an_unlabelled_dataset_is_refused() -> Result<()> {
        let d = Dataset::new(vec![vec![1.0]], None)?;
        assert_eq!(
            fit_logistic(&d, 100),
            Err(MlError::Unlabelled {
                operation: "fit_logistic"
            })
        );
        Ok(())
    }

    #[test]
    fn zero_iterations_are_refused() -> Result<()> {
        assert_eq!(
            fit_logistic(&separable(true)?, 0),
            Err(MlError::Iterations(0))
        );
        Ok(())
    }
}
