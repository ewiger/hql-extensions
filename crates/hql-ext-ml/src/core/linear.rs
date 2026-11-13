//! Ordinary least squares.

use linfa::traits::Fit;
use linfa_linear::LinearRegression;

use super::dataset::{Dataset, check_width};
use super::error::{MlError, Result};

/// A fitted linear regression: a coefficient per feature and an intercept.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}

/// Ordinary least squares over a labelled dataset; an unlabelled or empty
/// one is refused.
pub fn fit_linear(dataset: &Dataset) -> Result<Linear> {
    const OPERATION: &str = "fit_linear";
    let target = dataset.labels(OPERATION)?;
    let data = linfa::Dataset::new(dataset.matrix(), target);
    let fitted = LinearRegression::new()
        .fit(&data)
        .map_err(|e| MlError::Fit {
            operation: OPERATION,
            reason: e.to_string(),
        })?;
    Ok(Linear {
        coefficients: fitted.params().to_vec(),
        intercept: fitted.intercept(),
    })
}

impl Linear {
    /// The predicted target per row. A row whose length differs from the
    /// coefficient count is refused.
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>> {
        check_width(rows, self.coefficients.len())?;
        Ok(rows
            .iter()
            .map(|row| self.intercept + dot(&self.coefficients, row))
            .collect())
    }
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn a_line_is_recovered_and_predicts() -> Result<()> {
        let d = Dataset::new(
            vec![vec![1.0], vec![2.0], vec![3.0]],
            Some(vec![3.0, 5.0, 7.0]),
        )?;
        let model = fit_linear(&d)?;
        assert!(close(model.coefficients[0], 2.0), "{model:?}");
        assert!(close(model.intercept, 1.0), "{model:?}");
        let predicted = model.predict(&[vec![3.0], vec![0.0]])?;
        assert!(
            close(predicted[0], 7.0) && close(predicted[1], 1.0),
            "{predicted:?}"
        );
        Ok(())
    }

    #[test]
    fn two_features() -> Result<()> {
        // y = 1 + 2a - b
        let rows = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 3.0],
        ];
        let target = rows.iter().map(|r| 1.0 + 2.0 * r[0] - r[1]).collect();
        let model = fit_linear(&Dataset::new(rows, Some(target))?)?;
        assert!(close(model.coefficients[0], 2.0) && close(model.coefficients[1], -1.0));
        assert!(close(model.intercept, 1.0));
        Ok(())
    }

    #[test]
    fn an_unlabelled_dataset_is_refused() -> Result<()> {
        let d = Dataset::new(vec![vec![1.0]], None)?;
        assert_eq!(
            fit_linear(&d),
            Err(MlError::Unlabelled {
                operation: "fit_linear"
            })
        );
        Ok(())
    }

    #[test]
    fn a_row_of_another_width_is_refused() {
        let model = Linear {
            coefficients: vec![1.0, 2.0],
            intercept: 0.0,
        };
        assert_eq!(
            model.predict(&[vec![1.0, 2.0], vec![1.0]]),
            Err(MlError::FeatureCount {
                row: 1,
                expected: 2,
                found: 1
            })
        );
    }
}
