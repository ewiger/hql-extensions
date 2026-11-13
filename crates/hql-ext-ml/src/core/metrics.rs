//! How far predictions are from what was observed. Each function refuses
//! lists of unequal length.

use super::error::{MlError, Result};

/// The counts of a classification against what was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Confusion {
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
}

fn pairs<'a>(
    predicted: &'a [f64],
    actual: &'a [f64],
) -> Result<impl Iterator<Item = (f64, f64)> + 'a> {
    if predicted.len() != actual.len() {
        return Err(MlError::UnequalLengths {
            predicted: predicted.len(),
            actual: actual.len(),
        });
    }
    Ok(predicted.iter().copied().zip(actual.iter().copied()))
}

#[allow(clippy::cast_precision_loss)]
fn mean(sum: f64, n: usize) -> f64 {
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// The root of the mean squared difference; NaN over empty lists.
pub fn rmse(predicted: &[f64], actual: &[f64]) -> Result<f64> {
    let sum = pairs(predicted, actual)?
        .map(|(p, a)| (p - a) * (p - a))
        .sum();
    Ok(mean(sum, actual.len()).sqrt())
}

/// The mean absolute difference; NaN over empty lists.
pub fn mae(predicted: &[f64], actual: &[f64]) -> Result<f64> {
    let sum = pairs(predicted, actual)?.map(|(p, a)| (p - a).abs()).sum();
    Ok(mean(sum, actual.len()))
}

/// The coefficient of determination: 1 for a perfect fit, 0 for the mean,
/// below 0 for worse than the mean. NaN where the actual values do not vary.
pub fn r2(predicted: &[f64], actual: &[f64]) -> Result<f64> {
    let residual: f64 = pairs(predicted, actual)?
        .map(|(p, a)| (p - a) * (p - a))
        .sum();
    let centre = mean(actual.iter().sum(), actual.len());
    let total: f64 = actual.iter().map(|a| (a - centre) * (a - centre)).sum();
    Ok(if total == 0.0 {
        f64::NAN
    } else {
        1.0 - residual / total
    })
}

/// The counts at `threshold`: a value above it is the positive class, on
/// both sides.
pub fn confusion(predicted: &[f64], actual: &[f64], threshold: f64) -> Result<Confusion> {
    let mut counts = Confusion::default();
    for (p, a) in pairs(predicted, actual)? {
        match (p > threshold, a > threshold) {
            (true, true) => counts.true_positive += 1,
            (true, false) => counts.false_positive += 1,
            (false, false) => counts.true_negative += 1,
            (false, true) => counts.false_negative += 1,
        }
    }
    Ok(counts)
}

/// The fraction of rows classified as observed at `threshold`; NaN over
/// empty lists.
pub fn accuracy(predicted: &[f64], actual: &[f64], threshold: f64) -> Result<f64> {
    let c = confusion(predicted, actual, threshold)?;
    #[allow(clippy::cast_precision_loss)]
    let right = (c.true_positive + c.true_negative) as f64;
    Ok(mean(right, actual.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_of_a_known_residual() -> Result<()> {
        let (p, a) = ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0]);
        assert_eq!(rmse(&p, &a)?, 1.0);
        assert_eq!(mae(&p, &a)?, 0.5);
        Ok(())
    }

    #[test]
    fn r2_is_one_for_a_perfect_fit_and_zero_for_the_mean() -> Result<()> {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(r2(&a, &a)?, 1.0);
        assert_eq!(r2(&[2.0, 2.0, 2.0], &a)?, 0.0);
        Ok(())
    }

    #[test]
    fn confusion_and_accuracy_at_a_threshold() -> Result<()> {
        let p = [0.9, 0.8, 0.2, 0.1, 0.6];
        let a = [1.0, 0.0, 0.0, 1.0, 1.0];
        let c = confusion(&p, &a, 0.5)?;
        assert_eq!(
            c,
            Confusion {
                true_positive: 2,
                false_positive: 1,
                true_negative: 1,
                false_negative: 1
            }
        );
        assert_eq!(accuracy(&p, &a, 0.5)?, 0.6);
        Ok(())
    }

    #[test]
    fn unequal_lengths_are_refused() {
        let refused = Err(MlError::UnequalLengths {
            predicted: 2,
            actual: 1,
        });
        assert_eq!(rmse(&[1.0, 2.0], &[1.0]), refused);
        assert_eq!(accuracy(&[1.0, 2.0], &[1.0], 0.5), refused);
    }
}
