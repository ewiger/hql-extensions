//! K-means clustering.

use linfa::traits::Fit;
use linfa_clustering::KMeans as LinfaKMeans;
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::rand_core::SeedableRng;

use super::dataset::{Dataset, check_width};
use super::error::{MlError, Result};

/// Fitted k-means: a centroid per cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeans {
    pub centroids: Vec<Vec<f64>>,
}

/// K-means with `clusters` centroids over the features; a target is
/// ignored. `seed` fixes the initial centroids, so a fit is deterministic.
/// A cluster count below 1 or above the row count is refused.
pub fn fit_kmeans(dataset: &Dataset, clusters: i64, seed: u64) -> Result<KMeans> {
    const OPERATION: &str = "fit_kmeans";
    let k = usize::try_from(clusters)
        .ok()
        .filter(|&k| (1..=dataset.rows()).contains(&k))
        .ok_or(MlError::Clusters {
            clusters,
            rows: dataset.rows(),
        })?;
    let data = linfa::DatasetBase::from(dataset.matrix());
    let fitted = LinfaKMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed))
        .fit(&data)
        .map_err(|e| MlError::Fit {
            operation: OPERATION,
            reason: e.to_string(),
        })?;
    Ok(KMeans {
        centroids: fitted
            .centroids()
            .outer_iter()
            .map(|c| c.to_vec())
            .collect(),
    })
}

impl KMeans {
    /// The index of the nearest centroid per row, the lower index on a
    /// tie. A row whose length differs from the centroids' is refused.
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<usize>> {
        let width = self.centroids.first().map_or(0, Vec::len);
        check_width(rows, width)?;
        Ok(rows.iter().map(|row| self.nearest(row)).collect())
    }

    fn nearest(&self, row: &[f64]) -> usize {
        let distance =
            |c: &[f64]| -> f64 { c.iter().zip(row).map(|(a, b)| (a - b) * (a - b)).sum() };
        self.centroids
            .iter()
            .enumerate()
            .fold((0, f64::INFINITY), |(best, d), (i, c)| {
                let di = distance(c);
                if di < d { (i, di) } else { (best, d) }
            })
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_blobs() -> Result<Dataset> {
        let rows = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        Dataset::new(rows, None)
    }

    #[test]
    fn two_blobs_fall_in_two_clusters() -> Result<()> {
        let d = two_blobs()?;
        let model = fit_kmeans(&d, 2, 7)?;
        let labels = model.predict(d.features())?;
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
        Ok(())
    }

    #[test]
    fn one_seed_fits_the_same_twice() -> Result<()> {
        let d = two_blobs()?;
        assert_eq!(fit_kmeans(&d, 2, 42)?, fit_kmeans(&d, 2, 42)?);
        Ok(())
    }

    #[test]
    fn a_cluster_count_out_of_range_is_refused() -> Result<()> {
        let d = two_blobs()?;
        assert_eq!(
            fit_kmeans(&d, 0, 1),
            Err(MlError::Clusters {
                clusters: 0,
                rows: 6
            })
        );
        assert_eq!(
            fit_kmeans(&d, 7, 1),
            Err(MlError::Clusters {
                clusters: 7,
                rows: 6
            })
        );
        Ok(())
    }
}
