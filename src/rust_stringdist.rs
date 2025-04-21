use crate::custom_levenshtein_distance_with_cost_map as _weighted_lev_with_map;
use crate::OcrCostMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

// Calculates the weighted Levenshtein distance with a custom cost map from Python.
#[pyfunction]
#[pyo3(signature = (a, b, cost_map, symmetric = true, default_cost = 1.0))]
fn _weighted_levenshtein_distance(
    a: &str,
    b: &str,
    cost_map: &Bound<'_, PyDict>,
    symmetric: bool,
    default_cost: Option<f64>,
) -> PyResult<f64> {
    Ok(_weighted_lev_with_map(
        a,
        b,
        &OcrCostMap::from_py_dict(cost_map, default_cost.unwrap_or(1.0), symmetric),
    ))
}

// Calculates the weighted Levenshtein distance between a string and a list of candidates.
#[pyfunction]
#[pyo3(signature = (s, candidates, cost_map, symmetric = true, default_cost = 1.0))]
fn _batch_weighted_levenshtein_distance(
    s: &str,
    candidates: Vec<String>,
    cost_map: &Bound<'_, PyDict>,
    symmetric: bool,
    default_cost: Option<f64>,
) -> PyResult<Vec<f64>> {
    let custom_cost_map =
        OcrCostMap::from_py_dict(cost_map, default_cost.unwrap_or(1.0), symmetric);

    // Calculate distances for each candidate in parallel
    let distances: Vec<f64> = candidates
        .par_iter()
        .map(|candidate| _weighted_lev_with_map(s, candidate, &custom_cost_map))
        .collect();

    Ok(distances)
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _rust_stringdist(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_weighted_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(_batch_weighted_levenshtein_distance, m)?)?;
    Ok(())
}
