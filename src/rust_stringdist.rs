use crate::custom_levenshtein_distance_with_cost_map as _weighted_lev_with_map;
use crate::longest_key_string_length;
use crate::OcrCostMap;
use pyo3::exceptions::PyValueError;
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
    default_cost: f64,
) -> PyResult<f64> {
    if default_cost < 0.0 {
        return Err(PyValueError::new_err("default_cost must be non-negative"));
    }

    let ocr_cost_map = OcrCostMap::from_py_dict(cost_map, default_cost, symmetric);
    let max_token_characters = longest_key_string_length(&ocr_cost_map.costs);
    Ok(_weighted_lev_with_map(
        a,
        b,
        &ocr_cost_map,
        max_token_characters,
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
    default_cost: f64,
) -> PyResult<Vec<f64>> {
    if default_cost < 0.0 {
        return Err(PyValueError::new_err("default_cost must be non-negative"));
    }

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let ocr_cost_map = OcrCostMap::from_py_dict(cost_map, default_cost, symmetric);
    let max_token_characters = longest_key_string_length(&ocr_cost_map.costs);

    // Calculate distances for each candidate in parallel
    let distances: Vec<f64> = candidates
        .par_iter()
        .map(|candidate| _weighted_lev_with_map(s, candidate, &ocr_cost_map, max_token_characters))
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
