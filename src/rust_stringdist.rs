use crate::cost_map::CostMap;
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::custom_levenshtein_distance_with_cost_maps as _weighted_lev_with_maps;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Validates that the default cost is non-negative
fn validate_default_cost(default_cost: f64) -> PyResult<()> {
    if default_cost < 0.0 {
        return Err(PyValueError::new_err(format!(
            "Default cost must be non-negative, got value: {default_cost}"
        )));
    }
    Ok(())
}

// Calculates the weighted Levenshtein distance with a custom cost map from Python.
#[pyfunction]
#[pyo3(signature = (
    a,
    b,
    substitution_costs,
    insertion_costs,
    deletion_costs,
    symmetric_substitution = true,
    default_substitution_cost = 1.0,
    default_insertion_cost = 1.0,
    default_deletion_cost = 1.0,
))]
fn _weighted_levenshtein_distance(
    a: &str,
    b: &str,
    substitution_costs: &Bound<'_, PyDict>,
    insertion_costs: &Bound<'_, PyDict>,
    deletion_costs: &Bound<'_, PyDict>,
    symmetric_substitution: bool,
    default_substitution_cost: f64,
    default_insertion_cost: f64,
    default_deletion_cost: f64,
) -> PyResult<f64> {
    validate_default_cost(default_substitution_cost)?;
    validate_default_cost(default_insertion_cost)?;
    validate_default_cost(default_deletion_cost)?;

    let substitution_cost_map = CostMap::<SubstitutionKey>::from_py_dict(
        substitution_costs,
        default_substitution_cost,
        symmetric_substitution,
    );

    let insertion_cost_map =
        CostMap::<SingleTokenKey>::from_py_dict(insertion_costs, default_insertion_cost);

    let deletion_cost_map =
        CostMap::<SingleTokenKey>::from_py_dict(deletion_costs, default_deletion_cost);

    Ok(_weighted_lev_with_maps(
        a,
        b,
        &substitution_cost_map,
        &insertion_cost_map,
        &deletion_cost_map,
    ))
}

// Calculates the weighted Levenshtein distance between a string and a list of candidates.
#[pyfunction]
#[pyo3(signature = (
    s,
    candidates,
    substitution_costs,
    insertion_costs,
    deletion_costs,
    symmetric_substitution = true,
    default_substitution_cost = 1.0,
    default_insertion_cost = 1.0,
    default_deletion_cost = 1.0,
))]
fn _batch_weighted_levenshtein_distance(
    s: &str,
    candidates: Vec<String>,
    substitution_costs: &Bound<'_, PyDict>,
    insertion_costs: &Bound<'_, PyDict>,
    deletion_costs: &Bound<'_, PyDict>,
    symmetric_substitution: bool,
    default_substitution_cost: f64,
    default_insertion_cost: f64,
    default_deletion_cost: f64,
) -> PyResult<Vec<f64>> {
    validate_default_cost(default_substitution_cost)?;
    validate_default_cost(default_insertion_cost)?;
    validate_default_cost(default_deletion_cost)?;

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let substitution_cost_map = CostMap::<SubstitutionKey>::from_py_dict(
        substitution_costs,
        default_substitution_cost,
        symmetric_substitution,
    );

    let insertion_cost_map =
        CostMap::<SingleTokenKey>::from_py_dict(insertion_costs, default_insertion_cost);

    let deletion_cost_map =
        CostMap::<SingleTokenKey>::from_py_dict(deletion_costs, default_deletion_cost);

    // Calculate distances for each candidate in parallel
    let distances: Vec<f64> = candidates
        .par_iter()
        .map(|candidate| {
            _weighted_lev_with_maps(
                s,
                candidate,
                &substitution_cost_map,
                &insertion_cost_map,
                &deletion_cost_map,
            )
        })
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
