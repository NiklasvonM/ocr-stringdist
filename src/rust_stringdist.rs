use crate::custom_levenshtein_distance_with_cost_map as _weighted_lev_with_map;
use crate::OcrCostMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

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
    let default_cost_value = default_cost.unwrap_or(1.0);
    let mut char_costs: HashMap<(char, char), f64> = HashMap::new();

    // Convert Python dictionary to Rust HashMap
    for (key, value) in cost_map.iter() {
        if let Ok(key_tuple) = key.extract::<(String, String)>() {
            if let Ok(cost) = value.extract::<f64>() {
                // Extract the first character from each string, if they exist
                if let (Some(c1), Some(c2)) =
                    (key_tuple.0.chars().next(), key_tuple.1.chars().next())
                {
                    char_costs.insert((c1, c2), cost);
                }
            }
        }
    }

    // Create a custom cost map and calculate the distance
    let custom_cost_map = OcrCostMap::new(char_costs, default_cost_value, symmetric);
    Ok(_weighted_lev_with_map(a, b, &custom_cost_map))
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _rust_stringdist(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_weighted_levenshtein_distance, m)?)?;
    Ok(())
}
