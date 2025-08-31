use crate::cost_map::CostMap;
use crate::explanation::EditOperation;
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::custom_levenshtein_distance_with_cost_maps as calculate_core;
use crate::weighted_levenshtein::explain_custom_levenshtein_distance as explain_core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

impl<'py> IntoPyObject<'py> for EditOperation {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    /// Converts the `EditOperation` into a Python tuple
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            EditOperation::Substitute {
                source,
                target,
                cost,
            } => ("substitute", Some(source), Some(target), cost),
            EditOperation::Insert { target, cost } => ("insert", None, Some(target), cost),
            EditOperation::Delete { source, cost } => ("delete", Some(source), None, cost),
        }
        .into_pyobject(py)
        .map(|tuple| tuple.into_any())
    }
}

/// Facade between the Python interface and the core algorithm implementation.
struct LevenshteinCalculator {
    substitution_cost_map: CostMap<SubstitutionKey>,
    insertion_cost_map: CostMap<SingleTokenKey>,
    deletion_cost_map: CostMap<SingleTokenKey>,
}

impl LevenshteinCalculator {
    fn new(
        substitution_costs: &Bound<'_, PyDict>,
        insertion_costs: &Bound<'_, PyDict>,
        deletion_costs: &Bound<'_, PyDict>,
        symmetric_substitution: bool,
        default_substitution_cost: f64,
        default_insertion_cost: f64,
        default_deletion_cost: f64,
    ) -> PyResult<Self> {
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

        Ok(Self {
            substitution_cost_map,
            insertion_cost_map,
            deletion_cost_map,
        })
    }

    fn distance(&self, a: &str, b: &str) -> f64 {
        calculate_core(
            a,
            b,
            &self.substitution_cost_map,
            &self.insertion_cost_map,
            &self.deletion_cost_map,
        )
    }

    fn explain(&self, a: &str, b: &str) -> Vec<EditOperation> {
        explain_core(
            a,
            b,
            &self.substitution_cost_map,
            &self.insertion_cost_map,
            &self.deletion_cost_map,
        )
    }
}

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
    let calculator = LevenshteinCalculator::new(
        substitution_costs,
        insertion_costs,
        deletion_costs,
        symmetric_substitution,
        default_substitution_cost,
        default_insertion_cost,
        default_deletion_cost,
    )?;

    Ok(calculator.distance(a, b))
}

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
fn _explain_weighted_levenshtein_distance(
    py: Python, // For conversion
    a: &str,
    b: &str,
    substitution_costs: &Bound<'_, PyDict>,
    insertion_costs: &Bound<'_, PyDict>,
    deletion_costs: &Bound<'_, PyDict>,
    symmetric_substitution: bool,
    default_substitution_cost: f64,
    default_insertion_cost: f64,
    default_deletion_cost: f64,
) -> PyResult<Vec<PyObject>> {
    let calculator = LevenshteinCalculator::new(
        substitution_costs,
        insertion_costs,
        deletion_costs,
        symmetric_substitution,
        default_substitution_cost,
        default_insertion_cost,
        default_deletion_cost,
    )?;

    let path = calculator.explain(a, b);

    path.into_iter()
        .map(|op| op.into_pyobject(py).map(|bound| bound.into()))
        .collect::<PyResult<Vec<PyObject>>>()
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
    let calculator = LevenshteinCalculator::new(
        substitution_costs,
        insertion_costs,
        deletion_costs,
        symmetric_substitution,
        default_substitution_cost,
        default_insertion_cost,
        default_deletion_cost,
    )?;

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Calculate distances for each candidate in parallel
    let distances: Vec<f64> = candidates
        .par_iter()
        .map(|candidate| calculator.distance(s, candidate))
        .collect();

    Ok(distances)
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _rust_stringdist(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_weighted_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(_batch_weighted_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(_explain_weighted_levenshtein_distance, m)?)?;
    Ok(())
}
