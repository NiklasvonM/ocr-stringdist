use crate::cost_map::CostMap;
use crate::explanation::EditOperation;
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::custom_levenshtein_distance_with_cost_maps as calculate_core;
use crate::weighted_levenshtein::explain_custom_levenshtein_distance as explain_core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rayon::prelude::*;

impl<'py> IntoPyObject<'py> for EditOperation {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    /// Converts the `EditOperation` into a Python tuple
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let tuple_result = match self {
            EditOperation::Substitute {
                source,
                target,
                cost,
            } => ("substitute", Some(source), Some(target), cost).into_pyobject(py),
            EditOperation::Insert { target, cost } => {
                ("insert", None::<&str>, Some(target), cost).into_pyobject(py)
            }
            EditOperation::Delete { source, cost } => {
                ("delete", Some(source), None::<&str>, cost).into_pyobject(py)
            }
            EditOperation::Match { token } => {
                ("match", Some(token.clone()), Some(token), 0.0).into_pyobject(py)
            }
        };

        tuple_result
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList, PyTuple};

    #[test]
    fn test_levenshtein_distance_with_empty_costs() {
        Python::with_gil(|py| {
            let a = "hello";
            let b = "hxllo";

            let substitution_costs = PyDict::new(py);
            let insertion_costs = PyDict::new(py);
            let deletion_costs = PyDict::new(py);

            let distance = _weighted_levenshtein_distance(
                a,
                b,
                &substitution_costs,
                &insertion_costs,
                &deletion_costs,
                true,
                1.0,
                1.0,
                1.0,
            )
            .unwrap();

            assert_eq!(distance, 1.0);
        });
    }

    #[test]
    fn test_levenshtein_with_custom_substitution_cost() {
        Python::with_gil(|py| {
            let a = "hello";
            let b = "hxllo";

            let substitution_costs = PyDict::new(py);
            substitution_costs.set_item(("e", "x"), 0.2).unwrap();

            let insertion_costs = PyDict::new(py);
            let deletion_costs = PyDict::new(py);

            let distance = _weighted_levenshtein_distance(
                a,
                b,
                &substitution_costs,
                &insertion_costs,
                &deletion_costs,
                true,
                1.0,
                1.0,
                1.0,
            )
            .unwrap();

            assert!((distance - 0.2).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_levenshtein_asymmetric_substitution() {
        Python::with_gil(|py| {
            let a = "ab";
            let b = "ba";

            let substitution_costs = PyDict::new(py);
            substitution_costs.set_item(("a", "b"), 0.1).unwrap();

            let insertion_costs = PyDict::new(py);
            let deletion_costs = PyDict::new(py);

            let distance = _weighted_levenshtein_distance(
                a,
                b,
                &substitution_costs,
                &insertion_costs,
                &deletion_costs,
                false,
                1.0,
                1.0,
                1.0,
            )
            .unwrap();

            // Cost should be 0.1 (a->b) + 1.0 (b->a, default)
            assert!((distance - 1.1).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_negative_default_cost_errors() {
        Python::with_gil(|py| {
            let a = "test";
            let b = "toast";
            let empty_costs = PyDict::new(py);

            // Test negative substitution cost
            let sub_err = _weighted_levenshtein_distance(
                a,
                b,
                &empty_costs,
                &empty_costs,
                &empty_costs,
                true,
                -1.0,
                1.0,
                1.0,
            );
            assert!(sub_err.is_err());
            assert!(sub_err.unwrap_err().is_instance_of::<PyValueError>(py));

            // Test negative insertion cost
            let ins_err = _weighted_levenshtein_distance(
                a,
                b,
                &empty_costs,
                &empty_costs,
                &empty_costs,
                true,
                1.0,
                -1.0,
                1.0,
            );
            assert!(ins_err.is_err());
            assert!(ins_err.unwrap_err().is_instance_of::<PyValueError>(py));

            // Test negative deletion cost
            let del_err = _weighted_levenshtein_distance(
                a,
                b,
                &empty_costs,
                &empty_costs,
                &empty_costs,
                true,
                1.0,
                1.0,
                -1.0,
            );
            assert!(del_err.is_err());
            assert!(del_err.unwrap_err().is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn test_all_edit_operation_variants_into_pyobject() {
        Python::with_gil(|py| {
            // Substitute
            let sub_op = EditOperation::Substitute {
                source: "a".to_string(),
                target: "b".to_string(),
                cost: 0.75,
            };
            let sub_tuple = sub_op.into_pyobject(py).unwrap();
            assert_eq!(sub_tuple.to_string(), "('substitute', 'a', 'b', 0.75)");

            // Insert
            let ins_op = EditOperation::Insert {
                target: "c".to_string(),
                cost: 1.0,
            };
            let ins_tuple = ins_op.into_pyobject(py).unwrap();
            assert_eq!(ins_tuple.to_string(), "('insert', None, 'c', 1.0)");

            // Delete
            let del_op = EditOperation::Delete {
                source: "d".to_string(),
                cost: 1.2,
            };
            let del_tuple = del_op.into_pyobject(py).unwrap();
            assert_eq!(del_tuple.to_string(), "('delete', 'd', None, 1.2)");

            // Match
            let match_op = EditOperation::Match {
                token: "e".to_string(),
            };
            let match_tuple = match_op.into_pyobject(py).unwrap();
            assert_eq!(match_tuple.to_string(), "('match', 'e', 'e', 0.0)");
        });
    }

    #[test]
    fn test_explain_weighted_levenshtein_distance() {
        Python::with_gil(|py| {
            let a = "cat";
            let b = "car";
            let empty_costs = PyDict::new(py);

            let result = _explain_weighted_levenshtein_distance(
                py,
                a,
                b,
                &empty_costs,
                &empty_costs,
                &empty_costs,
                true,
                1.0,
                1.0,
                1.0,
            )
            .unwrap();

            let py_list = PyList::new(py, result).unwrap();
            assert_eq!(py_list.clone().len(), 3);

            let first_op = py_list
                .clone()
                .get_item(0)
                .unwrap()
                .downcast_into::<PyTuple>()
                .unwrap();
            assert_eq!(
                first_op.get_item(0).unwrap().extract::<&str>().unwrap(),
                "match"
            );

            let second_op = py_list
                .clone()
                .get_item(1)
                .unwrap()
                .downcast_into::<PyTuple>()
                .unwrap();
            assert_eq!(
                second_op.get_item(0).unwrap().extract::<&str>().unwrap(),
                "match"
            );

            let third_op = py_list
                .clone()
                .get_item(2)
                .unwrap()
                .downcast_into::<PyTuple>()
                .unwrap();
            assert_eq!(
                third_op.get_item(0).unwrap().extract::<&str>().unwrap(),
                "substitute"
            );
            assert_eq!(third_op.get_item(3).unwrap().extract::<f64>().unwrap(), 1.0);
        });
    }

    #[test]
    fn test_batch_weighted_levenshtein_distance() {
        Python::with_gil(|py| {
            let s = "book";
            let candidates = vec!["back".to_string(), "books".to_string(), "look".to_string()];
            let empty_costs = PyDict::new(py);

            let distances = _batch_weighted_levenshtein_distance(
                s,
                candidates,
                &empty_costs,
                &empty_costs,
                &empty_costs,
                true,
                1.0,
                1.0,
                1.0,
            )
            .unwrap();

            assert_eq!(distances.len(), 3);
            assert_eq!(distances, vec![2.0, 1.0, 1.0]);
        });
    }

    #[test]
    fn test_batch_with_empty_candidate_list() {
        Python::with_gil(|py| {
            let s = "test";
            let candidates: Vec<String> = vec![];
            let empty_costs = PyDict::new(py);

            let distances = _batch_weighted_levenshtein_distance(
                s,
                candidates,
                &empty_costs,
                &empty_costs,
                &empty_costs,
                true,
                1.0,
                1.0,
                1.0,
            )
            .unwrap();

            assert!(distances.is_empty());
        });
    }
}
