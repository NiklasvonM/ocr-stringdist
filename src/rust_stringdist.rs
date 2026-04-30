use crate::cost_map::CostMap;
use crate::explanation::EditOperation;
use crate::transitive_costs::{compute_effective_costs, EffectiveCosts};
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::custom_levenshtein_distance_precomputed;
use crate::weighted_levenshtein::explain_custom_levenshtein_precomputed;
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
        match self {
            EditOperation::Substitute {
                source,
                target,
                cost,
            } => ("substitute", Some(source), Some(target), cost),
            EditOperation::Insert { target, cost } => ("insert", None, Some(target), cost),
            EditOperation::Delete { source, cost } => ("delete", Some(source), None, cost),
            EditOperation::Match { token } => ("match", Some(token.clone()), Some(token), 0.0),
        }
        .into_pyobject(py)
    }
}

/// Precomputes effective substitution, insertion, and deletion costs once and
/// reuses them for every `.distance()` / `.batch_distance()` call.
#[pyclass]
#[derive(Debug)]
struct RustLevenshteinCalculator {
    costs: EffectiveCosts,
}

#[pymethods]
impl RustLevenshteinCalculator {
    #[new]
    #[pyo3(signature = (
        substitution_costs,
        insertion_costs,
        deletion_costs,
        symmetric_substitution = true,
        default_substitution_cost = 1.0,
        default_insertion_cost = 1.0,
        default_deletion_cost = 1.0,
    ))]
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

        let sub_map = CostMap::<SubstitutionKey>::from_py_dict(
            substitution_costs,
            default_substitution_cost,
            symmetric_substitution,
        );
        let ins_map =
            CostMap::<SingleTokenKey>::from_py_dict(insertion_costs, default_insertion_cost);
        let del_map =
            CostMap::<SingleTokenKey>::from_py_dict(deletion_costs, default_deletion_cost);

        let costs = compute_effective_costs(&sub_map, &ins_map, &del_map);

        Ok(Self { costs })
    }

    fn distance(&self, a: &str, b: &str) -> f64 {
        custom_levenshtein_distance_precomputed(a, b, &self.costs)
    }

    fn batch_distance(&self, py: Python<'_>, s: String, candidates: Vec<String>) -> Vec<f64> {
        if candidates.is_empty() {
            return Vec::new();
        }
        py.allow_threads(|| {
            candidates
                .par_iter()
                .map(|c| custom_levenshtein_distance_precomputed(&s, c, &self.costs))
                .collect()
        })
    }

    fn explain(&self, py: Python<'_>, a: &str, b: &str) -> PyResult<Vec<PyObject>> {
        explain_custom_levenshtein_precomputed(a, b, &self.costs)
            .into_iter()
            .map(|op| op.into_pyobject(py).map(|bound| bound.into()))
            .collect::<PyResult<Vec<PyObject>>>()
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

/// A Python module implemented in Rust.
#[pymodule]
pub fn _rust_stringdist(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLevenshteinCalculator>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList, PyTuple};

    fn make_calculator<'py>(
        py: Python<'py>,
        sub_costs: &[((&str, &str), f64)],
        ins_costs: &[(&str, f64)],
        del_costs: &[(&str, f64)],
        symmetric: bool,
    ) -> RustLevenshteinCalculator {
        let sub = PyDict::new(py);
        for ((a, b), c) in sub_costs {
            sub.set_item((*a, *b), c).unwrap();
        }
        let ins = PyDict::new(py);
        for (k, v) in ins_costs {
            ins.set_item(k, v).unwrap();
        }
        let del = PyDict::new(py);
        for (k, v) in del_costs {
            del.set_item(k, v).unwrap();
        }
        RustLevenshteinCalculator::new(&sub, &ins, &del, symmetric, 1.0, 1.0, 1.0).unwrap()
    }

    #[test]
    fn test_distance_with_empty_costs() {
        Python::with_gil(|py| {
            let calc = make_calculator(py, &[], &[], &[], true);
            assert_eq!(calc.distance("hello", "hxllo"), 1.0);
        });
    }

    #[test]
    fn test_distance_with_custom_substitution_cost() {
        Python::with_gil(|py| {
            let calc = make_calculator(py, &[(("e", "x"), 0.2)], &[], &[], true);
            assert!((calc.distance("hello", "hxllo") - 0.2).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_asymmetric_substitution() {
        Python::with_gil(|py| {
            let calc = make_calculator(py, &[(("a", "b"), 0.1)], &[], &[], false);
            // a->b costs 0.1; b->a uses default 1.0 -> total 1.1
            assert!((calc.distance("ab", "ba") - 1.1).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_negative_default_cost_errors() {
        Python::with_gil(|py| {
            let empty = PyDict::new(py);

            let sub_err =
                RustLevenshteinCalculator::new(&empty, &empty, &empty, true, -1.0, 1.0, 1.0);
            assert!(sub_err.is_err());
            assert!(sub_err.unwrap_err().is_instance_of::<PyValueError>(py));

            let ins_err =
                RustLevenshteinCalculator::new(&empty, &empty, &empty, true, 1.0, -1.0, 1.0);
            assert!(ins_err.is_err());
            assert!(ins_err.unwrap_err().is_instance_of::<PyValueError>(py));

            let del_err =
                RustLevenshteinCalculator::new(&empty, &empty, &empty, true, 1.0, 1.0, -1.0);
            assert!(del_err.is_err());
            assert!(del_err.unwrap_err().is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn test_edit_op_substitute_into_pyobject() {
        Python::with_gil(|py| {
            let op = EditOperation::Substitute {
                source: "a".to_string(),
                target: "b".to_string(),
                cost: 0.75,
            };
            let tuple = op.into_pyobject(py).unwrap();
            assert_eq!(tuple.to_string(), "('substitute', 'a', 'b', 0.75)");
        });
    }

    #[test]
    fn test_edit_op_insert_into_pyobject() {
        Python::with_gil(|py| {
            let op = EditOperation::Insert {
                target: "c".to_string(),
                cost: 1.0,
            };
            let tuple = op.into_pyobject(py).unwrap();
            assert_eq!(tuple.to_string(), "('insert', None, 'c', 1.0)");
        });
    }

    #[test]
    fn test_edit_op_delete_into_pyobject() {
        Python::with_gil(|py| {
            let op = EditOperation::Delete {
                source: "d".to_string(),
                cost: 1.2,
            };
            let tuple = op.into_pyobject(py).unwrap();
            assert_eq!(tuple.to_string(), "('delete', 'd', None, 1.2)");
        });
    }

    #[test]
    fn test_edit_op_match_into_pyobject() {
        Python::with_gil(|py| {
            let op = EditOperation::Match {
                token: "e".to_string(),
            };
            let tuple = op.into_pyobject(py).unwrap();
            assert_eq!(tuple.to_string(), "('match', 'e', 'e', 0.0)");
        });
    }

    #[test]
    fn test_explain() {
        Python::with_gil(|py| {
            let calc = make_calculator(py, &[], &[], &[], true);
            let result = calc.explain(py, "cat", "car").unwrap();

            let py_list = PyList::new(py, result).unwrap();
            assert_eq!(py_list.len(), 3);

            let op = |i: usize| -> String {
                py_list
                    .get_item(i)
                    .unwrap()
                    .downcast_into::<PyTuple>()
                    .unwrap()
                    .get_item(0)
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            };
            assert_eq!(op(0), "match");
            assert_eq!(op(1), "match");
            assert_eq!(op(2), "substitute");
        });
    }

    #[test]
    fn test_batch_distance() {
        Python::with_gil(|py| {
            let calc = make_calculator(py, &[], &[], &[], true);
            let distances = calc.batch_distance(
                py,
                "book".to_string(),
                vec!["back".to_string(), "books".to_string(), "look".to_string()],
            );
            assert_eq!(distances, vec![2.0, 1.0, 1.0]);
        });
    }

    #[test]
    fn test_batch_distance_empty() {
        Python::with_gil(|py| {
            let calc = make_calculator(py, &[], &[], &[], true);
            assert!(calc
                .batch_distance(py, "test".to_string(), vec![])
                .is_empty());
        });
    }
}
