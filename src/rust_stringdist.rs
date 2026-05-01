use crate::cost_map::{validate_cost, CostMap};
use crate::explanation::EditOperation;
use crate::transitive_costs::compute_closed_cost_maps;
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::{custom_levenshtein_distance, explain_custom_levenshtein};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rayon::prelude::*;

impl<'py> IntoPyObject<'py> for EditOperation {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

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

/// Holds raw cost maps and runs the DP against them. Transitive closure is
/// opt-in via [`closed_cost_maps`]; the constructor never runs Floyd-Warshall.
#[pyclass]
#[derive(Debug)]
struct RustLevenshteinCalculator {
    sub: CostMap<SubstitutionKey>,
    ins: CostMap<SingleTokenKey>,
    del: CostMap<SingleTokenKey>,
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
        validate_cost(default_substitution_cost, "Default substitution cost")?;
        validate_cost(default_insertion_cost, "Default insertion cost")?;
        validate_cost(default_deletion_cost, "Default deletion cost")?;

        let sub = CostMap::<SubstitutionKey>::from_py_dict(
            substitution_costs,
            default_substitution_cost,
            symmetric_substitution,
        )?;
        let ins = CostMap::<SingleTokenKey>::from_py_dict(insertion_costs, default_insertion_cost)?;
        let del = CostMap::<SingleTokenKey>::from_py_dict(deletion_costs, default_deletion_cost)?;

        Ok(Self { sub, ins, del })
    }

    fn distance(&self, a: &str, b: &str) -> f64 {
        custom_levenshtein_distance(a, b, &self.sub, &self.ins, &self.del)
    }

    fn batch_distance(&self, py: Python<'_>, s: String, candidates: Vec<String>) -> Vec<f64> {
        if candidates.is_empty() {
            return Vec::new();
        }
        py.allow_threads(|| {
            candidates
                .par_iter()
                .map(|c| custom_levenshtein_distance(&s, c, &self.sub, &self.ins, &self.del))
                .collect()
        })
    }

    fn explain(&self, py: Python<'_>, a: &str, b: &str) -> PyResult<Vec<PyObject>> {
        explain_custom_levenshtein(a, b, &self.sub, &self.ins, &self.del)
            .into_iter()
            .map(|op| op.into_pyobject(py).map(|bound| bound.into()))
            .collect::<PyResult<Vec<PyObject>>>()
    }

    /// Computes effective edit costs via transitive closure and returns three
    /// Python dicts: `(substitution_costs, insertion_costs, deletion_costs)`.
    /// Generated substitutions are pruned only when `prune` is true.
    /// `max_node_length` caps the length of intermediate graph nodes; pass
    /// `None` to derive it from the input.
    ///
    /// The Python wrapper assembles these into a new `WeightedLevenshtein`
    /// whose `.distance()` and `.explain()` use the closed costs directly.
    #[pyo3(signature = (prune = false, max_node_length = None))]
    fn closed_cost_maps<'py>(
        &self,
        py: Python<'py>,
        prune: bool,
        max_node_length: Option<usize>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let (closed_sub, closed_ins, closed_del) =
            compute_closed_cost_maps(&self.sub, &self.ins, &self.del, prune, max_node_length)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let sub_dict = PyDict::new(py);
        for ((source, target), cost) in closed_sub {
            sub_dict.set_item((source, target), cost)?;
        }

        let ins_dict = PyDict::new(py);
        for (token, cost) in closed_ins {
            ins_dict.set_item(token, cost)?;
        }

        let del_dict = PyDict::new(py);
        for (token, cost) in closed_del {
            del_dict.set_item(token, cost)?;
        }

        Ok((sub_dict, ins_dict, del_dict))
    }
}

#[pymodule]
pub fn _rust_stringdist(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLevenshteinCalculator>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyDict, PyList, PyTuple};

    fn make_calculator(
        py: Python<'_>,
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
        });
    }

    #[test]
    fn test_constructor_does_not_apply_closure() {
        // Without calling closed_cost_maps, transitive paths are not auto-applied.
        // sub(a->b)=0.1, sub(b->c)=0.1: direct a->c lookup falls back to default 1.0.
        Python::with_gil(|py| {
            let calc =
                make_calculator(py, &[(("a", "b"), 0.1), (("b", "c"), 0.1)], &[], &[], false);
            assert!((calc.distance("a", "c") - 1.0).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_closed_cost_maps_finds_chain() {
        Python::with_gil(|py| {
            let calc =
                make_calculator(py, &[(("a", "b"), 0.1), (("b", "c"), 0.1)], &[], &[], false);
            let (sub, _ins, _del) = calc.closed_cost_maps(py, false, None).unwrap();
            let cost: f64 = sub
                .get_item(("a".to_string(), "c".to_string()))
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((cost - 0.2).abs() < 1e-9);
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
}
