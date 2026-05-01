use crate::types::{SingleTokenCostMap, SingleTokenKey, SubstitutionCostMap, SubstitutionKey};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A trait for cost map keys, allowing us to constrain the generic parameter
pub trait CostKey: Clone + Debug + Eq + Hash {}

// Implement the trait for both key types
impl CostKey for SingleTokenKey {}
impl CostKey for SubstitutionKey {}

/// Generic cost map structure that works with different key types
#[derive(Clone, Debug)]
pub struct CostMap<K: CostKey> {
    pub costs: HashMap<K, f64>,
    default_cost: f64,
    max_token_length: usize,
}

impl<K: CostKey> Default for CostMap<K>
where
    K: Default,
{
    fn default() -> Self {
        Self {
            costs: HashMap::new(),
            default_cost: 1.0,
            max_token_length: 1,
        }
    }
}

// Implementation for SubstitutionKey (pair of strings)
impl CostMap<SubstitutionKey> {
    /// Creates a new substitution CostMap with specified costs.
    /// Ensures symmetry by adding both (a, b) and (b, a) if only one is provided when symmetric is true.
    pub fn new(
        custom_costs_input: SubstitutionCostMap,
        default_cost: f64,
        symmetric: bool,
    ) -> Self {
        let mut costs = HashMap::with_capacity(custom_costs_input.len() * 2);

        for ((s1, s2), cost) in custom_costs_input {
            insert_min_cost(&mut costs, (s1.clone(), s2.clone()), cost);
            if symmetric {
                insert_min_cost(&mut costs, (s2.clone(), s1.clone()), cost);
            }
        }

        let max_token_length = costs
            .keys()
            .flat_map(|(s, t)| [s.chars().count(), t.chars().count()])
            .max()
            .unwrap_or(0)
            .max(1);

        CostMap {
            costs,
            default_cost,
            max_token_length,
        }
    }

    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64, symmetric: bool) -> PyResult<Self>
    where
        D: PyDictMethods<'a>,
    {
        let mut substitution_costs = SubstitutionCostMap::new();

        for (key, value) in py_dict.iter() {
            let key_tuple = key.extract::<(String, String)>()?;
            let cost = value.extract::<f64>()?;
            validate_cost(
                cost,
                &format!(
                    "Substitution cost for key ({}, {})",
                    key_tuple.0, key_tuple.1
                ),
            )?;
            substitution_costs.insert((key_tuple.0, key_tuple.1), cost);
        }

        Ok(Self::new(substitution_costs, default_cost, symmetric))
    }

    #[inline]
    pub fn get_cost(&self, source: &str, target: &str) -> f64 {
        self.costs
            .get(&(source.to_string(), target.to_string()))
            .copied()
            .unwrap_or(self.default_cost)
    }

    #[inline]
    pub fn has_key(&self, source: &str, target: &str) -> bool {
        self.costs
            .contains_key(&(source.to_string(), target.to_string()))
    }
}

// Implementation for SingleTokenKey (single string)
impl CostMap<SingleTokenKey> {
    pub fn new(custom_costs_input: SingleTokenCostMap, default_cost: f64) -> Self {
        let max_token_length = custom_costs_input
            .keys()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(0)
            .max(1);
        CostMap {
            costs: custom_costs_input,
            default_cost,
            max_token_length,
        }
    }

    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64) -> PyResult<Self>
    where
        D: PyDictMethods<'a>,
    {
        let mut single_token_costs = SingleTokenCostMap::new();

        for (key, value) in py_dict.iter() {
            let token = key.extract::<String>()?;
            let cost = value.extract::<f64>()?;
            validate_cost(cost, "Cost")?;
            single_token_costs.insert(token, cost);
        }

        Ok(Self::new(single_token_costs, default_cost))
    }

    #[inline]
    pub fn get_cost(&self, token: &str) -> f64 {
        self.costs.get(token).copied().unwrap_or(self.default_cost)
    }

    #[inline]
    pub fn has_key(&self, token: &str) -> bool {
        self.costs.contains_key(token)
    }
}

fn insert_min_cost<K: CostKey>(costs: &mut HashMap<K, f64>, key: K, cost: f64) {
    costs
        .entry(key)
        .and_modify(|existing| *existing = (*existing).min(cost))
        .or_insert(cost);
}

pub(crate) fn validate_cost(cost: f64, label: &str) -> PyResult<()> {
    if !cost.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{label} must be finite, got value: {cost}"
        )));
    }
    if cost < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{label} must be non-negative, got value: {cost}"
        )));
    }
    Ok(())
}

// Common methods for any type of CostMap
impl<K: CostKey> CostMap<K> {
    pub fn default_cost(&self) -> f64 {
        self.default_cost
    }

    #[inline]
    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_token_map_default() {
        let cost_map: CostMap<SingleTokenKey> = CostMap::default();
        assert_eq!(cost_map.default_cost(), 1.0);
        assert!(cost_map.costs.is_empty());
    }

    #[test]
    fn test_single_token_map_with_custom_default() {
        let mut custom_costs = SingleTokenCostMap::new();
        custom_costs.insert("test".to_string(), 0.3);

        let cost_map = CostMap::<SingleTokenKey>::new(custom_costs, 2.0);

        assert_eq!(cost_map.default_cost(), 2.0);
        assert_eq!(cost_map.costs["test"], 0.3);
        assert!(!cost_map.costs.contains_key("unknown"));
    }

    #[test]
    fn test_substitution_map_default() {
        let cost_map = CostMap::<SubstitutionKey>::new(SubstitutionCostMap::new(), 1.0, true);

        assert_eq!(cost_map.default_cost(), 1.0);
        assert!(cost_map.costs.is_empty());
    }

    #[test]
    fn test_substitution_map_asymmetric() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("a".to_string(), "b".to_string()), 0.4);

        let cost_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.5, false);

        assert_eq!(cost_map.costs[&("a".to_string(), "b".to_string())], 0.4);
        assert!(!cost_map
            .costs
            .contains_key(&("b".to_string(), "a".to_string())));
        assert_eq!(cost_map.default_cost(), 1.5);
    }

    #[test]
    fn test_symmetric_substitution_map_conflicts_use_minimum_cost() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("a".to_string(), "b".to_string()), 0.4);
        custom_costs.insert(("b".to_string(), "a".to_string()), 0.2);

        let cost_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.0, true);

        assert_eq!(cost_map.costs[&("a".to_string(), "b".to_string())], 0.2);
        assert_eq!(cost_map.costs[&("b".to_string(), "a".to_string())], 0.2);
    }

    #[test]
    fn test_default_cost_accessor() {
        let sub_map = CostMap::<SubstitutionKey>::new(HashMap::new(), 2.5, true);
        assert_eq!(sub_map.default_cost(), 2.5);

        let single_map = CostMap::<SingleTokenKey>::new(HashMap::new(), 3.0);
        assert_eq!(single_map.default_cost(), 3.0);
    }
}
