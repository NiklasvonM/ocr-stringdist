use crate::types::{SingleTokenCostMap, SingleTokenKey, SubstitutionCostMap, SubstitutionKey};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

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
}

impl<K: CostKey> Default for CostMap<K>
where
    K: Default,
{
    fn default() -> Self {
        Self {
            costs: HashMap::new(),
            default_cost: 1.0,
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
            costs.entry((s1.clone(), s2.clone())).or_insert(cost);
            if symmetric {
                costs.entry((s2.clone(), s1.clone())).or_insert(cost);
            }
        }

        CostMap {
            costs,
            default_cost,
        }
    }

    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64, symmetric: bool) -> Self
    where
        D: PyDictMethods<'a>,
    {
        let mut substitution_costs = SubstitutionCostMap::new();

        for (key, value) in py_dict.iter() {
            if let Ok(key_tuple) = key.extract::<(String, String)>() {
                if let Ok(cost) = value.extract::<f64>() {
                    substitution_costs.insert((key_tuple.0, key_tuple.1), cost);
                }
            }
        }

        Self::new(substitution_costs, default_cost, symmetric)
    }
}

// Implementation for SingleTokenKey (single string)
impl CostMap<SingleTokenKey> {
    pub fn new(custom_costs_input: SingleTokenCostMap, default_cost: f64) -> Self {
        CostMap {
            costs: custom_costs_input,
            default_cost,
        }
    }

    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64) -> Self
    where
        D: PyDictMethods<'a>,
    {
        let mut single_token_costs = SingleTokenCostMap::new();

        for (key, value) in py_dict.iter() {
            if let Ok(token) = key.extract::<String>() {
                if let Ok(cost) = value.extract::<f64>() {
                    single_token_costs.insert(token, cost);
                }
            }
        }

        Self::new(single_token_costs, default_cost)
    }

    pub fn get_cost(&self, token: &str) -> f64 {
        self.costs.get(token).copied().unwrap_or(self.default_cost)
    }

    pub fn has_key(&self, token: &str) -> bool {
        self.costs.contains_key(token)
    }
}

// Common methods for any type of CostMap
impl<K: CostKey> CostMap<K> {
    pub fn default_cost(&self) -> f64 {
        self.default_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_token_map_default() {
        let cost_map: CostMap<SingleTokenKey> = CostMap::default();
        assert_eq!(cost_map.default_cost(), 1.0);
        assert_eq!(cost_map.get_cost("any_token"), 1.0);
        assert!(!cost_map.has_key("any_token"));
    }

    #[test]
    fn test_single_token_map_with_custom_default() {
        let mut custom_costs = SingleTokenCostMap::new();
        custom_costs.insert("test".to_string(), 0.3);

        let cost_map = CostMap::<SingleTokenKey>::new(custom_costs, 2.0);

        assert_eq!(cost_map.default_cost(), 2.0);
        assert_eq!(cost_map.get_cost("test"), 0.3);
        assert_eq!(cost_map.get_cost("unknown"), 2.0);
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
    fn test_default_cost_accessor() {
        let sub_map = CostMap::<SubstitutionKey>::new(HashMap::new(), 2.5, true);
        assert_eq!(sub_map.default_cost(), 2.5);

        let single_map = CostMap::<SingleTokenKey>::new(HashMap::new(), 3.0);
        assert_eq!(single_map.default_cost(), 3.0);
    }
}
