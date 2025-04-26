use crate::types::{SingleTokenCostMap, SingleTokenKey, SubstitutionCostMap, SubstitutionKey};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// A trait for cost map keys, allowing us to constrain the generic parameter
pub trait CostKey: Clone + Debug + Eq + Hash {}

// Implement the trait for both key types
impl CostKey for SingleTokenKey {}
impl CostKey for SubstitutionKey {}

/// Generic cost map structure that works with different key types
#[derive(Clone, Debug)]
pub struct CostMap<K: CostKey> {
    /// The costs map
    pub costs: HashMap<K, f64>,
    /// Default cost for operations not found in the map
    default_cost: f64,
    /// Maximum token length in the map
    pub max_token_length: usize,
}

impl<K: CostKey> Default for CostMap<K>
where
    K: Default,
{
    /// Creates a new CostMap with default values
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
        let mut max_length = 1;

        for ((s1, s2), cost) in custom_costs_input {
            costs.entry((s1.clone(), s2.clone())).or_insert(cost);
            if symmetric {
                costs.entry((s2.clone(), s1.clone())).or_insert(cost);
            }

            // Update max token length
            max_length = max_length.max(s1.chars().count()).max(s2.chars().count());
        }

        CostMap {
            costs,
            default_cost,
            max_token_length: max_length,
        }
    }

    /// Creates a new substitution CostMap with the specified custom costs.
    /// Uses default values for other parameters.
    pub fn with_costs(custom_costs: SubstitutionCostMap) -> Self {
        Self::new(custom_costs, 1.0, true)
    }

    #[cfg(feature = "python")]
    /// Creates a substitution CostMap from a Python dictionary.
    /// This method is only available when the "python" feature is enabled.
    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64, symmetric: bool) -> Self
    where
        D: PyDictMethods<'a>,
    {
        let mut substitution_costs = SubstitutionCostMap::new();
        let mut max_length = 1;

        // Convert Python dictionary to Rust HashMap
        for (key, value) in py_dict.iter() {
            if let Ok(key_tuple) = key.extract::<(String, String)>() {
                if let Ok(cost) = value.extract::<f64>() {
                    substitution_costs.insert((key_tuple.0.clone(), key_tuple.1.clone()), cost);

                    // Update max token length
                    max_length = max_length
                        .max(key_tuple.0.chars().count())
                        .max(key_tuple.1.chars().count());
                }
            }
        }

        // Create the CostMap
        Self::new(substitution_costs, default_cost, symmetric)
    }

    /// Gets the substitution cost between two strings.
    pub fn get_cost(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            0.0 // No cost if strings are identical
        } else {
            let key_pair = (s1.to_string(), s2.to_string());

            // Lookup the pair (symmetry is handled by storage in `new`)
            // Use the map's configured default_cost as the fallback.
            self.costs
                .get(&key_pair)
                .copied()
                .unwrap_or(self.default_cost)
        }
    }

    /// Checks if the cost map contains a specific substitution
    pub fn has_key(&self, s1: &str, s2: &str) -> bool {
        let key_pair = (s1.to_string(), s2.to_string());
        self.costs.contains_key(&key_pair)
    }
}

// Implementation for SingleTokenKey (single string)
impl CostMap<SingleTokenKey> {
    /// Creates a new single token CostMap for insertion or deletion operations
    pub fn new(custom_costs_input: SingleTokenCostMap, default_cost: f64) -> Self {
        let mut max_length = 1;

        // Calculate max token length
        for key in custom_costs_input.keys() {
            max_length = max_length.max(key.chars().count());
        }

        CostMap {
            costs: custom_costs_input,
            default_cost,
            max_token_length: max_length,
        }
    }

    /// Creates a new single token CostMap with the specified custom costs.
    /// Uses default value for default cost.
    pub fn with_costs(custom_costs: SingleTokenCostMap) -> Self {
        Self::new(custom_costs, 1.0)
    }

    #[cfg(feature = "python")]
    /// Creates a single token CostMap from a Python dictionary.
    /// This method is only available when the "python" feature is enabled.
    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64) -> Self
    where
        D: PyDictMethods<'a>,
    {
        let mut single_token_costs = SingleTokenCostMap::new();
        let mut max_length = 1;

        // Convert Python dictionary to Rust HashMap
        for (key, value) in py_dict.iter() {
            if let Ok(token) = key.extract::<String>() {
                if let Ok(cost) = value.extract::<f64>() {
                    single_token_costs.insert(token.clone(), cost);

                    // Update max token length
                    max_length = max_length.max(token.chars().count());
                }
            }
        }

        // Create the CostMap
        Self::new(single_token_costs, default_cost)
    }

    /// Gets the cost for a single token (insertion or deletion).
    pub fn get_cost(&self, token: &str) -> f64 {
        self.costs.get(token).copied().unwrap_or(self.default_cost)
    }

    /// Checks if the cost map contains a specific single token
    pub fn has_key(&self, token: &str) -> bool {
        self.costs.contains_key(token)
    }
}

// Common methods for any type of CostMap
impl<K: CostKey> CostMap<K> {
    /// Returns the default cost for this cost map
    pub fn default_cost(&self) -> f64 {
        self.default_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_token_map_default() {
        // Test with default initialization
        let cost_map: CostMap<SingleTokenKey> = CostMap::default();
        assert_eq!(cost_map.default_cost(), 1.0);
        assert_eq!(cost_map.get_cost("any_token"), 1.0);
        assert!(!cost_map.has_key("any_token"));
    }

    #[test]
    fn test_single_token_map_with_costs() {
        let mut custom_costs = SingleTokenCostMap::new();
        custom_costs.insert("a".to_string(), 0.5);
        custom_costs.insert("b".to_string(), 0.8);

        // Test with_costs constructor (default cost 1.0)
        let cost_map = CostMap::<SingleTokenKey>::with_costs(custom_costs);

        // Test getting costs for tokens
        assert_eq!(cost_map.get_cost("a"), 0.5);
        assert_eq!(cost_map.get_cost("b"), 0.8);
        assert_eq!(cost_map.get_cost("c"), 1.0); // Default cost

        // Test has_key
        assert!(cost_map.has_key("a"));
        assert!(cost_map.has_key("b"));
        assert!(!cost_map.has_key("c"));
    }

    #[test]
    fn test_single_token_map_with_custom_default() {
        let mut custom_costs = SingleTokenCostMap::new();
        custom_costs.insert("test".to_string(), 0.3);

        // Test new constructor with custom default cost
        let cost_map = CostMap::<SingleTokenKey>::new(custom_costs, 2.0);

        assert_eq!(cost_map.default_cost(), 2.0);
        assert_eq!(cost_map.get_cost("test"), 0.3);
        assert_eq!(cost_map.get_cost("unknown"), 2.0);
    }

    #[test]
    fn test_substitution_map_default() {
        let cost_map: CostMap<SubstitutionKey> = CostMap {
            costs: HashMap::new(),
            default_cost: 1.0,
            max_token_length: 1,
        };

        assert_eq!(cost_map.default_cost(), 1.0);
        assert_eq!(cost_map.get_cost("a", "b"), 1.0);
        assert!(!cost_map.has_key("a", "b"));
    }

    #[test]
    fn test_substitution_map_with_costs() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("0".to_string(), "o".to_string()), 0.2);
        custom_costs.insert(("l".to_string(), "1".to_string()), 0.3);

        // Test with_costs constructor (symmetric by default)
        let cost_map = CostMap::<SubstitutionKey>::with_costs(custom_costs);

        // Test getting costs
        assert_eq!(cost_map.get_cost("0", "o"), 0.2);
        assert_eq!(cost_map.get_cost("o", "0"), 0.2); // Symmetry check
        assert_eq!(cost_map.get_cost("l", "1"), 0.3);
        assert_eq!(cost_map.get_cost("1", "l"), 0.3); // Symmetry check
        assert_eq!(cost_map.get_cost("a", "b"), 1.0); // Default

        // Test same character
        assert_eq!(cost_map.get_cost("a", "a"), 0.0); // Same char = 0 cost

        // Test has_key
        assert!(cost_map.has_key("0", "o"));
        assert!(cost_map.has_key("o", "0")); // Symmetry check
        assert!(!cost_map.has_key("a", "b"));
    }

    #[test]
    fn test_substitution_map_asymmetric() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("a".to_string(), "b".to_string()), 0.4);

        // Create with symmetric=false
        let cost_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.5, false);

        // Test asymmetry
        assert_eq!(cost_map.get_cost("a", "b"), 0.4);
        assert_eq!(cost_map.get_cost("b", "a"), 1.5); // Should be default cost

        assert!(cost_map.has_key("a", "b"));
        assert!(!cost_map.has_key("b", "a")); // Should not exist
    }

    #[test]
    fn test_default_cost_accessor() {
        // Test for SubstitutionKey
        let sub_map = CostMap::<SubstitutionKey>::new(HashMap::new(), 2.5, true);
        assert_eq!(sub_map.default_cost(), 2.5);

        // Test for SingleTokenKey
        let single_map = CostMap::<SingleTokenKey>::new(HashMap::new(), 3.0);
        assert_eq!(single_map.default_cost(), 3.0);
    }
}
