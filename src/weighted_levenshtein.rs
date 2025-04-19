use smallvec::SmallVec;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct OcrCostMap {
    /// Maps pairs of characters to their specific substitution cost.
    /// Stores pairs symmetrically for efficient lookup.
    costs: HashMap<(char, char), f64>,
    /// Default cost for substitutions not found in the map.
    default_substitution_cost: f64,
}

impl OcrCostMap {
    /// Creates a new OcrCostMap with specified costs.
    /// Ensures symmetry by adding both (a, b) and (b, a) if only one is provided.
    /// If symmetric, keys are inserted in both directions.
    pub fn new(
        custom_costs_input: HashMap<(char, char), f64>,
        default_substitution_cost: f64,
        symmetric: bool,
    ) -> Self {
        let mut costs = HashMap::with_capacity(custom_costs_input.len() * 2); // Pre-allocate
        for ((c1, c2), cost) in custom_costs_input {
            costs.entry((c1, c2)).or_insert(cost);
            if symmetric {
                costs.entry((c2, c1)).or_insert(cost);
            }
        }

        OcrCostMap {
            costs,
            default_substitution_cost,
        }
    }

    /// Gets the substitution cost between two characters.
    /// Checks the custom map first, then falls back to the
    /// default substitution cost configured within this map instance.
    pub fn get_substitution_cost(&self, c1: char, c2: char) -> f64 {
        if c1 == c2 {
            0.0 // No cost if characters are identical
        } else {
            // Lookup the pair (symmetry is handled by storage in `new`)
            // Use the map's configured default_substitution_cost as the fallback.
            self.costs
                .get(&(c1, c2))
                .copied() // Get the cost if the key exists
                .unwrap_or(self.default_substitution_cost) // Fallback to configured default
        }
    }
}

// Helper to create a range vector with f64 values
fn range_vec_f64(size: usize) -> SmallVec<[f64; 16]> {
    let mut vec = SmallVec::with_capacity(size);
    for i in 0..size {
        vec.push(i as f64);
    }
    vec
}

/// Calculates Levenshtein distance between two vectors using a specified cost map.
pub fn vec_custom_levenshtein_distance_with_cost_map(
    v1: &[char],
    v2: &[char],
    cost_map: &OcrCostMap,
) -> f64 {
    let rows = v1.len() + 1;
    let cols = v2.len() + 1;

    if rows == 1 {
        return (cols - 1) as f64;
    } else if cols == 1 {
        return (rows - 1) as f64;
    }

    let mut cur: SmallVec<[f64; 16]> = range_vec_f64(cols);

    for r in 1..rows {
        let prev = cur.clone();
        cur = SmallVec::from_elem(0.0, cols);
        cur[0] = r as f64;

        let item1 = v1[r - 1];

        for c in 1..cols {
            let item2 = v2[c - 1];

            let deletion = prev[c] + 1.0;
            let insertion = cur[c - 1] + 1.0;

            // Use the provided cost map to get substitution cost
            let substitution_cost = cost_map.get_substitution_cost(item1, item2);
            let substitution = prev[c - 1] + substitution_cost;

            cur[c] = deletion.min(insertion).min(substitution);
        }
    }
    cur[cols - 1]
}

/// Calculates custom Levenshtein distance between two strings using a provided cost map.
pub fn custom_levenshtein_distance_with_cost_map(s1: &str, s2: &str, cost_map: &OcrCostMap) -> f64 {
    if s1 == s2 {
        return 0.0;
    }

    let v1: Vec<char> = s1.chars().collect();
    let v2: Vec<char> = s2.chars().collect();

    vec_custom_levenshtein_distance_with_cost_map(&v1, &v2, cost_map)
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
        assert!(
            (a - b).abs() < epsilon,
            "Assertion failed: {} != {} within epsilon {}",
            a,
            b,
            epsilon
        );
    }

    #[test]
    fn test_custom_levenshtein_with_custom_map() {
        let mut custom_costs = HashMap::new();
        custom_costs.insert(('a', 'b'), 0.1);
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("abc", "bbc", &cost_map),
            0.1,
            1e-9,
        );
    }
}
