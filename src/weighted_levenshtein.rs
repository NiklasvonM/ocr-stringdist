use crate::types::SubstitutionCostMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct OcrCostMap {
    /// Maps pairs of strings to their specific substitution cost.
    pub costs: SubstitutionCostMap,
    /// Default cost for substitutions not found in the map.
    default_substitution_cost: f64,
}

impl Default for OcrCostMap {
    /// Creates a new OcrCostMap with default values.
    fn default() -> Self {
        Self {
            costs: SubstitutionCostMap::new(),
            default_substitution_cost: 1.0,
        }
    }
}

impl OcrCostMap {
    /// Creates a new OcrCostMap with specified costs.
    /// Ensures symmetry by adding both (a, b) and (b, a) if only one is provided.
    /// If symmetric, keys are inserted in both directions.
    pub fn new(
        custom_costs_input: SubstitutionCostMap,
        default_substitution_cost: f64,
        symmetric: bool,
    ) -> Self {
        let mut costs = SubstitutionCostMap::with_capacity(custom_costs_input.len() * 2);
        for ((s1, s2), cost) in custom_costs_input {
            costs.entry((s1.clone(), s2.clone())).or_insert(cost);
            if symmetric {
                costs.entry((s2, s1)).or_insert(cost);
            }
        }

        OcrCostMap {
            costs,
            default_substitution_cost,
        }
    }

    /// Creates a new OcrCostMap with the specified custom costs.
    /// Uses default values for other parameters.
    pub fn with_costs(custom_costs: SubstitutionCostMap) -> Self {
        Self::new(custom_costs, 1.0, true)
    }

    #[cfg(feature = "python")]
    /// Creates an OcrCostMap from a Python dictionary.
    /// This method is only available when the "python" feature is enabled.
    pub fn from_py_dict<'a, D>(py_dict: &'a D, default_cost: f64, symmetric: bool) -> Self
    where
        D: PyDictMethods<'a>,
    {
        let mut substitution_costs = SubstitutionCostMap::new();

        // Convert Python dictionary to Rust HashMap
        for (key, value) in py_dict.iter() {
            if let Ok(key_tuple) = key.extract::<(String, String)>() {
                if let Ok(cost) = value.extract::<f64>() {
                    substitution_costs.insert((key_tuple.0, key_tuple.1), cost);
                }
            }
        }

        // Create the OcrCostMap
        Self::new(substitution_costs, default_cost, symmetric)
    }

    /// Gets the substitution cost between two strings.
    /// Checks the custom map first, then falls back to the
    /// default substitution cost configured within this map instance.
    pub fn get_substitution_cost(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            0.0 // No cost if strings are identical
        } else {
            let key_pair = (s1.to_string(), s2.to_string());

            // Lookup the pair (symmetry is handled by storage in `new`)
            // Use the map's configured default_substitution_cost as the fallback.
            self.costs
                .get(&key_pair)
                .copied() // Get the cost if the key exists
                .unwrap_or(self.default_substitution_cost) // Fallback to configured default
        }
    }

    /// Checks if the cost map contains a specific substitution
    pub fn has_substitution(&self, s1: &str, s2: &str) -> bool {
        let key_pair = (s1.to_string(), s2.to_string());
        self.costs.contains_key(&key_pair)
    }
}

/// Calculates custom Levenshtein distance between two strings using a provided cost map.
/// This implementation considers string-to-string substitutions rather than just characters.
pub fn custom_levenshtein_distance_with_cost_map(
    source: &str,
    target: &str,
    cost_map: &OcrCostMap,
    max_token_characters: usize,
) -> f64 {
    if source == target {
        return 0.0;
    }

    let len_source = source.chars().count();
    let len_target = target.chars().count();

    // For empty strings, return the length of the other string
    if len_source == 0 {
        return len_target as f64;
    }
    if len_target == 0 {
        return len_source as f64;
    }

    // Convert to character vectors for correct Unicode handling
    let source_chars: Vec<char> = source.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();

    // Create dynamic programming matrix
    let mut dp = vec![vec![0.0; len_target + 1]; len_source + 1];

    // Initialize first row and column
    for i in 0..=len_source {
        dp[i][0] = i as f64;
    }
    for j in 0..=len_target {
        dp[0][j] = j as f64;
    }

    // Limit on substring lengths to check
    let max_substr_len = max_token_characters.min(len_source.max(len_target));

    // Fill the dp matrix
    for i in 1..=len_source {
        for j in 1..=len_target {
            // Standard Levenshtein operations
            let deletion = dp[i - 1][j] + 1.0;
            let insertion = dp[i][j - 1] + 1.0;

            // Single character substitution
            let c1 = source_chars[i - 1].to_string();
            let c2 = target_chars[j - 1].to_string();
            let char_sub_cost = if source_chars[i - 1] == target_chars[j - 1] {
                0.0
            } else {
                cost_map.get_substitution_cost(&c1, &c2)
            };
            let char_substitution = dp[i - 1][j - 1] + char_sub_cost;

            // Initialize with standard operations
            dp[i][j] = deletion.min(insertion).min(char_substitution);

            // Check multi-character substitutions
            check_multi_char_substitutions(
                i,
                j,
                &source_chars,
                &target_chars,
                &mut dp,
                max_substr_len,
                cost_map,
            );
        }
    }

    dp[len_source][len_target]
}

/// Helper function to check multi-character substitutions for the Levenshtein algorithm
fn check_multi_char_substitutions(
    current_i: usize,
    current_j: usize,
    source_chars: &[char],
    target_chars: &[char],
    dp: &mut [Vec<f64>],
    max_substr_len: usize,
    cost_map: &OcrCostMap,
) {
    // Skip single character case as it's already handled in the main function
    let max_source_len = max_substr_len.min(current_i);
    let max_target_len = max_substr_len.min(current_j);

    // If we can't form multi-character substitutions, exit early
    if max_source_len < 1 || max_target_len < 1 {
        return;
    }

    for source_len in 1..=max_source_len {
        for target_len in 1..=max_target_len {
            // Skip single-char to single-char substitution as it's already handled
            if source_len == 1 && target_len == 1 {
                continue;
            }

            // Calculate substring bounds
            let source_start = current_i - source_len;
            let target_start = current_j - target_len;

            // Extract substrings as strings
            let source_substr: String = source_chars[source_start..current_i].iter().collect();
            let target_substr: String = target_chars[target_start..current_j].iter().collect();

            // Only check if this substitution exists in the cost map
            if !cost_map.has_substitution(&source_substr, &target_substr) {
                continue;
            }

            let sub_cost = cost_map.get_substitution_cost(&source_substr, &target_substr);
            let new_cost = dp[source_start][target_start] + sub_cost;

            // Update if this gives a better result
            if new_cost < dp[current_i][current_j] {
                dp[current_i][current_j] = new_cost;
            }
        }
    }
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
        let cost_map = OcrCostMap::new(
            SubstitutionCostMap::from([(("a".to_string(), "b".to_string()), 0.1)]),
            1.0,
            true,
        );

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("abc", "bbc", &cost_map, 3),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_multi_character_substitutions() {
        let cost_map = OcrCostMap::new(
            SubstitutionCostMap::from([(("h".to_string(), "In".to_string()), 0.2)]),
            1.0,
            true,
        );

        // Test that "hi" with "Ini" has a low cost due to the special substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("hi", "Ini", &cost_map, 2),
            0.2, // Only the h->In substitution cost
            1e-9,
        );

        // Test another example
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("hello", "Inello", &cost_map, 2),
            0.2, // Only the h->In substitution cost
            1e-9,
        );
    }

    #[test]
    fn test_multiple_substitutions_in_same_string() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("h".to_string(), "In".to_string()), 0.2);
        custom_costs.insert(("l".to_string(), "1".to_string()), 0.3);
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        // Test multiple substitutions in the same string
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("hello", "Ine11o", &cost_map, 2),
            0.8, // 0.2 for h->In and 0.3+0.3 for l->1 twice
            1e-9,
        );
    }

    #[test]
    fn test_overlapping_substitution_patterns() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("rn".to_string(), "m".to_string()), 0.1); // common OCR confusion
        custom_costs.insert(("cl".to_string(), "d".to_string()), 0.2); // another common confusion
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        // Test the rn->m substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("corner", "comer", &cost_map, 2),
            0.1,
            1e-9,
        );

        // Test the cl->d substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("class", "dass", &cost_map, 2),
            0.2,
            1e-9,
        );
    }

    #[test]
    fn test_asymmetric_costs() {
        // Sometimes OCR errors aren't symmetric - going from reference to OCR
        // might have different likelihood than going from OCR to reference
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("0".to_string(), "O".to_string()), 0.1); // 0->O is common
        custom_costs.insert(("O".to_string(), "0".to_string()), 0.5); // O->0 is less common
        let cost_map = OcrCostMap::new(custom_costs, 1.0, false); // asymmetric costs

        // Test 0->O substitution (lower cost)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("R0AD", "ROAD", &cost_map, 1),
            0.1,
            1e-9,
        );

        // Test O->0 substitution (higher cost)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("rOad", "r0ad", &cost_map, 1),
            0.5,
            1e-9,
        );
    }

    #[test]
    fn test_substitution_at_word_boundaries() {
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("rn".to_string(), "m".to_string()), 0.1);
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        // Test substitution at start of word
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("rnat", "mat", &cost_map, 2),
            0.1,
            1e-9,
        );

        // Test substitution at end of word
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("burn", "bum", &cost_map, 2),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_empty_cost_map() {
        // Create a cost map with no custom substitution costs
        let cost_map = OcrCostMap::new(SubstitutionCostMap::new(), 1.0, true);

        // Test that "h" -> "In" costs 2.0 (1 deletion + 1 substitution) since there's no custom mapping
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("h", "In", &cost_map, 1),
            2.0,
            1e-9,
        );

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("kitten", "sitting", &cost_map, 1),
            3.0,
            1e-9,
        );

        // Test with non-ASCII characters - correct distance is 4
        // café -> coffee:
        // - Replace 'a' with 'o' (1)
        // - Keep 'f' (0)
        // - Replace 'é' with 'f' (1)
        // - Insert 'e' (1)
        // - Insert 'e' (1)
        // Total: 4 operations
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("café", "coffee", &cost_map, 1),
            4.0, // 4 edits required
            1e-9,
        );
    }

    #[test]
    fn test_default_cost_map() {
        // Create a cost map with no custom substitution costs
        let cost_map = OcrCostMap::default();

        // Test that "h" -> "In" costs 2.0 (1 deletion + 1 substitution) since there's no custom mapping
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("h", "In", &cost_map, 1),
            2.0,
            1e-9,
        );

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("kitten", "sitting", &cost_map, 1),
            3.0,
            1e-9,
        );

        // Test with non-ASCII characters - correct distance is 4
        // café -> coffee:
        // - Replace 'a' with 'o' (1)
        // - Keep 'f' (0)
        // - Replace 'é' with 'f' (1)
        // - Insert 'e' (1)
        // - Insert 'e' (1)
        // Total: 4 operations
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("café", "coffee", &cost_map, 1),
            4.0, // 4 edits required
            1e-9,
        );
    }
}
