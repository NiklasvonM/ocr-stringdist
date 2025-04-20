use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct OcrCostMap {
    /// Maps pairs of strings to their specific substitution cost.
    costs: HashMap<(String, String), f64>,
    /// Default cost for substitutions not found in the map.
    default_substitution_cost: f64,
}

impl OcrCostMap {
    /// Creates a new OcrCostMap with specified costs.
    /// Ensures symmetry by adding both (a, b) and (b, a) if only one is provided.
    /// If symmetric, keys are inserted in both directions.
    pub fn new(
        custom_costs_input: HashMap<(String, String), f64>,
        default_substitution_cost: f64,
        symmetric: bool,
    ) -> Self {
        let mut costs = HashMap::with_capacity(custom_costs_input.len() * 2); // Pre-allocate
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

    /// Gets the substitution cost between two strings.
    /// Checks the custom map first, then falls back to the
    /// default substitution cost configured within this map instance.
    pub fn get_substitution_cost(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            0.0 // No cost if strings are identical
        } else {
            // Lookup the pair (symmetry is handled by storage in `new`)
            // Use the map's configured default_substitution_cost as the fallback.
            self.costs
                .get(&(s1.to_string(), s2.to_string()))
                .copied() // Get the cost if the key exists
                .unwrap_or(self.default_substitution_cost) // Fallback to configured default
        }
    }
}

/// Calculates custom Levenshtein distance between two strings using a provided cost map.
/// This implementation considers string-to-string substitutions rather than just characters.
pub fn custom_levenshtein_distance_with_cost_map(s1: &str, s2: &str, cost_map: &OcrCostMap) -> f64 {
    if s1 == s2 {
        return 0.0;
    }

    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    // For empty strings, return the length of the other string
    if len1 == 0 {
        return len2 as f64;
    } else if len2 == 0 {
        return len1 as f64;
    }

    // Convert to character vectors for correct Unicode handling
    let v1: Vec<char> = s1.chars().collect();
    let v2: Vec<char> = s2.chars().collect();

    // Create dynamic programming matrix
    let mut dp = vec![vec![0.0; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for i in 0..=len1 {
        dp[i][0] = i as f64;
    }
    for j in 0..=len2 {
        dp[0][j] = j as f64;
    }

    // Check for multi-character substitutions
    // We'll check various lengths of substrings up to a reasonable limit
    let max_substr_len = 5; // Limit to avoid excessive computation

    // Fill the dp matrix
    for i in 1..=len1 {
        for j in 1..=len2 {
            // Check for regular single character operations
            let deletion = dp[i - 1][j] + 1.0;
            let insertion = dp[i][j - 1] + 1.0;

            // Check for single character substitution
            let char_sub_cost = if v1[i - 1] == v2[j - 1] {
                0.0
            } else {
                // Use character comparison instead of substring comparison
                let c1 = v1[i - 1].to_string();
                let c2 = v2[j - 1].to_string();
                cost_map.get_substitution_cost(&c1, &c2)
            };
            let char_substitution = dp[i - 1][j - 1] + char_sub_cost;

            dp[i][j] = deletion.min(insertion).min(char_substitution);

            for len_a in 1..=max_substr_len.min(i) {
                for len_b in 1..=max_substr_len.min(j) {
                    if len_a == 1 && len_b == 1 {
                        continue; // Already handled above
                    }

                    // Get the substring ranges
                    let start_a = i - len_a;
                    let start_b = j - len_b;

                    // Extract substrings as strings
                    let substr_a = v1[start_a..i].iter().collect::<String>();
                    let substr_b = v2[start_b..j].iter().collect::<String>();

                    // Only apply the substitution if it's explicitly in our cost map
                    // Check if this specific substitution is in our cost map
                    let has_custom_cost = cost_map
                        .costs
                        .contains_key(&(substr_a.clone(), substr_b.clone()))
                        || cost_map
                            .costs
                            .contains_key(&(substr_b.clone(), substr_a.clone()));

                    if has_custom_cost {
                        // Get cost from the map
                        let sub_cost = cost_map.get_substitution_cost(&substr_a, &substr_b);

                        let previous_cost = dp[start_a][start_b];
                        let with_substitution = previous_cost + sub_cost;

                        // Update if this gives a better result
                        dp[i][j] = dp[i][j].min(with_substitution);
                    }
                }
            }
        }
    }

    dp[len1][len2]
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
            HashMap::from([(("a".to_string(), "b".to_string()), 0.1)]),
            1.0,
            true,
        );

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("abc", "bbc", &cost_map),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_multi_character_substitutions() {
        let cost_map = OcrCostMap::new(
            HashMap::from([(("h".to_string(), "In".to_string()), 0.2)]),
            1.0,
            true,
        );

        // Test that "hi" with "Ini" has a low cost due to the special substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("hi", "Ini", &cost_map),
            0.2, // Only the h->In substitution cost
            1e-9,
        );

        // Test another example
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("hello", "Inello", &cost_map),
            0.2, // Only the h->In substitution cost
            1e-9,
        );
    }

    #[test]
    fn test_multiple_substitutions_in_same_string() {
        let mut custom_costs = HashMap::new();
        custom_costs.insert(("h".to_string(), "In".to_string()), 0.2);
        custom_costs.insert(("l".to_string(), "1".to_string()), 0.3);
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        // Test multiple substitutions in the same string
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("hello", "Ine11o", &cost_map),
            0.8, // 0.2 for h->In and 0.3+0.3 for l->1 twice
            1e-9,
        );
    }

    #[test]
    fn test_overlapping_substitution_patterns() {
        let mut custom_costs = HashMap::new();
        custom_costs.insert(("rn".to_string(), "m".to_string()), 0.1); // common OCR confusion
        custom_costs.insert(("cl".to_string(), "d".to_string()), 0.2); // another common confusion
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        // Test the rn->m substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("corner", "comer", &cost_map),
            0.1,
            1e-9,
        );

        // Test the cl->d substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("class", "dass", &cost_map),
            0.2,
            1e-9,
        );
    }

    #[test]
    fn test_asymmetric_costs() {
        // Sometimes OCR errors aren't symmetric - going from reference to OCR
        // might have different likelihood than going from OCR to reference
        let mut custom_costs = HashMap::new();
        custom_costs.insert(("0".to_string(), "O".to_string()), 0.1); // 0->O is common
        custom_costs.insert(("O".to_string(), "0".to_string()), 0.5); // O->0 is less common
        let cost_map = OcrCostMap::new(custom_costs, 1.0, false); // asymmetric costs

        // Test 0->O substitution (lower cost)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("R0AD", "ROAD", &cost_map),
            0.1,
            1e-9,
        );

        // Test O->0 substitution (higher cost)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("rOad", "r0ad", &cost_map),
            0.5,
            1e-9,
        );
    }

    #[test]
    fn test_substitution_at_word_boundaries() {
        let mut custom_costs = HashMap::new();
        custom_costs.insert(("rn".to_string(), "m".to_string()), 0.1);
        let cost_map = OcrCostMap::new(custom_costs, 1.0, true);

        // Test substitution at start of word
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("rnat", "mat", &cost_map),
            0.1,
            1e-9,
        );

        // Test substitution at end of word
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("burn", "bum", &cost_map),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_empty_cost_map() {
        // Create a cost map with no custom substitution costs
        let cost_map = OcrCostMap::new(HashMap::new(), 1.0, true);

        // Test that "h" -> "In" costs 2.0 (1 deletion + 1 substitution) since there's no custom mapping
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("h", "In", &cost_map),
            2.0,
            1e-9,
        );

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_map("kitten", "sitting", &cost_map),
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
            custom_levenshtein_distance_with_cost_map("café", "coffee", &cost_map),
            4.0, // 4 edits required
            1e-9,
        );
    }
}
