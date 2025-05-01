use crate::cost_map::CostMap;
use crate::types::{SingleTokenKey, SubstitutionKey};

/// Calculates custom Levenshtein distance between two strings using provided cost maps.
/// This implementation considers multi-character insertions, deletions, and substitutions.
pub fn custom_levenshtein_distance_with_cost_maps(
    source: &str,
    target: &str,
    substitution_cost_map: &CostMap<SubstitutionKey>,
    insertion_cost_map: &CostMap<SingleTokenKey>,
    deletion_cost_map: &CostMap<SingleTokenKey>,
) -> f64 {
    // Early exit if strings are identical
    if source == target {
        return 0.0;
    }

    let len_source = source.chars().count();
    let len_target = target.chars().count();

    // Convert to character vectors for correct Unicode handling
    let source_chars: Vec<char> = source.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();

    // The following algorithm is a modification of the Wagner-Fischer algorithm.

    // Create dynamic programming matrix
    let mut dp = vec![vec![0.0; len_target + 1]; len_source + 1];

    // Initialize first row - insertions to make empty string into target prefix
    dp[0][0] = 0.0;
    for j in 1..=len_target {
        // Start with single character insertion cost
        let char_str = target_chars[j - 1].to_string();
        dp[0][j] = dp[0][j - 1] + insertion_cost_map.get_cost(&char_str);

        // Check for cheaper multi-character insertions ending at j
        let max_len = insertion_cost_map.max_token_length.min(j);
        for token_len in 2..=max_len {
            let token_start = j - token_len;
            let token: String = target_chars[token_start..j].iter().collect();

            if insertion_cost_map.has_key(&token) {
                let insertion_cost = insertion_cost_map.get_cost(&token);
                let new_cost = dp[0][token_start] + insertion_cost;
                // Update if this multi-char insertion is cheaper
                if new_cost < dp[0][j] {
                    dp[0][j] = new_cost;
                }
            }
        }
    }

    // Initialize first column - deletions to make source prefix into empty string
    for i in 1..=len_source {
        // Start with single character deletion cost
        let char_str = source_chars[i - 1].to_string();
        dp[i][0] = dp[i - 1][0] + deletion_cost_map.get_cost(&char_str);

        // Check for cheaper multi-character deletions ending at i
        let max_len = deletion_cost_map.max_token_length.min(i);
        for token_len in 2..=max_len {
            let token_start = i - token_len;
            let token: String = source_chars[token_start..i].iter().collect();

            if deletion_cost_map.has_key(&token) {
                let deletion_cost = deletion_cost_map.get_cost(&token);
                let new_cost = dp[token_start][0] + deletion_cost;
                // Update if this multi-char deletion is cheaper
                if new_cost < dp[i][0] {
                    dp[i][0] = new_cost;
                }
            }
        }
    }

    // Handle the empty string cases using the initialized DP table values
    // (If source was empty, cost is inserting target; if target was empty, cost is deleting source)
    if len_source == 0 {
        return dp[0][len_target];
    }
    if len_target == 0 {
        return dp[len_source][0];
    }

    // Fill the rest of the dp matrix
    for i in 1..=len_source {
        for j in 1..=len_target {
            // Standard Levenshtein operations with custom costs (single character)
            let source_char_str = source_chars[i - 1].to_string();
            let target_char_str = target_chars[j - 1].to_string();

            let deletion_cost = deletion_cost_map.get_cost(&source_char_str);
            let insertion_cost = insertion_cost_map.get_cost(&target_char_str);

            let deletion = dp[i - 1][j] + deletion_cost;
            let insertion = dp[i][j - 1] + insertion_cost;

            // Single character substitution or match
            let char_sub_cost = if source_chars[i - 1] == target_chars[j - 1] {
                0.0
            } else {
                substitution_cost_map.get_cost(&source_char_str, &target_char_str)
            };
            let char_substitution = dp[i - 1][j - 1] + char_sub_cost;

            // Initialize current cell with the minimum of standard single-character operations
            dp[i][j] = deletion.min(insertion).min(char_substitution);

            // Consider multi-character substitutions ending at (i, j)
            check_multi_char_substitutions(
                i,
                j,
                &source_chars,
                &target_chars,
                &mut dp,
                substitution_cost_map,
            );

            // Consider multi-character insertions ending at j
            check_multi_char_insertions(i, j, &target_chars, &mut dp, insertion_cost_map);

            // Consider multi-character deletions ending at i
            check_multi_char_deletions(i, j, &source_chars, &mut dp, deletion_cost_map);
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
    cost_map: &CostMap<SubstitutionKey>,
) {
    // Determine the maximum lengths to check based on cost map and current position
    let max_source_len = cost_map.max_token_length.min(current_i);
    let max_target_len = cost_map.max_token_length.min(current_j);

    // Iterate through possible multi-character lengths for source and target
    for source_len in 1..=max_source_len {
        for target_len in 1..=max_target_len {
            // Skip single-char to single-char substitution as it's handled efficiently before
            if source_len == 1 && target_len == 1 {
                continue;
            }

            // Calculate substring start indices
            let source_start = current_i - source_len;
            let target_start = current_j - target_len;

            // Extract substrings
            let source_substr: String = source_chars[source_start..current_i].iter().collect();
            let target_substr: String = target_chars[target_start..current_j].iter().collect();

            // Check if a custom cost exists for this specific substitution pair
            if cost_map.has_key(&source_substr, &target_substr) {
                let sub_cost = cost_map.get_cost(&source_substr, &target_substr);
                // Calculate cost: cost to reach previous state + substitution cost
                let new_cost = dp[source_start][target_start] + sub_cost;

                // Update dp table if this path is cheaper
                if new_cost < dp[current_i][current_j] {
                    dp[current_i][current_j] = new_cost;
                }
            }
        }
    }
}

/// Helper function to check multi-character insertions for the Levenshtein algorithm
fn check_multi_char_insertions(
    current_i: usize,
    current_j: usize,
    target_chars: &[char],
    dp: &mut [Vec<f64>],
    cost_map: &CostMap<SingleTokenKey>,
) {
    // Determine the maximum insertion length to check
    let max_len = cost_map.max_token_length.min(current_j);

    // Check multi-character insertions of length 2 up to max_len
    // (Length 1 is handled by the standard insertion operation)
    for token_len in 2..=max_len {
        // Calculate token start index
        let token_start = current_j - token_len;

        // Extract token string
        let token: String = target_chars[token_start..current_j].iter().collect();

        // Check if a custom cost exists for inserting this token
        if cost_map.has_key(&token) {
            let insertion_cost = cost_map.get_cost(&token);
            // Calculate cost: cost to reach state before insertion + insertion cost
            let new_cost = dp[current_i][token_start] + insertion_cost;

            // Update dp table if this path is cheaper
            if new_cost < dp[current_i][current_j] {
                dp[current_i][current_j] = new_cost;
            }
        }
    }
}

/// Helper function to check multi-character deletions for the Levenshtein algorithm
fn check_multi_char_deletions(
    current_i: usize,
    current_j: usize,
    source_chars: &[char],
    dp: &mut [Vec<f64>],
    cost_map: &CostMap<SingleTokenKey>,
) {
    // Determine the maximum deletion length to check
    let max_len = cost_map.max_token_length.min(current_i);

    // Check multi-character deletions of length 2 up to max_len
    // (Length 1 is handled by the standard deletion operation)
    for token_len in 2..=max_len {
        // Calculate token start index
        let token_start = current_i - token_len;

        // Extract token string
        let token: String = source_chars[token_start..current_i].iter().collect();

        // Check if a custom cost exists for deleting this token
        if cost_map.has_key(&token) {
            let deletion_cost = cost_map.get_cost(&token);
            // Calculate cost: cost to reach state before deletion + deletion cost
            let new_cost = dp[token_start][current_j] + deletion_cost;

            // Update dp table if this path is cheaper
            if new_cost < dp[current_i][current_j] {
                dp[current_i][current_j] = new_cost;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{SingleTokenCostMap, SubstitutionCostMap};

    fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
        assert!(
            (a - b).abs() < epsilon,
            "Assertion failed: {} != {} within epsilon {}",
            a,
            b,
            epsilon
        );
    }

    // Helper function to create default cost maps for testing
    fn create_default_cost_maps() -> (
        CostMap<SubstitutionKey>,
        CostMap<SingleTokenKey>,
        CostMap<SingleTokenKey>,
    ) {
        let sub_map = CostMap::<SubstitutionKey>::new(SubstitutionCostMap::new(), 1.0, true);
        let ins_map = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);
        let del_map = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);
        (sub_map, ins_map, del_map)
    }

    #[test]
    fn test_custom_levenshtein_with_custom_sub_map() {
        let (_, ins_map, del_map) = create_default_cost_maps();

        // Create a custom substitution map with specific a->b cost
        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("a".to_string(), "b".to_string()), 0.1)]),
            1.0,
            true,
        );

        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("abc", "bbc", &sub_map, &ins_map, &del_map),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_mixed_custom_costs() {
        // Create cost maps
        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("a".to_string(), "b".to_string()), 0.1)]),
            1.0,
            true,
        );

        let ins_map =
            CostMap::<SingleTokenKey>::new(SingleTokenCostMap::from([("x".to_string(), 0.3)]), 1.0);

        let del_map =
            CostMap::<SingleTokenKey>::new(SingleTokenCostMap::from([("y".to_string(), 0.4)]), 1.0);

        // Test with all three maps: delete 'y' (0.4) + insert 'x' (0.3)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("aby", "abx", &sub_map, &ins_map, &del_map),
            0.7,
            1e-9,
        );

        // Test substitution: substitute 'a' with 'b' (0.1)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("abc", "bbc", &sub_map, &ins_map, &del_map),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_multi_character_substitutions() {
        let (_, ins_map, del_map) = create_default_cost_maps();

        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("h".to_string(), "In".to_string()), 0.2)]),
            1.0,
            true,
        );

        // Test that "hi" with "Ini" has a low cost due to the special substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("hi", "Ini", &sub_map, &ins_map, &del_map),
            0.2, // Only the h->In substitution cost
            1e-9,
        );

        // Test another example
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "hello", "Inello", &sub_map, &ins_map, &del_map,
            ),
            0.2, // Only the h->In substitution cost
            1e-9,
        );
    }

    #[test]
    fn test_multiple_substitutions_in_same_string() {
        let (_, ins_map, del_map) = create_default_cost_maps();

        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("h".to_string(), "In".to_string()), 0.2);
        custom_costs.insert(("l".to_string(), "1".to_string()), 0.3);
        let sub_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.0, true);

        // Test multiple substitutions in the same string
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "hello", "Ine11o", &sub_map, &ins_map, &del_map,
            ),
            0.8, // 0.2 for h->In and 0.3+0.3 for l->1 twice
            1e-9,
        );
    }

    #[test]
    fn test_overlapping_substitution_patterns() {
        let (_, ins_map, del_map) = create_default_cost_maps();

        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("rn".to_string(), "m".to_string()), 0.1); // common OCR confusion
        custom_costs.insert(("cl".to_string(), "d".to_string()), 0.2); // another common confusion
        let sub_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.0, true);

        // Test the rn->m substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "corner", "comer", &sub_map, &ins_map, &del_map,
            ),
            0.1,
            1e-9,
        );

        // Test the cl->d substitution
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "class", "dass", &sub_map, &ins_map, &del_map,
            ),
            0.2,
            1e-9,
        );
    }

    #[test]
    fn test_asymmetric_costs() {
        let (_, ins_map, del_map) = create_default_cost_maps();

        // Sometimes OCR errors aren't symmetric
        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("0".to_string(), "O".to_string()), 0.1); // 0->O is common
        custom_costs.insert(("O".to_string(), "0".to_string()), 0.5); // O->0 is less common
        let sub_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.0, false); // asymmetric costs

        // Test 0->O substitution (lower cost)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "R0AD", "ROAD", &sub_map, &ins_map, &del_map,
            ),
            0.1,
            1e-9,
        );

        // Test O->0 substitution (higher cost)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "rOad", "r0ad", &sub_map, &ins_map, &del_map,
            ),
            0.5,
            1e-9,
        );
    }

    #[test]
    fn test_substitution_at_word_boundaries() {
        let (_, ins_map, del_map) = create_default_cost_maps();

        let mut custom_costs = SubstitutionCostMap::new();
        custom_costs.insert(("rn".to_string(), "m".to_string()), 0.1);
        let sub_map = CostMap::<SubstitutionKey>::new(custom_costs, 1.0, true);

        // Test substitution at start of word
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("rnat", "mat", &sub_map, &ins_map, &del_map),
            0.1,
            1e-9,
        );

        // Test substitution at end of word
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("burn", "bum", &sub_map, &ins_map, &del_map),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_specific_custom_ins_del_costs() {
        let sub_map = CostMap::<SubstitutionKey>::new(SubstitutionCostMap::new(), 1.0, true);

        // Test with custom insertion cost
        let ins_map_custom = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("a".to_string(), 0.2), ("b".to_string(), 0.3)]),
            1.0,
        );
        let del_map_default = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);

        // Test insertion with custom cost: Insert 'a' with cost 0.2
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "bc",
                "abc",
                &sub_map,
                &ins_map_custom,
                &del_map_default,
            ),
            0.2,
            1e-9,
        );

        // Test with custom deletion cost
        let ins_map_default = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);
        let del_map_custom = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("a".to_string(), 0.4), ("c".to_string(), 0.5)]),
            1.0,
        );

        // Test deletion with custom cost: Delete 'a' with cost 0.4
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "abc",
                "bc",
                &sub_map,
                &ins_map_default,
                &del_map_custom,
            ),
            0.4,
            1e-9,
        );

        // Test with both custom insertion and deletion costs, forcing ins/del
        let ins_map_force =
            CostMap::<SingleTokenKey>::new(SingleTokenCostMap::from([("b".to_string(), 0.3)]), 1.0);
        let del_map_force =
            CostMap::<SingleTokenKey>::new(SingleTokenCostMap::from([("x".to_string(), 0.5)]), 1.0);

        // Create a substitution map with very high cost to force deletion+insertion
        let high_cost_sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::new(), // Empty map uses default cost
            2.0, // High default cost to ensure deletion+insertion is preferred
            true,
        );

        // Test combined operations: Delete 'x' (0.5) + insert 'b' (0.3)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "axc",
                "abc",
                &high_cost_sub_map,
                &ins_map_force,
                &del_map_force,
            ),
            0.8,
            1e-9,
        );
    }

    #[test]
    fn test_edge_cases() {
        let (sub_map, ins_map, del_map) = create_default_cost_maps();

        // Test empty strings: Empty strings have zero distance
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("", "", &sub_map, &ins_map, &del_map),
            0.0,
            1e-9,
        );

        // Test source empty, target not empty: Insert 'a', 'b', 'c' with default cost 1.0 each
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("", "abc", &sub_map, &ins_map, &del_map),
            3.0,
            1e-9,
        );

        // Test source not empty, target empty: Delete 'a', 'b', 'c' with default cost 1.0 each
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("abc", "", &sub_map, &ins_map, &del_map),
            3.0,
            1e-9,
        );

        // Test with custom insertion costs for empty source
        let custom_ins_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([
                ("a".to_string(), 0.2),
                ("b".to_string(), 0.3),
                ("c".to_string(), 0.4),
            ]),
            1.0,
        );

        // Test with custom insertion costs: Insert 'a' (0.2) + 'b' (0.3) + 'c' (0.4)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "",
                "abc",
                &sub_map,
                &custom_ins_map,
                &del_map,
            ),
            0.9,
            1e-9,
        );

        // Test with custom deletion costs for empty target
        let custom_del_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([
                ("a".to_string(), 0.5),
                ("b".to_string(), 0.6),
                ("c".to_string(), 0.7),
            ]),
            1.0,
        );

        // Test with custom deletion costs: Delete 'a' (0.5) + 'b' (0.6) + 'c' (0.7)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "abc",
                "",
                &sub_map,
                &ins_map,
                &custom_del_map,
            ),
            1.8,
            1e-9,
        );
    }

    #[test]
    fn test_overall_mixed_operations() {
        // Create maps with various custom costs
        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([
                (("a".to_string(), "A".to_string()), 0.1),
                (("b".to_string(), "B".to_string()), 0.2),
            ]),
            1.0,
            true,
        );

        let ins_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("x".to_string(), 0.3), ("y".to_string(), 0.4)]),
            1.0,
        );

        let del_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("m".to_string(), 0.5), ("n".to_string(), 0.6)]),
            1.0,
        );

        // Test with a mix of operations: Sub 'a'→'A' (0.1) + Sub 'b'→'B' (0.2) + delete 'm' (0.5) + delete 'n' (0.6) + insert 'x' (0.3) + insert 'y' (0.4)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "abmn", "ABxy", &sub_map, &ins_map, &del_map,
            ),
            2.1,
            1e-9,
        );
    }

    #[test]
    fn test_unicode_handling() {
        let (sub_map, ins_map, del_map) = create_default_cost_maps();

        // Test with Unicode characters: Substitute 'é' with 'e' with default cost 1.0
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "café", "cafe", &sub_map, &ins_map, &del_map,
            ),
            1.0,
            1e-9,
        );

        // Test with emoji: Delete ' ' and '😊' with default cost 1.0 each
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("hi 😊", "hi", &sub_map, &ins_map, &del_map),
            2.0,
            1e-9,
        );

        // Test with custom costs for Unicode
        let sub_map_unicode = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("e".to_string(), "é".to_string()), 0.1)]), // Custom e->é cost
            1.0,
            true,
        );
        let ins_map_unicode = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("é".to_string(), 0.2), ("😊".to_string(), 0.3)]),
            1.0,
        );
        let del_map_unicode = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("é".to_string(), 0.4), ("😊".to_string(), 0.5)]),
            1.0,
        );

        // Test substitution of Unicode with custom cost
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "cafe",
                "café",
                &sub_map_unicode,
                &ins_map_unicode,
                &del_map_unicode,
            ),
            0.1, // Custom substitution cost for 'e'->'é'
            1e-9,
        );

        // Test deletion of Unicode with custom cost
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "hi 😊",
                "hi",
                &sub_map,
                &ins_map_unicode,
                &del_map_unicode,
            ),
            1.5, // Delete ' ' (default 1.0) and '😊' (custom 0.5)
            1e-9,
        );
    }

    #[test]
    fn test_various_multi_char_substitutions() {
        // Test multi-character substitutions with different lengths
        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([
                (("th".to_string(), "T".to_string()), 0.2),    // 2 -> 1
                (("ing".to_string(), "in'".to_string()), 0.3), // 3 -> 3
                (("o".to_string(), "ou".to_string()), 0.1),    // 1 -> 2
            ]),
            1.0,
            true,
        );
        let (_, ins_map, del_map) = create_default_cost_maps();

        // Test 2-to-1 character substitution: Substitute "th" with "T" with cost 0.2
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("this", "Tis", &sub_map, &ins_map, &del_map),
            0.2,
            1e-9,
        );

        // Test 3-to-3 character substitution: Substitute "ing" with "in'" with cost 0.3
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "singing", "singin'", &sub_map, &ins_map, &del_map,
            ),
            0.3,
            1e-9,
        );

        // Test 1-to-2 character substitution: Substitute "o" with "ou" with cost 0.1
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("go", "gou", &sub_map, &ins_map, &del_map),
            0.1,
            1e-9,
        );

        // Test multiple multi-character substitutions: Sub "th"->"T" (0.2) + Sub "ing"->"in'" (0.3)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "thinking", "Tinkin'", &sub_map, &ins_map, &del_map,
            ),
            0.5,
            1e-9,
        );
    }

    #[test]
    fn test_multi_character_insertions_and_deletions() {
        let (sub_map, _, _) = create_default_cost_maps();

        let ins_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([
                ("ab".to_string(), 0.3),
                ("xyz".to_string(), 0.2),
                ("123".to_string(), 0.1),
                ("bc".to_string(), 0.25),
            ]),
            1.0,
        );

        let del_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([
                ("cd".to_string(), 0.4),
                ("ef".to_string(), 0.5),
                ("789".to_string(), 0.6),
                ("bc".to_string(), 0.35),
            ]),
            1.0,
        );

        // Test multi-character insertion: insert 'ab' (0.3)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("x", "xab", &sub_map, &ins_map, &del_map),
            0.3,
            1e-9,
        );

        // Test multi-character deletion: delete 'cd' (0.4)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("ycd", "y", &sub_map, &ins_map, &del_map),
            0.4,
            1e-9,
        );

        // Test both insertion and deletion: delete 'ef' (0.5) + insert 'ab' (0.3)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("aef", "aab", &sub_map, &ins_map, &del_map),
            0.8,
            1e-9,
        );

        // Test with longer token insertion: insert 'xyz' (0.2)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "test", "testxyz", &sub_map, &ins_map, &del_map,
            ),
            0.2,
            1e-9,
        );

        // Test with mixed operations: delete '789' (0.6) + insert 'xyz' (0.2)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "a789b", "axyzb", &sub_map, &ins_map, &del_map,
            ),
            0.8,
            1e-9,
        );

        // Test multi-character deletion "bc" at the beginning: delete 'bc' (cost 0.35)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("bcd", "d", &sub_map, &ins_map, &del_map),
            0.35,
            1e-9,
        );

        // Test multi-character insertion "bc" at the beginning: insert 'bc' (cost 0.25)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps("c", "bcc", &sub_map, &ins_map, &del_map),
            0.25,
            1e-9,
        );
    }

    #[test]
    fn test_fallback_to_default_costs_when_multi_char_sub_missing() {
        // Create cost maps with multi-character substitutions
        let sub_map_full = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([
                (("abc".to_string(), "xyz".to_string()), 0.1),
                (("de".to_string(), "uv".to_string()), 0.2),
            ]),
            1.0,
            true,
        );
        // Create map with only the 2-char substitution
        let sub_map_partial = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("de".to_string(), "uv".to_string()), 0.2)]),
            1.0,
            true,
        );
        let (sub_map_empty, ins_map, del_map) = create_default_cost_maps();

        // Test with full map (allows abc->xyz and de->uv): Sub "abc"->"xyz" (0.1) + Sub "de"->"uv" (0.2)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "abcde",
                "xyzuv",
                &sub_map_full,
                &ins_map,
                &del_map,
            ),
            0.3,
            1e-9,
        );

        // Test with partial map (does not allow abc->xyz, forces default): Sub a->x(1.0) + b->y(1.0) + c->z(1.0) + Sub "de"->"uv"(0.2)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "abcde",
                "xyzuv",
                &sub_map_partial,
                &ins_map,
                &del_map,
            ),
            3.2,
            1e-9,
        );

        // Test with empty map (only single character default operations): 5 * default sub cost (1.0)
        assert_approx_eq(
            custom_levenshtein_distance_with_cost_maps(
                "abcde",
                "xyzuv",
                &sub_map_empty,
                &ins_map,
                &del_map,
            ),
            5.0,
            1e-9,
        );
    }

    #[test]
    fn test_check_multi_char_insertions() {
        // Create a simple DP matrix
        let mut dp = vec![vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 0.0, 1.0, 2.0]];

        // Create a cost map with multi-character insertion
        let ins_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("ab".to_string(), 0.3)]),
            1.0,
        );

        // Target chars representing "abc"
        let target_chars = vec!['a', 'b', 'c'];

        // Initial dp[1][3] = 2.0. "bc" is not in the map, so it shouldn't change.
        check_multi_char_insertions(1, 3, &target_chars, &mut dp, &ins_map);
        assert_approx_eq(dp[1][3], 2.0, 1e-9); // Value should remain unchanged

        // Now let's check with a matching token
        // Target chars representing "cab", dp[1][3] still 2.0
        let target_chars2 = vec!['c', 'a', 'b'];
        let mut dp2 = vec![vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 0.0, 1.0, 2.0]]; // Reset dp state

        // Check position (1, 3) for target "cab". Token is "ab".
        // Should use dp[1][1] + cost("ab") = 0.0 + 0.3 = 0.3
        // Since 0.3 < dp2[1][3] (which is 2.0), it should update.
        check_multi_char_insertions(1, 3, &target_chars2, &mut dp2, &ins_map);
        assert_approx_eq(dp2[1][3], 0.3, 1e-9); // Value should be updated
    }

    #[test]
    fn test_check_multi_char_deletions() {
        // Create a simple DP matrix (4x3 matrix)
        let mut dp = vec![
            vec![0.0, 1.0, 2.0], // Row 0
            vec![1.0, 0.0, 1.0], // Row 1
            vec![2.0, 1.0, 0.0], // Row 2
            vec![3.0, 2.0, 1.0], // Row 3
        ];

        // Create a cost map with multi-character deletion
        let del_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("bc".to_string(), 0.4)]),
            1.0,
        );

        // Source chars representing "abcd"
        let source_chars = vec!['a', 'b', 'c', 'd'];

        // Check position (3, 1). Current dp[3][1] is 2.0.
        // Check token "bc" (source_chars[1..3]).
        // New cost would be dp[1][1] + cost("bc") = 0.0 + 0.4 = 0.4
        // Since 0.4 < 2.0, dp[3][1] should be updated to 0.4.
        let token_start = 3 - 2; // current_i = 3, token_len = 2
        let expected_new_cost = dp[token_start][1] + del_map.get_cost("bc");

        check_multi_char_deletions(3, 1, &source_chars, &mut dp, &del_map);

        // Check if the function updated the dp value correctly
        assert_approx_eq(dp[3][1], expected_new_cost, 1e-9); // Should be 0.4
        assert_approx_eq(dp[3][1], 0.4, 1e-9); // Explicit check
    }

    #[test]
    fn test_empty_cost_maps_for_helpers() {
        // Tests helper functions with empty cost maps
        let ins_map = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);
        let del_map = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);

        // Test multi-character operations directly on DP state
        let mut dp = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
            vec![3.0, 2.0, 1.0],
        ];
        let source_chars = vec!['a', 'b', 'c', 'd'];
        let target_chars = vec!['x', 'y', 'z'];

        // Store original values
        let original_dp_2_2 = dp[2][2];
        let original_dp_3_2 = dp[3][2];

        // Since the cost maps are empty, these should not find any multi-char keys and thus not modify dp values
        check_multi_char_insertions(2, 2, &target_chars, &mut dp, &ins_map);
        check_multi_char_deletions(3, 2, &source_chars, &mut dp, &del_map);

        // DP values should remain unchanged
        assert_approx_eq(dp[2][2], original_dp_2_2, 1e-9);
        assert_approx_eq(dp[3][2], original_dp_3_2, 1e-9);
    }

    #[test]
    fn test_main_function_with_multi_char_ins_del() {
        // Define source and target strings
        let source = "hello";
        let target = "helloxyz";

        // Create a custom insertion cost map
        let ins_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("xyz".to_string(), 0.2)]),
            1.0,
        );
        let sub_map = CostMap::<SubstitutionKey>::new(SubstitutionCostMap::new(), 1.0, true);
        let del_map = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);

        // Test multi-char insertion via main function
        let dist = custom_levenshtein_distance_with_cost_maps(
            source, target, &sub_map, &ins_map, &del_map,
        );
        assert_approx_eq(dist, 0.2, 1e-9); // Should be 0.2 (insert "xyz")

        // Now test a multi-character deletion via main function
        let source2 = "helloxyz";
        let target2 = "hello";

        // Create a custom deletion cost map
        let del_map2 = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("xyz".to_string(), 0.3)]),
            1.0,
        );
        // Use default insertion map for this test
        let ins_map2 = CostMap::<SingleTokenKey>::new(SingleTokenCostMap::new(), 1.0);

        let dist2 = custom_levenshtein_distance_with_cost_maps(
            source2, target2, &sub_map, &ins_map2, &del_map2,
        );
        assert_approx_eq(dist2, 0.3, 1e-9); // Should be 0.3 (delete "xyz")
    }
}
