use crate::cost_map::CostMap;
use crate::explanation::{EditOperation, Predecessor};
use crate::types::{SingleTokenKey, SubstitutionKey};

// --- Public Functions ---

pub fn custom_levenshtein_distance_with_cost_maps(
    source: &str,
    target: &str,
    substitution_cost_map: &CostMap<SubstitutionKey>,
    insertion_cost_map: &CostMap<SingleTokenKey>,
    deletion_cost_map: &CostMap<SingleTokenKey>,
) -> f64 {
    if source == target {
        return 0.0;
    }
    let mut processor = LevenshteinProcessor::new(
        source,
        target,
        substitution_cost_map,
        insertion_cost_map,
        deletion_cost_map,
        false,
    );
    processor.run();
    processor.distance()
}

pub fn explain_custom_levenshtein_distance(
    source: &str,
    target: &str,
    substitution_cost_map: &CostMap<SubstitutionKey>,
    insertion_cost_map: &CostMap<SingleTokenKey>,
    deletion_cost_map: &CostMap<SingleTokenKey>,
) -> Vec<EditOperation> {
    if source == target {
        return source
            .chars()
            .map(|c| EditOperation::Match {
                token: c.to_string(),
            })
            .collect();
    }
    let mut processor = LevenshteinProcessor::new(
        source,
        target,
        substitution_cost_map,
        insertion_cost_map,
        deletion_cost_map,
        true,
    );
    processor.run();
    processor.into_result()
}

// --- Algorithm Implementation ---

struct LevenshteinProcessor<'a> {
    source_chars: Vec<char>,
    target_chars: Vec<char>,
    sub_map: &'a CostMap<SubstitutionKey>,
    ins_map: &'a CostMap<SingleTokenKey>,
    del_map: &'a CostMap<SingleTokenKey>,
    dp: Vec<Vec<f64>>,
    predecessors: Option<Vec<Vec<Predecessor>>>,
}

impl<'a> LevenshteinProcessor<'a> {
    fn new(
        source: &str,
        target: &str,
        sub_map: &'a CostMap<SubstitutionKey>,
        ins_map: &'a CostMap<SingleTokenKey>,
        del_map: &'a CostMap<SingleTokenKey>,
        explain: bool,
    ) -> Self {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();
        let len_source = source_chars.len();
        let len_target = target_chars.len();

        let mut processor = Self {
            source_chars,
            target_chars,
            sub_map,
            ins_map,
            del_map,
            dp: vec![vec![0.0; len_target + 1]; len_source + 1],
            predecessors: if explain {
                Some(vec![
                    vec![Predecessor::None; len_target + 1];
                    len_source + 1
                ])
            } else {
                None
            },
        };
        processor.initialize();
        processor
    }

    /// Fill the DP table.
    fn run(&mut self) {
        for i in 1..=self.source_chars.len() {
            for j in 1..=self.target_chars.len() {
                self.compute_cell(i, j);
            }
        }
    }

    /// Get the final computed distance.
    fn distance(&self) -> f64 {
        self.dp[self.source_chars.len()][self.target_chars.len()]
    }

    /// Convert the computed predecessors into a sequence of edit operations.
    fn into_result(self) -> Vec<EditOperation> {
        match self.predecessors.as_ref() {
            Some(preds) => self.backtrack(preds),
            None => Vec::new(),
        }
    }

    fn record(&mut self, i: usize, j: usize, op: Predecessor) {
        if let Some(preds) = self.predecessors.as_mut() {
            preds[i][j] = op;
        }
    }

    /// Compute the cost for cell (i, j) in the DP table.
    fn compute_cell(&mut self, i: usize, j: usize) {
        let source_char_str = self.source_chars[i - 1].to_string();
        let target_char_str = self.target_chars[j - 1].to_string();

        let deletion_cost = self.dp[i - 1][j] + self.del_map.get_cost(&source_char_str);
        let insertion_cost = self.dp[i][j - 1] + self.ins_map.get_cost(&target_char_str);
        let sub_cost = self.sub_map.get_cost(&source_char_str, &target_char_str);
        let substitution_cost = self.dp[i - 1][j - 1] + sub_cost;

        // Check for exact match
        let match_cost = self.dp[i - 1][j - 1];
        let (mut min_cost, mut best_op) = if source_char_str == target_char_str {
            (match_cost, Predecessor::Match(1))
        } else {
            (substitution_cost, Predecessor::Substitute(1, 1))
        };

        if insertion_cost < min_cost {
            min_cost = insertion_cost;
            best_op = Predecessor::Insert(1);
        }
        if deletion_cost < min_cost {
            min_cost = deletion_cost;
            best_op = Predecessor::Delete(1);
        }

        self.dp[i][j] = min_cost;
        self.record(i, j, best_op);

        self.check_multi_char_ops(i, j);
    }

    /// Initialize the first row and column of the DP table.
    fn initialize(&mut self) {
        let len_source = self.source_chars.len();
        let len_target = self.target_chars.len();

        self.dp[0][0] = 0.0;
        // First row (insertions)
        for j in 1..=len_target {
            let char_str = self.target_chars[j - 1].to_string();
            self.dp[0][j] = self.dp[0][j - 1] + self.ins_map.get_cost(&char_str);
            self.record(0, j, Predecessor::Insert(1));

            let max_len = self.ins_map.max_token_length.min(j);
            for token_len in 2..=max_len {
                let token_start = j - token_len;
                let token: String = self.target_chars[token_start..j].iter().collect();
                if self.ins_map.has_key(&token) {
                    let new_cost = self.dp[0][token_start] + self.ins_map.get_cost(&token);
                    if new_cost < self.dp[0][j] {
                        self.dp[0][j] = new_cost;
                        self.record(0, j, Predecessor::Insert(token_len));
                    }
                }
            }
        }
        // First column (deletions)
        for i in 1..=len_source {
            let char_str = self.source_chars[i - 1].to_string();
            self.dp[i][0] = self.dp[i - 1][0] + self.del_map.get_cost(&char_str);
            self.record(i, 0, Predecessor::Delete(1));

            let max_len = self.del_map.max_token_length.min(i);
            for token_len in 2..=max_len {
                let token_start = i - token_len;
                let token: String = self.source_chars[token_start..i].iter().collect();
                if self.del_map.has_key(&token) {
                    let new_cost = self.dp[token_start][0] + self.del_map.get_cost(&token);
                    if new_cost < self.dp[i][0] {
                        self.dp[i][0] = new_cost;
                        self.record(i, 0, Predecessor::Delete(token_len));
                    }
                }
            }
        }
    }

    fn check_multi_char_substitutions(&mut self, i: usize, j: usize) {
        let max_source_len = self.sub_map.max_token_length.min(i);
        let max_target_len = self.sub_map.max_token_length.min(j);
        for source_len in 1..=max_source_len {
            for target_len in 1..=max_target_len {
                if source_len == 1 && target_len == 1 {
                    continue;
                }
                let source_start = i - source_len;
                let target_start = j - target_len;
                let source_substr: String = self.source_chars[source_start..i].iter().collect();
                let target_substr: String = self.target_chars[target_start..j].iter().collect();
                if self.sub_map.has_key(&source_substr, &target_substr) {
                    let new_cost = self.dp[source_start][target_start]
                        + self.sub_map.get_cost(&source_substr, &target_substr);
                    if new_cost < self.dp[i][j] {
                        self.dp[i][j] = new_cost;
                        self.record(i, j, Predecessor::Substitute(source_len, target_len));
                    }
                }
            }
        }
    }

    fn check_multi_char_insertions(&mut self, i: usize, j: usize) {
        let max_ins_len = self.ins_map.max_token_length.min(j);
        for token_len in 2..=max_ins_len {
            let token_start = j - token_len;
            let token: String = self.target_chars[token_start..j].iter().collect();
            if self.ins_map.has_key(&token) {
                let new_cost = self.dp[i][token_start] + self.ins_map.get_cost(&token);
                if new_cost < self.dp[i][j] {
                    self.dp[i][j] = new_cost;
                    self.record(i, j, Predecessor::Insert(token_len));
                }
            }
        }
    }

    fn check_multi_char_deletions(&mut self, i: usize, j: usize) {
        let max_del_len = self.del_map.max_token_length.min(i);
        for token_len in 2..=max_del_len {
            let token_start = i - token_len;
            let token: String = self.source_chars[token_start..i].iter().collect();
            if self.del_map.has_key(&token) {
                let new_cost = self.dp[token_start][j] + self.del_map.get_cost(&token);
                if new_cost < self.dp[i][j] {
                    self.dp[i][j] = new_cost;
                    self.record(i, j, Predecessor::Delete(token_len));
                }
            }
        }
    }

    /// Check for multi-character operations (substitutions, insertions, deletions).
    fn check_multi_char_ops(&mut self, i: usize, j: usize) {
        self.check_multi_char_substitutions(i, j);
        self.check_multi_char_insertions(i, j);
        self.check_multi_char_deletions(i, j);
    }

    /// Backtrack through the predecessors to construct the edit path.
    fn backtrack(&self, preds: &[Vec<Predecessor>]) -> Vec<EditOperation> {
        let mut path = Vec::new();
        let mut i = self.source_chars.len();
        let mut j = self.target_chars.len();

        while i > 0 || j > 0 {
            match preds[i][j] {
                Predecessor::Substitute(s_len, t_len) => {
                    let source_token: String = self.source_chars[i - s_len..i].iter().collect();
                    let target_token: String = self.target_chars[j - t_len..j].iter().collect();
                    if source_token != target_token {
                        let cost = self.sub_map.get_cost(&source_token, &target_token);
                        path.push(EditOperation::Substitute {
                            source: source_token,
                            target: target_token,
                            cost,
                        });
                    }
                    i -= s_len;
                    j -= t_len;
                }
                Predecessor::Insert(t_len) => {
                    let target_token: String = self.target_chars[j - t_len..j].iter().collect();
                    let cost = self.ins_map.get_cost(&target_token);
                    path.push(EditOperation::Insert {
                        target: target_token,
                        cost,
                    });
                    j -= t_len;
                }
                Predecessor::Delete(s_len) => {
                    let source_token: String = self.source_chars[i - s_len..i].iter().collect();
                    let cost = self.del_map.get_cost(&source_token);
                    path.push(EditOperation::Delete {
                        source: source_token,
                        cost,
                    });
                    i -= s_len;
                }
                Predecessor::Match(t_len) => {
                    let token: String = self.target_chars[j - t_len..j].iter().collect();
                    path.push(EditOperation::Match { token });
                    i -= t_len;
                    j -= t_len;
                }
                Predecessor::None => {
                    break;
                }
            }
        }
        path.reverse();
        path
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
    fn test_check_multi_char_ops_with_empty_maps() {
        let (sub_map, ins_map, del_map) = create_default_cost_maps();

        let mut processor =
            LevenshteinProcessor::new("abcd", "xyz", &sub_map, &ins_map, &del_map, true);

        // Simulate the DP state before the operation
        let original_dp_3_2 = processor.dp[3][2];
        processor.check_multi_char_ops(3, 2);

        // Verify that the DP value remains unchanged
        assert_approx_eq(processor.dp[3][2], original_dp_3_2, 1e-9);
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
