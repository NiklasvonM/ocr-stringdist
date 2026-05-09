use crate::cost_map::CostMap;
use crate::explanation::{EditOperation, Predecessor};
use crate::types::{SingleTokenKey, SubstitutionKey};

pub(crate) fn custom_levenshtein_distance(
    source: &str,
    target: &str,
    sub: &CostMap<SubstitutionKey>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
) -> f64 {
    if source == target {
        return 0.0;
    }
    let mut processor = LevenshteinProcessor::new(source, target, sub, ins, del, false);
    processor.run();
    processor.distance()
}

pub(crate) fn explain_custom_levenshtein(
    source: &str,
    target: &str,
    sub: &CostMap<SubstitutionKey>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
) -> Vec<EditOperation> {
    if source == target {
        return source
            .chars()
            .map(|c| EditOperation::Match {
                token: c.to_string(),
            })
            .collect();
    }
    let mut processor = LevenshteinProcessor::new(source, target, sub, ins, del, true);
    processor.run();
    processor.into_result()
}

struct LevenshteinProcessor<'a> {
    source_chars: Vec<char>,
    target_chars: Vec<char>,
    sub: &'a CostMap<SubstitutionKey>,
    ins: &'a CostMap<SingleTokenKey>,
    del: &'a CostMap<SingleTokenKey>,
    dp: Vec<Vec<f64>>,
    predecessors: Option<Vec<Vec<Predecessor>>>,
    multi_char_ops: bool,
}

impl<'a> LevenshteinProcessor<'a> {
    fn new(
        source: &str,
        target: &str,
        sub: &'a CostMap<SubstitutionKey>,
        ins: &'a CostMap<SingleTokenKey>,
        del: &'a CostMap<SingleTokenKey>,
        explain: bool,
    ) -> Self {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();
        let len_source = source_chars.len();
        let len_target = target_chars.len();

        let mut processor = Self {
            source_chars,
            target_chars,
            multi_char_ops: sub.max_token_length() > 1
                || ins.max_token_length() > 1
                || del.max_token_length() > 1,
            sub,
            ins,
            del,
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

    fn run(&mut self) {
        for i in 1..=self.source_chars.len() {
            for j in 1..=self.target_chars.len() {
                self.compute_cell(i, j);
            }
        }
    }

    #[inline]
    fn distance(&self) -> f64 {
        self.dp[self.source_chars.len()][self.target_chars.len()]
    }

    fn into_result(self) -> Vec<EditOperation> {
        match self.predecessors.as_ref() {
            Some(preds) => self.backtrack(preds),
            None => Vec::new(),
        }
    }

    #[inline]
    fn record(&mut self, i: usize, j: usize, op: Predecessor) {
        if let Some(preds) = self.predecessors.as_mut() {
            preds[i][j] = op;
        }
    }

    #[inline]
    fn compute_cell(&mut self, i: usize, j: usize) {
        let source_char = self.source_chars[i - 1];
        let target_char = self.target_chars[j - 1];

        let source_char_str = source_char.to_string();
        let target_char_str = target_char.to_string();

        let deletion_cost = self.dp[i - 1][j] + self.del.get_cost(&source_char_str);
        let insertion_cost = self.dp[i][j - 1] + self.ins.get_cost(&target_char_str);
        let sub_cost = self.sub.get_cost(&source_char_str, &target_char_str);
        let substitution_cost = self.dp[i - 1][j - 1] + sub_cost;

        let match_cost = self.dp[i - 1][j - 1];
        let (mut min_cost, mut best_op) = if source_char == target_char {
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

        if self.multi_char_ops {
            self.check_multi_char_ops(i, j);
        }
    }

    fn initialize(&mut self) {
        let len_source = self.source_chars.len();
        let len_target = self.target_chars.len();

        self.dp[0][0] = 0.0;
        for j in 1..=len_target {
            let char_str = self.target_chars[j - 1].to_string();
            self.dp[0][j] = self.dp[0][j - 1] + self.ins.get_cost(&char_str);
            self.record(0, j, Predecessor::Insert(1));

            let max_len = self.ins.max_token_length().min(j);
            for token_len in 2..=max_len {
                let token_start = j - token_len;
                let token: String = self.target_chars[token_start..j].iter().collect();
                if self.ins.has_key(&token) {
                    let new_cost = self.dp[0][token_start] + self.ins.get_cost(&token);
                    if new_cost < self.dp[0][j] {
                        self.dp[0][j] = new_cost;
                        self.record(0, j, Predecessor::Insert(token_len));
                    }
                }
            }
        }
        for i in 1..=len_source {
            let char_str = self.source_chars[i - 1].to_string();
            self.dp[i][0] = self.dp[i - 1][0] + self.del.get_cost(&char_str);
            self.record(i, 0, Predecessor::Delete(1));

            let max_len = self.del.max_token_length().min(i);
            for token_len in 2..=max_len {
                let token_start = i - token_len;
                let token: String = self.source_chars[token_start..i].iter().collect();
                if self.del.has_key(&token) {
                    let new_cost = self.dp[token_start][0] + self.del.get_cost(&token);
                    if new_cost < self.dp[i][0] {
                        self.dp[i][0] = new_cost;
                        self.record(i, 0, Predecessor::Delete(token_len));
                    }
                }
            }
        }
    }

    fn check_multi_char_substitutions(&mut self, i: usize, j: usize) {
        let max_source_len = self.sub.max_token_length().min(i);
        let max_target_len = self.sub.max_token_length().min(j);
        for source_len in 1..=max_source_len {
            for target_len in 1..=max_target_len {
                if source_len == 1 && target_len == 1 {
                    continue;
                }
                let source_start = i - source_len;
                let target_start = j - target_len;
                let source_substr: String = self.source_chars[source_start..i].iter().collect();
                let target_substr: String = self.target_chars[target_start..j].iter().collect();
                if self.sub.has_key(&source_substr, &target_substr) {
                    let new_cost = self.dp[source_start][target_start]
                        + self.sub.get_cost(&source_substr, &target_substr);
                    if new_cost < self.dp[i][j] {
                        self.dp[i][j] = new_cost;
                        self.record(i, j, Predecessor::Substitute(source_len, target_len));
                    }
                }
            }
        }
    }

    fn check_multi_char_insertions(&mut self, i: usize, j: usize) {
        let max_ins_len = self.ins.max_token_length().min(j);
        for token_len in 2..=max_ins_len {
            let token_start = j - token_len;
            let token: String = self.target_chars[token_start..j].iter().collect();
            if self.ins.has_key(&token) {
                let new_cost = self.dp[i][token_start] + self.ins.get_cost(&token);
                if new_cost < self.dp[i][j] {
                    self.dp[i][j] = new_cost;
                    self.record(i, j, Predecessor::Insert(token_len));
                }
            }
        }
    }

    fn check_multi_char_deletions(&mut self, i: usize, j: usize) {
        let max_del_len = self.del.max_token_length().min(i);
        for token_len in 2..=max_del_len {
            let token_start = i - token_len;
            let token: String = self.source_chars[token_start..i].iter().collect();
            if self.del.has_key(&token) {
                let new_cost = self.dp[token_start][j] + self.del.get_cost(&token);
                if new_cost < self.dp[i][j] {
                    self.dp[i][j] = new_cost;
                    self.record(i, j, Predecessor::Delete(token_len));
                }
            }
        }
    }

    fn check_multi_char_ops(&mut self, i: usize, j: usize) {
        self.check_multi_char_substitutions(i, j);
        self.check_multi_char_insertions(i, j);
        self.check_multi_char_deletions(i, j);
    }

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
                        let cost = self.sub.get_cost(&source_token, &target_token);
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
                    let cost = self.ins.get_cost(&target_token);
                    path.push(EditOperation::Insert {
                        target: target_token,
                        cost,
                    });
                    j -= t_len;
                }
                Predecessor::Delete(s_len) => {
                    let source_token: String = self.source_chars[i - s_len..i].iter().collect();
                    let cost = self.del.get_cost(&source_token);
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

    fn calc_distance(
        source: &str,
        target: &str,
        sub_map: &CostMap<SubstitutionKey>,
        ins_map: &CostMap<SingleTokenKey>,
        del_map: &CostMap<SingleTokenKey>,
    ) -> f64 {
        custom_levenshtein_distance(source, target, sub_map, ins_map, del_map)
    }

    #[test]
    fn test_custom_levenshtein_with_custom_sub_map() {
        let (_, ins_map, del_map) = create_default_cost_maps();
        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("a".to_string(), "b".to_string()), 0.1)]),
            1.0,
            true,
        );
        assert_approx_eq(
            calc_distance("abc", "bbc", &sub_map, &ins_map, &del_map),
            0.1,
            1e-9,
        );
    }

    #[test]
    fn test_mixed_custom_costs() {
        let sub_map = CostMap::<SubstitutionKey>::new(
            SubstitutionCostMap::from([(("a".to_string(), "b".to_string()), 0.1)]),
            1.0,
            true,
        );
        let ins_map =
            CostMap::<SingleTokenKey>::new(SingleTokenCostMap::from([("x".to_string(), 0.3)]), 1.0);
        let del_map =
            CostMap::<SingleTokenKey>::new(SingleTokenCostMap::from([("y".to_string(), 0.4)]), 1.0);

        assert_approx_eq(
            calc_distance("aby", "abx", &sub_map, &ins_map, &del_map),
            0.7,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("abc", "bbc", &sub_map, &ins_map, &del_map),
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
        assert_approx_eq(
            calc_distance("hi", "Ini", &sub_map, &ins_map, &del_map),
            0.2,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("hello", "Inello", &sub_map, &ins_map, &del_map),
            0.2,
            1e-9,
        );
    }

    #[test]
    fn test_edge_cases() {
        let (sub_map, ins_map, del_map) = create_default_cost_maps();
        assert_approx_eq(
            calc_distance("", "", &sub_map, &ins_map, &del_map),
            0.0,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("", "abc", &sub_map, &ins_map, &del_map),
            3.0,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("abc", "", &sub_map, &ins_map, &del_map),
            3.0,
            1e-9,
        );
    }

    #[test]
    fn test_unicode_handling() {
        let (sub_map, ins_map, del_map) = create_default_cost_maps();
        assert_approx_eq(
            calc_distance("café", "cafe", &sub_map, &ins_map, &del_map),
            1.0,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("hi 😊", "hi", &sub_map, &ins_map, &del_map),
            2.0,
            1e-9,
        );
    }

    #[test]
    fn test_multi_character_insertions_and_deletions() {
        let (sub_map, _, _) = create_default_cost_maps();
        let ins_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("ab".to_string(), 0.3), ("xyz".to_string(), 0.2)]),
            1.0,
        );
        let del_map = CostMap::<SingleTokenKey>::new(
            SingleTokenCostMap::from([("cd".to_string(), 0.4), ("ef".to_string(), 0.5)]),
            1.0,
        );
        assert_approx_eq(
            calc_distance("x", "xab", &sub_map, &ins_map, &del_map),
            0.3,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("ycd", "y", &sub_map, &ins_map, &del_map),
            0.4,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("aef", "aab", &sub_map, &ins_map, &del_map),
            0.8,
            1e-9,
        );
        assert_approx_eq(
            calc_distance("test", "testxyz", &sub_map, &ins_map, &del_map),
            0.2,
            1e-9,
        );
    }
}
