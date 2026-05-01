//! Closed (transitive) edit-cost maps from a unified token graph.
//!
//! [`compute_closed_cost_maps`] seeds one graph from the raw sub/ins/del maps,
//! grows the node set by applying configured ins/del transformations until
//! fixpoint, closes the graph with Floyd-Warshall, and projects distances back
//! into three plain cost maps:
//!   - `dist[a][b]` for substitutions,
//!   - `dist[a][ε]` for deletions,
//!   - `dist[ε][b]` for insertions.
//!
//! Optional pruning removes generated substitutions that are already represented
//! by matches, insertions, deletions, and shorter substitutions.

use crate::cost_map::CostMap;
use crate::types::{SingleTokenCostMap, SingleTokenKey, SubstitutionCostMap, SubstitutionKey};
use crate::weighted_levenshtein::custom_levenshtein_distance;
use std::collections::{HashMap, HashSet};
use std::fmt;

// When the caller passes `None` for `max_node_length`, the cap is derived as
// `max(longest raw token across all maps) * MAX_NODE_LENGTH_MULTIPLIER`, with a
// floor of `MIN_DERIVED_NODE_LENGTH` so trivial single-char maps still leave
// headroom for chained compositions. Length is what makes the graph finite —
// without any cap, configurations like `ins("A")=0.1` would grow the node set
// without bound (A → AA → AAA → …).
const MAX_NODE_LENGTH_MULTIPLIER: usize = 2;
const MIN_DERIVED_NODE_LENGTH: usize = 4;

const REDUNDANT_SUBSTITUTION_EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitiveCostError {
    MatrixSizeOverflow { node_count: usize },
    MatrixAllocationFailed { node_count: usize, bytes: usize },
}

impl fmt::Display for TransitiveCostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixSizeOverflow { node_count } => write!(
                f,
                "transitive closure generated {node_count} graph nodes, too many to address in a dense matrix; pass a smaller max_node_length or reduce the number of configured insertion/deletion tokens"
            ),
            Self::MatrixAllocationFailed { node_count, bytes } => write!(
                f,
                "transitive closure generated {node_count} graph nodes and could not allocate a {bytes}-byte dense matrix; pass a smaller max_node_length or reduce the number of configured insertion/deletion tokens"
            ),
        }
    }
}

/// Interned identifier for a token graph node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct NodeId(u32);

impl NodeId {
    fn new(index: usize) -> Self {
        assert!(index < u32::MAX as usize, "too many token graph nodes");
        Self(index as u32)
    }

    #[inline]
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Dense square matrix stored in row-major order.
#[derive(Debug)]
struct Matrix<T> {
    width: usize,
    cells: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    fn try_filled(width: usize, value: T) -> Result<Self, TransitiveCostError> {
        let cell_count = width
            .checked_mul(width)
            .ok_or(TransitiveCostError::MatrixSizeOverflow { node_count: width })?;
        let bytes = cell_count
            .checked_mul(std::mem::size_of::<T>())
            .unwrap_or(usize::MAX);
        let mut cells = Vec::new();
        cells.try_reserve_exact(cell_count).map_err(|_| {
            TransitiveCostError::MatrixAllocationFailed {
                node_count: width,
                bytes,
            }
        })?;
        cells.resize(cell_count, value);
        Ok(Self { width, cells })
    }
}

impl<T> Matrix<T> {
    #[inline]
    fn idx(&self, row: NodeId, col: NodeId) -> usize {
        row.index() * self.width + col.index()
    }

    #[inline]
    fn get(&self, row: NodeId, col: NodeId) -> &T {
        &self.cells[self.idx(row, col)]
    }

    #[inline]
    fn set(&mut self, row: NodeId, col: NodeId, value: T) {
        let idx = self.idx(row, col);
        self.cells[idx] = value;
    }
}

/// Computes closed sub/ins/del cost maps via Floyd-Warshall on a unified graph.
///
/// `max_node_length` caps the length (in characters) of intermediate nodes the
/// growth phase may construct and of substrings expanded from raw tokens. Pass
/// `None` to derive a sensible default from the input
/// (`max raw-token length × 2`, floored at `MIN_DERIVED_NODE_LENGTH`); pass
/// `Some(n)` to override. The cap is what guarantees termination — without it,
/// configurations like `ins("A")` produce an infinite graph.
///
/// If `prune` is true, generated substitutions that the returned edit maps can
/// already express are removed from the substitution map.
pub fn compute_closed_cost_maps(
    sub: &CostMap<SubstitutionKey>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
    prune: bool,
    max_node_length: Option<usize>,
) -> Result<(SubstitutionCostMap, SingleTokenCostMap, SingleTokenCostMap), TransitiveCostError> {
    let max_node_length = max_node_length.unwrap_or_else(|| derive_max_node_length(sub, ins, del));
    let tokens = collect_nodes(sub, ins, del, max_node_length);
    let token_to_id: HashMap<&str, NodeId> = tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.as_str(), NodeId::new(i)))
        .collect();
    let epsilon_id = token_to_id[""];
    let dist = run_closure(&tokens, &token_to_id, epsilon_id, sub, ins, del)?;

    let closed_ins = project_single_token(&tokens, &token_to_id, &dist, ins, epsilon_id, true);
    let closed_del = project_single_token(&tokens, &token_to_id, &dist, del, epsilon_id, false);
    let closed_sub = project_substitutions(&tokens, &token_to_id, &dist, sub);
    let closed_sub = if prune {
        prune_redundant_substitutions(closed_sub, &closed_ins, &closed_del, sub, ins, del)
    } else {
        closed_sub
    };

    Ok((closed_sub, closed_ins, closed_del))
}

/// Default `max_node_length` derivation: twice the longest raw token across all
/// three maps, floored at `MIN_DERIVED_NODE_LENGTH`. The doubling leaves room
/// for compositions like `sub("AB","C") + sub("CC","D")` that need an
/// intermediate (`"ABAB" → "CC" → "D"`) longer than any single raw token.
fn derive_max_node_length(
    sub: &CostMap<SubstitutionKey>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
) -> usize {
    let max_raw = sub
        .costs
        .keys()
        .flat_map(|(s, t)| [s.chars().count(), t.chars().count()])
        .chain(ins.costs.keys().map(|k| k.chars().count()))
        .chain(del.costs.keys().map(|k| k.chars().count()))
        .max()
        .unwrap_or(0);
    (max_raw * MAX_NODE_LENGTH_MULTIPLIER).max(MIN_DERIVED_NODE_LENGTH)
}

/// Collects graph nodes. Includes raw tokens, all substrings of raw tokens
/// (capped at `max_node_length`), and strings reachable by iteratively applying
/// configured ins/del transformations until fixpoint. The length cap is the
/// only termination guarantee — without it, `ins("A")` alone would grow the
/// node set without bound.
fn collect_nodes(
    sub: &CostMap<SubstitutionKey>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
    max_node_length: usize,
) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
    tokens.insert(String::new()); // ε
    tokens.extend(ins.costs.keys().cloned());
    tokens.extend(del.costs.keys().cloned());
    for (s, t) in sub.costs.keys() {
        tokens.insert(s.clone());
        tokens.insert(t.clone());
    }

    let seeds: Vec<String> = tokens.iter().cloned().collect();
    for token in &seeds {
        for substring in substrings(token, max_node_length) {
            tokens.insert(substring);
        }
    }

    // Configured ins/del tokens are applied in BOTH directions to grow the set:
    // inserting them produces forward-direction successors, removing them
    // produces predecessors that lie one configured ins/del edge away. This is
    // what lets the closure bridge a user-provided source like "ADC" through
    // intermediate nodes "AC" and "ABC" to a configured target "Z".
    let single_op_tokens: Vec<String> = ins
        .costs
        .keys()
        .cloned()
        .chain(del.costs.keys().cloned())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    // Run growth to fixpoint. The length cap bounds the set of strings reachable
    // from the seeds, so this terminates after a finite number of rounds for any
    // input — typically only a handful, even for large maps.
    loop {
        let snapshot: Vec<String> = tokens.iter().cloned().collect();
        let prev_size = snapshot.len();

        for source in &snapshot {
            let source_len = source.chars().count();
            for op_token in &single_op_tokens {
                if source_len + op_token.chars().count() <= max_node_length {
                    for variant in insert_token_variants(source, op_token) {
                        tokens.insert(variant);
                    }
                }
                for variant in delete_token_variants(source, op_token) {
                    tokens.insert(variant);
                }
            }
        }

        if tokens.len() == prev_size {
            break;
        }
    }

    tokens.into_iter().collect()
}

fn substrings(token: &str, max_node_length: usize) -> Vec<String> {
    let char_count = token.chars().count();
    if char_count > max_node_length {
        return Vec::new();
    }
    let mut boundaries: Vec<usize> = token.char_indices().map(|(idx, _)| idx).collect();
    boundaries.push(token.len());

    let mut out = Vec::with_capacity(char_count * (char_count + 1) / 2);
    for start in 0..char_count {
        for end in (start + 1)..=char_count {
            out.push(token[boundaries[start]..boundaries[end]].to_owned());
        }
    }
    out
}

/// All strings obtainable by inserting `inserted` at a char boundary in `source`.
fn insert_token_variants(source: &str, inserted: &str) -> Vec<String> {
    source
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(source.len()))
        .map(|idx| {
            let mut target = String::with_capacity(source.len() + inserted.len());
            target.push_str(&source[..idx]);
            target.push_str(inserted);
            target.push_str(&source[idx..]);
            target
        })
        .collect()
}

/// All strings obtainable by deleting one exact `deleted` occurrence from `source`.
///
/// Enumerates every char-aligned offset where `deleted` matches, including
/// overlapping ones: `delete_token_variants("ABABA", "ABA")` yields both `"BA"`
/// (delete at 0) and `"AB"` (delete at 2). `str::match_indices` would only
/// return the non-overlapping leftmost match, so we walk char boundaries
/// manually instead.
fn delete_token_variants(source: &str, deleted: &str) -> Vec<String> {
    if deleted.is_empty() || deleted.len() > source.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (idx, _) in source.char_indices() {
        let end = idx + deleted.len();
        if end > source.len() {
            break;
        }
        if !source.is_char_boundary(end) {
            continue;
        }
        if &source[idx..end] != deleted {
            continue;
        }
        let mut target = String::with_capacity(source.len() - deleted.len());
        target.push_str(&source[..idx]);
        target.push_str(&source[end..]);
        out.push(target);
    }
    out
}

fn run_closure(
    tokens: &[String],
    token_to_id: &HashMap<&str, NodeId>,
    epsilon_id: NodeId,
    sub: &CostMap<SubstitutionKey>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
) -> Result<Matrix<f64>, TransitiveCostError> {
    let n = tokens.len();
    let mut dist = Matrix::try_filled(n, f64::INFINITY)?;

    for index in 0..n {
        let node = NodeId::new(index);
        dist.set(node, node, 0.0);
    }

    for ((source, target), &cost) in &sub.costs {
        if let (Some(&s), Some(&t)) = (
            token_to_id.get(source.as_str()),
            token_to_id.get(target.as_str()),
        ) {
            relax(&mut dist, s, t, cost);
        }
    }

    for (token, &cost) in &ins.costs {
        if let Some(&id) = token_to_id.get(token.as_str()) {
            relax(&mut dist, epsilon_id, id, cost);
        }
    }

    for (token, &cost) in &del.costs {
        if let Some(&id) = token_to_id.get(token.as_str()) {
            relax(&mut dist, id, epsilon_id, cost);
        }
    }

    seed_embedded_edges(tokens, token_to_id, ins, del, &mut dist);
    floyd_warshall(&mut dist);
    Ok(dist)
}

#[inline]
fn relax(dist: &mut Matrix<f64>, source: NodeId, target: NodeId, cost: f64) {
    if cost < *dist.get(source, target) {
        dist.set(source, target, cost);
    }
}

/// Adds direct edges between two existing nodes that differ by exactly one
/// configured insertion or deletion.
fn seed_embedded_edges(
    tokens: &[String],
    token_to_id: &HashMap<&str, NodeId>,
    ins: &CostMap<SingleTokenKey>,
    del: &CostMap<SingleTokenKey>,
    dist: &mut Matrix<f64>,
) {
    let insertions: Vec<(&str, f64)> = ins
        .costs
        .iter()
        .map(|(token, &cost)| (token.as_str(), cost))
        .collect();
    let deletions: Vec<(&str, f64)> = del
        .costs
        .iter()
        .map(|(token, &cost)| (token.as_str(), cost))
        .collect();

    for source in tokens {
        let source_id = token_to_id[source.as_str()];
        for &(inserted, cost) in &insertions {
            for variant in insert_token_variants(source, inserted) {
                if let Some(&target_id) = token_to_id.get(variant.as_str()) {
                    relax(dist, source_id, target_id, cost);
                }
            }
        }
        for &(deleted, cost) in &deletions {
            for variant in delete_token_variants(source, deleted) {
                if let Some(&target_id) = token_to_id.get(variant.as_str()) {
                    relax(dist, source_id, target_id, cost);
                }
            }
        }
    }
}

fn floyd_warshall(dist: &mut Matrix<f64>) {
    let n = dist.width;
    for via in 0..n {
        let via = NodeId::new(via);
        for source in 0..n {
            let source = NodeId::new(source);
            let source_to_via = *dist.get(source, via);
            if !source_to_via.is_finite() {
                continue;
            }
            for target in 0..n {
                let target = NodeId::new(target);
                let candidate = source_to_via + *dist.get(via, target);
                if candidate < *dist.get(source, target) {
                    dist.set(source, target, candidate);
                }
            }
        }
    }
}

fn project_substitutions(
    tokens: &[String],
    token_to_id: &HashMap<&str, NodeId>,
    dist: &Matrix<f64>,
    raw: &CostMap<SubstitutionKey>,
) -> SubstitutionCostMap {
    let default_cost = raw.default_cost();
    let mut closed = SubstitutionCostMap::new();

    for source in tokens {
        if source.is_empty() {
            continue;
        }
        let source_id = token_to_id[source.as_str()];
        for target in tokens {
            if target.is_empty() || source == target {
                continue;
            }
            let target_id = token_to_id[target.as_str()];
            let key = (source.clone(), target.clone());

            let raw_cost = raw.costs.get(&key).copied();
            let closure_cost = *dist.get(source_id, target_id);
            let direct = raw_cost.unwrap_or(default_cost);
            let effective = direct.min(closure_cost);

            if raw_cost.is_some() || effective < default_cost {
                closed.insert(key, effective);
            }
        }
    }

    closed
}

fn prune_redundant_substitutions(
    closed_sub: SubstitutionCostMap,
    closed_ins: &SingleTokenCostMap,
    closed_del: &SingleTokenCostMap,
    raw_sub: &CostMap<SubstitutionKey>,
    raw_ins: &CostMap<SingleTokenKey>,
    raw_del: &CostMap<SingleTokenKey>,
) -> SubstitutionCostMap {
    // Shortest-first iteration is safe: redundancy is monotone under removal of
    // redundant edges. If a short substitution is dropped because an ins/del/sub
    // alternative matches its cost, any longer substitution that relied on it in
    // the closure can fall back to the same alternative at the same total cost,
    // so its redundancy verdict is unchanged.
    let mut keys: Vec<SubstitutionKey> = closed_sub.keys().cloned().collect();
    keys.sort_by(|(source_a, target_a), (source_b, target_b)| {
        (
            source_a.chars().count() + target_a.chars().count(),
            source_a,
            target_a,
        )
            .cmp(&(
                source_b.chars().count() + target_b.chars().count(),
                source_b,
                target_b,
            ))
    });

    // `max_token_length` is computed once at construction and is not updated as
    // keys are removed below. The DP will scan longer windows than strictly
    // necessary; correctness is preserved because overestimating the window only
    // costs CPU.
    let mut sub_map = CostMap::<SubstitutionKey>::new(closed_sub, raw_sub.default_cost(), false);
    let ins_map = CostMap::<SingleTokenKey>::new(closed_ins.clone(), raw_ins.default_cost());
    let del_map = CostMap::<SingleTokenKey>::new(closed_del.clone(), raw_del.default_cost());

    for key in keys {
        if raw_sub.costs.contains_key(&key) {
            continue;
        }
        let Some(cost) = sub_map.costs.remove(&key) else {
            continue;
        };
        let alternative = custom_levenshtein_distance(&key.0, &key.1, &sub_map, &ins_map, &del_map);
        if alternative > cost + REDUNDANT_SUBSTITUTION_EPSILON {
            sub_map.costs.insert(key, cost);
        }
    }

    sub_map.costs
}

fn project_single_token(
    tokens: &[String],
    token_to_id: &HashMap<&str, NodeId>,
    dist: &Matrix<f64>,
    raw: &CostMap<SingleTokenKey>,
    epsilon_id: NodeId,
    is_insertion: bool,
) -> SingleTokenCostMap {
    let default_cost = raw.default_cost();
    let mut closed = SingleTokenCostMap::new();

    for token in tokens {
        if token.is_empty() {
            continue;
        }
        let id = token_to_id[token.as_str()];
        let raw_cost = raw.costs.get(token).copied();
        let closure_cost = if is_insertion {
            *dist.get(epsilon_id, id)
        } else {
            *dist.get(id, epsilon_id)
        };
        let direct = raw_cost.unwrap_or(default_cost);
        let effective = direct.min(closure_cost);

        if raw_cost.is_some() || effective < default_cost {
            closed.insert(token.clone(), effective);
        }
    }

    closed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn make_sub(
        pairs: &[((&str, &str), f64)],
        default: f64,
        symmetric: bool,
    ) -> CostMap<SubstitutionKey> {
        let mut map = SubstitutionCostMap::new();
        for ((s, t), c) in pairs {
            map.insert(((*s).to_string(), (*t).to_string()), *c);
        }
        CostMap::<SubstitutionKey>::new(map, default, symmetric)
    }

    fn make_single(pairs: &[(&str, f64)], default: f64) -> CostMap<SingleTokenKey> {
        let mut map = SingleTokenCostMap::new();
        for (k, v) in pairs {
            map.insert((*k).to_string(), *v);
        }
        CostMap::<SingleTokenKey>::new(map, default)
    }

    fn compute_closed_cost_maps(
        sub: &CostMap<SubstitutionKey>,
        ins: &CostMap<SingleTokenKey>,
        del: &CostMap<SingleTokenKey>,
        prune: bool,
        max_node_length: Option<usize>,
    ) -> (SubstitutionCostMap, SingleTokenCostMap, SingleTokenCostMap) {
        super::compute_closed_cost_maps(sub, ins, del, prune, max_node_length).unwrap()
    }

    #[test]
    fn closure_finds_substitution_chain() {
        let sub = make_sub(&[(("a", "b"), 0.1), (("b", "c"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_sub[&("a".to_string(), "c".to_string())], 0.2));
    }

    #[test]
    fn closure_finds_deletion_chain() {
        let sub = make_sub(&[(("6", "G"), 0.5)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[("G", 0.01)], 1.0);
        let (_, _, closed_del) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_del["6"], 0.51));
    }

    #[test]
    fn closure_finds_insertion_chain() {
        let sub = make_sub(&[(("x", "y"), 0.2)], 1.0, false);
        let ins = make_single(&[("x", 0.1)], 1.0);
        let del = make_single(&[], 1.0);
        let (_, closed_ins, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_ins["y"], 0.3));
    }

    #[test]
    fn closure_finds_repeated_insertion_substitution() {
        // A -> AA -> AAA -> B at cost 0.2 + 0.2 + 0.1 = 0.5
        let sub = make_sub(&[(("AAA", "B"), 0.1)], 1.0, true);
        let ins = make_single(&[("A", 0.2)], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_sub[&("A".to_string(), "B".to_string())], 0.5));
    }

    #[test]
    fn closure_prunes_substitutions_represented_by_insertions() {
        let sub = make_sub(&[(("AAA", "B"), 0.1), (("A", "B"), 0.6)], 1.0, true);
        let ins = make_single(&[("A", 0.2)], 1.0);
        let del = make_single(&[], 1.0);

        let (closed_sub, closed_ins, _) = compute_closed_cost_maps(&sub, &ins, &del, true, None);

        assert!(approx(closed_ins["A"], 0.2));
        assert!(approx(closed_sub[&("A".to_string(), "B".to_string())], 0.5));
        assert!(!closed_sub.contains_key(&("AA".to_string(), "AAA".to_string())));
        assert!(!closed_sub.contains_key(&("B".to_string(), "AA".to_string())));
    }

    #[test]
    fn closure_preserves_raw_substitutions_even_when_redundant() {
        let sub = make_sub(&[(("AA", "AAA"), 0.2)], 1.0, false);
        let ins = make_single(&[("A", 0.2)], 1.0);
        let del = make_single(&[], 1.0);

        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, true, None);

        assert!(approx(
            closed_sub[&("AA".to_string(), "AAA".to_string())],
            0.2
        ));
    }

    #[test]
    fn closure_composes_del_ins_sub() {
        // ADC -> AC -> ABC -> Z at cost 0.1 + 0.1 + 0.1 = 0.3
        let sub = make_sub(&[(("ABC", "Z"), 0.1)], 1.0, true);
        let ins = make_single(&[("B", 0.1)], 1.0);
        let del = make_single(&[("D", 0.1)], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(
            closed_sub[&("ADC".to_string(), "Z".to_string())],
            0.3
        ));
    }

    #[test]
    fn closure_preserves_direct_when_chain_more_expensive() {
        let sub = make_sub(&[(("6", "G"), 0.5)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[("6", 0.2), ("G", 0.01)], 1.0);
        let (_, _, closed_del) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_del["6"], 0.2));
    }

    // --- helper-function tests --------------------------------------------------

    fn sorted(mut v: Vec<String>) -> Vec<String> {
        v.sort();
        v
    }

    #[test]
    fn substrings_includes_all_contiguous_slices() {
        let mut got = sorted(substrings("abc", 8));
        got.dedup();
        assert_eq!(got, vec!["a", "ab", "abc", "b", "bc", "c"]);
    }

    #[test]
    fn substrings_empty_for_empty_input() {
        assert!(substrings("", 8).is_empty());
    }

    #[test]
    fn substrings_handles_unicode() {
        let got = sorted(substrings("café", 8));
        // Chars are 'c', 'a', 'f', 'é'; substrings are all contiguous slices on
        // char boundaries.
        assert!(got.contains(&"é".to_string()));
        assert!(got.contains(&"fé".to_string()));
        assert!(got.contains(&"café".to_string()));
        // No partial-byte slice should appear.
        for s in &got {
            assert!(s.is_char_boundary(0) && s.is_char_boundary(s.len()));
        }
    }

    #[test]
    fn substrings_skipped_when_token_exceeds_cap() {
        // A token longer than the cap returns no substrings.
        let long_token = "A".repeat(9);
        assert!(substrings(&long_token, 8).is_empty());
    }

    #[test]
    fn insert_token_variants_inserts_at_every_boundary() {
        let mut got = sorted(insert_token_variants("ab", "X"));
        assert_eq!(got, vec!["Xab", "aXb", "abX"]);
        got.dedup();
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn insert_token_variants_into_empty() {
        assert_eq!(insert_token_variants("", "AB"), vec!["AB".to_string()]);
    }

    #[test]
    fn insert_token_variants_handles_unicode() {
        let got = sorted(insert_token_variants("é", "X"));
        assert_eq!(got, vec!["Xé".to_string(), "éX".to_string()]);
    }

    #[test]
    fn delete_token_variants_enumerates_overlapping_matches() {
        let mut variants = delete_token_variants("ABABA", "ABA");
        variants.sort();
        assert_eq!(variants, vec!["AB".to_string(), "BA".to_string()]);
    }

    #[test]
    fn delete_token_variants_no_match() {
        assert!(delete_token_variants("xyz", "ABA").is_empty());
    }

    #[test]
    fn delete_token_variants_full_match_yields_empty_string() {
        assert_eq!(delete_token_variants("ABA", "ABA"), vec![String::new()]);
    }

    #[test]
    fn delete_token_variants_deleted_longer_than_source() {
        assert!(delete_token_variants("AB", "ABCDE").is_empty());
    }

    #[test]
    fn delete_token_variants_empty_deleted_string() {
        // Deleting nothing produces no graph edges (we'd otherwise loop on ε).
        assert!(delete_token_variants("ABC", "").is_empty());
    }

    #[test]
    fn delete_token_variants_single_char_repeated() {
        // For single chars, every position matches; we get one variant per
        // occurrence (deduplicated downstream by the HashSet).
        let mut got = delete_token_variants("AAAA", "A");
        got.sort();
        // All four deletions produce "AAA".
        assert_eq!(got, vec!["AAA"; 4]);
    }

    #[test]
    fn delete_token_variants_handles_unicode() {
        // "café" minus "fé" yields "ca". A non-overlapping single-byte search
        // at byte index 2 ('f') would be wrong if it weren't gated by
        // is_char_boundary — confirm the unicode path works end-to-end.
        let got = delete_token_variants("café", "fé");
        assert_eq!(got, vec!["ca".to_string()]);
    }

    #[test]
    fn delete_token_variants_unicode_overlap() {
        // "🙂🙃🙂🙃🙂" minus "🙂🙃🙂": matches at char positions 0 and 2 (both
        // overlap). Expect both variants.
        let mut got = delete_token_variants("🙂🙃🙂🙃🙂", "🙂🙃🙂");
        got.sort();
        assert_eq!(got, vec!["🙂🙃".to_string(), "🙃🙂".to_string()]);
    }

    // --- closure correctness ----------------------------------------------------

    #[test]
    fn closure_finds_chain_through_overlapping_deletion() {
        // ABABA can become AB by deleting the "ABA" starting at index 2.
        // sub("AB","X")=0.1 then closes to sub("ABABA","X")=0.2.
        let sub = make_sub(&[(("AB", "X"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[("ABA", 0.1)], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(
            closed_sub[&("ABABA".to_string(), "X".to_string())],
            0.2
        ));
    }

    #[test]
    fn closure_emits_overlapping_deletion_intermediate_edge() {
        // Pin the projected substitution that the previous match_indices-based
        // implementation silently dropped. Without overlap support, this entry
        // would be missing from closed_sub.
        let sub = make_sub(&[(("AB", "X"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[("ABA", 0.1)], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        let expected_key = ("ABABA".to_string(), "AB".to_string());
        assert!(
            closed_sub.contains_key(&expected_key),
            "expected overlapping-delete edge to be projected as a substitution"
        );
        assert!(approx(closed_sub[&expected_key], 0.1));
    }

    #[test]
    fn closure_handles_empty_cost_maps() {
        let sub = make_sub(&[], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, closed_ins, closed_del) =
            compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(closed_sub.is_empty());
        assert!(closed_ins.is_empty());
        assert!(closed_del.is_empty());
    }

    #[test]
    fn closure_does_not_emit_pairs_above_default() {
        // No raw entry, no chain — projection should not insert anything that
        // would just equal the default cost.
        let sub = make_sub(&[(("a", "b"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        // ("a","z") has no path, falls back to default — must not be inserted.
        assert!(!closed_sub.contains_key(&("a".to_string(), "z".to_string())));
    }

    #[test]
    fn closure_keeps_raw_substitution_even_when_above_default() {
        // User-set 2.0 > default 1.0. The raw entry must survive so that the
        // round-trip preserves the user's explicit intent.
        let sub = make_sub(&[(("a", "b"), 2.0)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        let key = ("a".to_string(), "b".to_string());
        assert!(closed_sub.contains_key(&key));
        // No cheaper chain exists, so the raw 2.0 stays.
        assert!(approx(closed_sub[&key], 2.0));
    }

    #[test]
    fn closure_skips_self_substitution() {
        // (a,a) is never a useful substitution; projection must skip it.
        let sub = make_sub(&[(("a", "b"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        for ((s, t), _) in &closed_sub {
            assert_ne!(s, t, "self-substitution leaked into closed map");
        }
    }

    #[test]
    fn closure_symmetric_propagates_both_directions() {
        // Symmetric chain: (a,b)+ (b,c) at 0.1 each gives (a,c) and (c,a) both at 0.2.
        let sub = make_sub(&[(("a", "b"), 0.1), (("b", "c"), 0.1)], 1.0, true);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_sub[&("a".to_string(), "c".to_string())], 0.2));
        assert!(approx(closed_sub[&("c".to_string(), "a".to_string())], 0.2));
    }

    #[test]
    fn closure_asymmetric_does_not_invent_reverse() {
        // Asymmetric: only (a,b) is given. (b,a) must not appear because no
        // path exists in that direction.
        let sub = make_sub(&[(("a", "b"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(closed_sub.contains_key(&("a".to_string(), "b".to_string())));
        assert!(!closed_sub.contains_key(&("b".to_string(), "a".to_string())));
    }

    #[test]
    fn closure_materializes_del_then_ins_as_substitution() {
        // del("y")=0.1 + ins("x")=0.2 should fold into an effective sub("y","x")=0.3.
        let sub = make_sub(&[], 1.0, false);
        let ins = make_single(&[("x", 0.2)], 1.0);
        let del = make_single(&[("y", 0.1)], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_sub[&("y".to_string(), "x".to_string())], 0.3));
    }

    #[test]
    fn closure_finds_long_substitution_chain() {
        // a -> b -> c -> d at 0.1 + 0.1 + 0.1 = 0.3.
        let sub = make_sub(
            &[(("a", "b"), 0.1), (("b", "c"), 0.1), (("c", "d"), 0.1)],
            1.0,
            false,
        );
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_sub[&("a".to_string(), "d".to_string())], 0.3));
    }

    #[test]
    fn closure_handles_zero_cost_edges() {
        // Zero-cost ins should still appear in closed map.
        let sub = make_sub(&[(("x", "y"), 0.0)], 1.0, false);
        let ins = make_single(&[("x", 0.0)], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, closed_ins, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_ins["y"], 0.0));
        assert!(approx(closed_sub[&("x".to_string(), "y".to_string())], 0.0));
    }

    #[test]
    fn closure_handles_unicode_tokens() {
        // Same chain shape as the basic ASCII test, but with non-ASCII chars.
        let sub = make_sub(&[(("é", "ê"), 0.1), (("ê", "è"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        assert!(approx(closed_sub[&("é".to_string(), "è".to_string())], 0.2));
    }

    #[test]
    fn matrix_allocation_checks_size_overflow() {
        let err = Matrix::<f64>::try_filled(usize::MAX, 0.0).unwrap_err();
        assert!(matches!(
            err,
            TransitiveCostError::MatrixSizeOverflow {
                node_count: usize::MAX
            }
        ));
    }

    // --- pruning ----------------------------------------------------------------

    #[test]
    fn pruning_on_empty_input_is_noop() {
        let sub = make_sub(&[], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, true, None);
        assert!(closed_sub.is_empty());
    }

    #[test]
    fn pruning_keeps_strictly_cheaper_substitution() {
        // sub(a,b)=0.1 with ins(a)=0.5, del(b)=0.5: alternative path through ε
        // costs 1.0, so the substitution is strictly cheaper and must be kept.
        let sub = make_sub(&[(("a", "b"), 0.1)], 1.0, false);
        let ins = make_single(&[("a", 0.5)], 1.0);
        let del = make_single(&[("b", 0.5)], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, true, None);
        assert!(closed_sub.contains_key(&("a".to_string(), "b".to_string())));
    }

    #[test]
    fn pruning_preserves_distances_for_chains() {
        // Closure-then-pruning must not change distances on the strings the
        // chains describe. We verify by running the DP with the closed maps
        // before and after pruning and comparing.
        let sub = make_sub(&[(("AAA", "B"), 0.1)], 1.0, true);
        let ins = make_single(&[("A", 0.2)], 1.0);
        let del = make_single(&[], 1.0);
        let (sub_unpruned, ins_unpruned, del_unpruned) =
            compute_closed_cost_maps(&sub, &ins, &del, false, None);
        let (sub_pruned, ins_pruned, del_pruned) =
            compute_closed_cost_maps(&sub, &ins, &del, true, None);

        let unpruned_subs = CostMap::<SubstitutionKey>::new(sub_unpruned, 1.0, false);
        let unpruned_ins = CostMap::<SingleTokenKey>::new(ins_unpruned, 1.0);
        let unpruned_del = CostMap::<SingleTokenKey>::new(del_unpruned, 1.0);
        let pruned_subs = CostMap::<SubstitutionKey>::new(sub_pruned, 1.0, false);
        let pruned_ins = CostMap::<SingleTokenKey>::new(ins_pruned, 1.0);
        let pruned_del = CostMap::<SingleTokenKey>::new(del_pruned, 1.0);

        for (s, t) in [
            ("A", "B"),
            ("AA", "B"),
            ("AAA", "B"),
            ("B", "A"),
            ("AA", "AAA"),
        ] {
            let unpruned =
                custom_levenshtein_distance(s, t, &unpruned_subs, &unpruned_ins, &unpruned_del);
            let pruned = custom_levenshtein_distance(s, t, &pruned_subs, &pruned_ins, &pruned_del);
            assert!(
                approx(unpruned, pruned),
                "distance({s:?},{t:?}) drifted after pruning: {unpruned} vs {pruned}"
            );
        }
    }

    #[test]
    fn pruning_raw_entries_are_never_dropped() {
        // ("AA","AAA") at 0.2 is exactly representable as ins("A"), so it would
        // be redundant if generated. But it's user-provided, so prune must keep it.
        let sub = make_sub(&[(("AA", "AAA"), 0.2)], 1.0, false);
        let ins = make_single(&[("A", 0.2)], 1.0);
        let del = make_single(&[], 1.0);
        let (closed_sub, _, _) = compute_closed_cost_maps(&sub, &ins, &del, true, None);
        assert!(closed_sub.contains_key(&("AA".to_string(), "AAA".to_string())));
    }

    #[test]
    fn closure_idempotent_pure_substitution() {
        // Pure substitution chains converge in one closure round: round 1 adds
        // (a,c) at 0.2; round 2 sees it directly, finds the same path through b.
        let sub = make_sub(&[(("a", "b"), 0.1), (("b", "c"), 0.1)], 1.0, false);
        let ins = make_single(&[], 1.0);
        let del = make_single(&[], 1.0);
        let (s1, i1, d1) = compute_closed_cost_maps(&sub, &ins, &del, false, None);
        let sub2 = CostMap::<SubstitutionKey>::new(s1.clone(), 1.0, false);
        let ins2 = CostMap::<SingleTokenKey>::new(i1.clone(), 1.0);
        let del2 = CostMap::<SingleTokenKey>::new(d1.clone(), 1.0);
        let (s2, i2, d2) = compute_closed_cost_maps(&sub2, &ins2, &del2, false, None);
        assert_eq!(s1, s2);
        assert_eq!(i1, i2);
        assert_eq!(d1, d2);
    }
}
