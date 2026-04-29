//! Transitive closure of edit costs via Dijkstra.
//!
//! This module precomputes the globally cheapest cost for every operation via
//! Dijkstra, so the DP uses optimal paths without per-cell search.
//!
//! Results are cached in [`EffectiveSingleTokenCosts`] (del/ins) and
//! [`EffectiveSubstitutionCosts`] (sub), computed once at construction.

use crate::cost_map::CostMap;
use crate::types::{SingleTokenKey, SubstitutionKey};
use std::collections::{BinaryHeap, HashMap, HashSet};

type DistMap = HashMap<String, f64>;
type PrevMap = HashMap<String, Option<(String, f64)>>;

// Public types

/// How the minimum effective cost for a single-token operation was achieved.
#[derive(Clone, Debug, PartialEq)]
pub enum EffectiveOpChain {
    /// The token is deleted / inserted directly at its mapped or default cost.
    Direct,

    /// A cheaper path exists through a chain of substitutions.
    ///
    /// `steps` holds the substitution edges in forward order:
    /// - **Deletion**: `(source -> x1, c1), (x1 -> x2, c2), …, (xn-1 -> terminal, cn)` then
    ///   `delete(terminal)` at `terminal_cost`.
    /// - **Insertion**: `insert(initial)` at `terminal_cost`, then
    ///   `(initial -> x1, c1), …, (xn -> target, cn)`.
    Via {
        /// Substitution edges `(from, to, cost)` in forward order.
        steps: Vec<(String, String, f64)>,
        /// Cost of the direct `del` / `ins` at the terminal (deletion) or initial
        /// (insertion) node of the chain.
        terminal_cost: f64,
    },
}

/// Precomputed effective single-token operation costs (deletion or insertion).
///
/// Replaces the raw [`CostMap<SingleTokenKey>`] inside the DP so the algorithm
/// automatically uses the globally cheapest edit path.
#[derive(Debug)]
pub struct EffectiveSingleTokenCosts {
    entries: HashMap<String, (f64, EffectiveOpChain)>,
    default_cost: f64,
    pub max_token_length: usize,
}

impl EffectiveSingleTokenCosts {
    #[inline]
    pub fn get_cost(&self, token: &str) -> f64 {
        self.entries
            .get(token)
            .map(|(c, _)| *c)
            .unwrap_or(self.default_cost)
    }

    #[inline]
    pub fn get_chain(&self, token: &str) -> EffectiveOpChain {
        self.entries
            .get(token)
            .map(|(_, ch)| ch.clone())
            .unwrap_or(EffectiveOpChain::Direct)
    }

    #[inline]
    pub fn has_key(&self, token: &str) -> bool {
        self.entries.contains_key(token)
    }
}

/// How the minimum effective substitution cost was achieved.
#[derive(Clone, Debug, PartialEq)]
pub enum EffectiveSubChain {
    /// Direct substitution at the mapped cost.
    Direct,

    /// A cheaper path was found through a chain of substitutions.
    ///
    /// `steps` holds edges `(from, to, cost)` in forward order, covering the
    /// full path from the source token to the target token.
    Via {
        /// Substitution edges `(from, to, cost)` in forward order.
        steps: Vec<(String, String, f64)>,
    },
}

/// Precomputed effective substitution costs (all-pairs shortest paths).
///
/// Replaces the raw [`CostMap<SubstitutionKey>`] inside the DP so the algorithm
/// automatically uses the globally cheapest substitution path.
#[derive(Debug)]
pub struct EffectiveSubstitutionCosts {
    entries: HashMap<(String, String), (f64, EffectiveSubChain)>,
    default_cost: f64,
    pub max_token_length: usize,
}

impl EffectiveSubstitutionCosts {
    #[inline]
    pub fn get_cost(&self, source: &str, target: &str) -> f64 {
        self.entries
            .get(&(source.to_owned(), target.to_owned()))
            .map(|(c, _)| *c)
            .unwrap_or(self.default_cost)
    }

    #[inline]
    pub fn get_chain(&self, source: &str, target: &str) -> EffectiveSubChain {
        self.entries
            .get(&(source.to_owned(), target.to_owned()))
            .map(|(_, ch)| ch.clone())
            .unwrap_or(EffectiveSubChain::Direct)
    }

    #[inline]
    pub fn has_key(&self, source: &str, target: &str) -> bool {
        self.entries
            .contains_key(&(source.to_owned(), target.to_owned()))
    }
}

// Public constructors

/// Precomputes effective deletion costs.
///
/// Runs multi-source Dijkstra on the **reversed** substitution graph, seeded
/// with `del(x)` per node:
///
/// ```text
/// eff_del(s) = min_x { shortest_sub_path(s -> x) + del(x) }
/// ```
pub fn compute_effective_deletion_costs(
    del_map: &CostMap<SingleTokenKey>,
    sub_map: &CostMap<SubstitutionKey>,
) -> EffectiveSingleTokenCosts {
    let default_cost = del_map.default_cost();

    let all_tokens = collect_tokens(del_map, sub_map);

    // Initial distance for every token = its direct deletion cost.
    let initial: HashMap<String, f64> = all_tokens
        .iter()
        .map(|t| (t.clone(), del_map.get_cost(t)))
        .collect();

    let rev_graph = build_reversed_sub_graph(sub_map);
    let (dist, prev) = dijkstra(&rev_graph, &initial);

    build_entries(
        &all_tokens,
        &dist,
        &initial,
        &prev,
        |token| del_map.has_key(token),
        |terminal| del_map.get_cost(terminal),
        default_cost,
        ChainDirection::Deletion,
    )
}

/// Precomputes effective insertion costs.
///
/// Runs multi-source Dijkstra on the **forward** substitution graph, seeded
/// with `ins(x)` per node:
///
/// ```text
/// eff_ins(t) = min_x { ins(x) + shortest_sub_path(x -> t) }
/// ```
pub fn compute_effective_insertion_costs(
    ins_map: &CostMap<SingleTokenKey>,
    sub_map: &CostMap<SubstitutionKey>,
) -> EffectiveSingleTokenCosts {
    let default_cost = ins_map.default_cost();

    let all_tokens = collect_tokens(ins_map, sub_map);

    let initial: HashMap<String, f64> = all_tokens
        .iter()
        .map(|t| (t.clone(), ins_map.get_cost(t)))
        .collect();

    let fwd_graph = build_forward_sub_graph(sub_map);
    let (dist, prev) = dijkstra(&fwd_graph, &initial);

    build_entries(
        &all_tokens,
        &dist,
        &initial,
        &prev,
        |token| ins_map.has_key(token),
        |initial_node| ins_map.get_cost(initial_node),
        default_cost,
        ChainDirection::Insertion,
    )
}

/// Precomputes effective substitution costs via all-pairs shortest paths.
///
/// For every source token in the substitution graph, runs Dijkstra on the
/// forward graph.  When a chain `sub(a->b) + sub(b->c)` is cheaper than the
/// default cost for `sub(a->c)`, the result is stored so the DP automatically
/// uses the cheaper path.
///
/// ```text
/// eff_sub(s, t) = min_path { sum of edge costs along s -> t }
/// ```
pub fn compute_effective_substitution_costs(
    sub_map: &CostMap<SubstitutionKey>,
) -> EffectiveSubstitutionCosts {
    let default_cost = sub_map.default_cost();

    let all_tokens: HashSet<String> = sub_map
        .costs
        .keys()
        .flat_map(|(s, t)| [s.clone(), t.clone()])
        .collect();

    let fwd_graph = build_forward_sub_graph(sub_map);
    let mut entries: HashMap<(String, String), (f64, EffectiveSubChain)> = HashMap::new();

    for source in &all_tokens {
        let initial: DistMap = [(source.clone(), 0.0)].into_iter().collect();
        let (dist, prev) = dijkstra(&fwd_graph, &initial);

        for target in &all_tokens {
            if target == source {
                continue;
            }
            let chain_cost = dist.get(target).copied().unwrap_or(f64::INFINITY);
            let direct_cost_opt = sub_map
                .costs
                .get(&(source.clone(), target.clone()))
                .copied();
            if let Some(entry) =
                classify_sub_pair(chain_cost, direct_cost_opt, default_cost, target, &prev)
            {
                entries.insert((source.clone(), target.clone()), entry);
            }
        }
    }

    let max_len = entries
        .keys()
        .flat_map(|(s, t)| [s.chars().count(), t.chars().count()])
        .max()
        .unwrap_or(1)
        .max(1);

    EffectiveSubstitutionCosts {
        entries,
        default_cost,
        max_token_length: max_len,
    }
}

// Dijkstra

/// Min-heap entry (BinaryHeap is a max-heap; reversed comparison gives min behaviour).
#[derive(Debug, Clone, PartialEq)]
struct HeapEntry {
    cost: f64,
    token: String,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.token.cmp(&other.token))
    }
}

/// Generic Dijkstra.
///
/// Returns `(dist, prev)` where `prev[v] = Some((u, edge_cost))` means node
/// `v` was relaxed via the edge `u -> v`.
fn dijkstra(graph: &HashMap<String, Vec<(String, f64)>>, initial: &DistMap) -> (DistMap, PrevMap) {
    let mut dist = initial.clone();
    let mut prev: PrevMap = initial.keys().map(|k| (k.clone(), None)).collect();

    let mut heap: BinaryHeap<HeapEntry> = dist
        .iter()
        .map(|(t, &c)| HeapEntry {
            cost: c,
            token: t.clone(),
        })
        .collect();

    while let Some(HeapEntry { cost, token }) = heap.pop() {
        if cost > dist[&token] {
            continue; // stale entry
        }
        for (nbr, edge_cost) in graph.get(&token).into_iter().flatten() {
            let new_cost = cost + edge_cost;
            let cur = dist.get(nbr).copied().unwrap_or(f64::INFINITY);
            if new_cost < cur {
                dist.insert(nbr.clone(), new_cost);
                prev.insert(nbr.clone(), Some((token.clone(), *edge_cost)));
                heap.push(HeapEntry {
                    cost: new_cost,
                    token: nbr.clone(),
                });
            }
        }
    }

    (dist, prev)
}

// Chain reconstruction & entry building

enum ChainDirection {
    Deletion,
    Insertion,
}

/// Builds the `EffectiveSingleTokenCosts` entries after Dijkstra.
///
/// A token is added iff it was explicitly configured in the raw map OR the
/// Dijkstra found a chain that is strictly cheaper than the direct cost.
/// This prevents tokens that only appear in the substitution graph (but whose
/// chain is NOT cheaper) from being registered as "explicitly deletable/
/// insertable", which would otherwise wrongly enable multi-char DP operations.
#[allow(clippy::too_many_arguments)]
fn build_entries(
    all_tokens: &HashSet<String>,
    dist: &DistMap,
    initial: &DistMap,
    prev: &PrevMap,
    in_raw_map: impl Fn(&str) -> bool,
    terminal_cost_fn: impl Fn(&str) -> f64,
    default_cost: f64,
    direction: ChainDirection,
) -> EffectiveSingleTokenCosts {
    let mut entries: HashMap<String, (f64, EffectiveOpChain)> = HashMap::new();

    for token in all_tokens {
        let final_dist = dist[token];
        let initial_cost = initial[token];
        let improved = final_dist < initial_cost;

        if !improved && !in_raw_map(token) {
            continue;
        }

        let chain = if prev[token].is_none() {
            EffectiveOpChain::Direct
        } else {
            match direction {
                ChainDirection::Deletion => {
                    let steps = reconstruct_deletion_steps(token, prev);
                    let terminal = steps.last().map(|(_, t, _)| t.as_str()).unwrap_or(token);
                    EffectiveOpChain::Via {
                        terminal_cost: terminal_cost_fn(terminal),
                        steps,
                    }
                }
                ChainDirection::Insertion => {
                    let (steps, initial_token) = reconstruct_insertion_steps(token, prev);
                    EffectiveOpChain::Via {
                        terminal_cost: terminal_cost_fn(&initial_token),
                        steps,
                    }
                }
            }
        };

        entries.insert(token.clone(), (final_dist, chain));
    }

    let max_len = entries
        .keys()
        .map(|k| k.chars().count())
        .max()
        .unwrap_or(1)
        .max(1);

    EffectiveSingleTokenCosts {
        entries,
        default_cost,
        max_token_length: max_len,
    }
}

/// Follows `prev` forward from `source` (deletion direction).
/// Returns steps `[(s, x1, c1), (x1, x2, c2), …]` in forward order.
fn reconstruct_deletion_steps(source: &str, prev: &PrevMap) -> Vec<(String, String, f64)> {
    let mut steps = Vec::new();
    let mut cur = source.to_string();
    while let Some((next, c)) = prev.get(&cur).and_then(|o| o.as_ref()) {
        steps.push((cur.clone(), next.clone(), *c));
        cur = next.clone();
    }
    steps
}

/// Follows `prev` backward from `target`, returns `(steps, seed)` where
/// `steps` are in forward order and `seed` is the path origin (the node
/// with `prev[seed] = None`).
///
/// Used for both insertion chains and substitution chains.
fn reconstruct_steps_to(target: &str, prev: &PrevMap) -> Vec<(String, String, f64)> {
    let mut steps_rev = Vec::new();
    let mut cur = target.to_string();
    while let Some((from, c)) = prev.get(&cur).and_then(|o| o.as_ref()) {
        steps_rev.push((from.clone(), cur.clone(), *c));
        cur = from.clone();
    }
    steps_rev.reverse();
    steps_rev
}

/// Wrapper that also returns the seed (initial) token — used for insertion chains.
fn reconstruct_insertion_steps(
    target: &str,
    prev: &PrevMap,
) -> (Vec<(String, String, f64)>, String) {
    let steps = reconstruct_steps_to(target, prev);
    let initial = steps
        .first()
        .map(|(f, _, _)| f.clone())
        .unwrap_or_else(|| target.to_string());
    (steps, initial)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn collect_tokens(
    single_map: &CostMap<SingleTokenKey>,
    sub_map: &CostMap<SubstitutionKey>,
) -> HashSet<String> {
    let mut tokens: HashSet<String> = single_map.costs.keys().cloned().collect();
    for (src, tgt) in sub_map.costs.keys() {
        tokens.insert(src.clone());
        tokens.insert(tgt.clone());
    }
    tokens
}

/// Builds the forward substitution graph: an edge `src -> tgt` has the sub cost.
fn build_forward_sub_graph(
    sub_map: &CostMap<SubstitutionKey>,
) -> HashMap<String, Vec<(String, f64)>> {
    let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for ((src, tgt), &c) in &sub_map.costs {
        graph.entry(src.clone()).or_default().push((tgt.clone(), c));
    }
    graph
}

/// Builds the reversed substitution graph: an edge `tgt -> src` has the sub cost.
fn build_reversed_sub_graph(
    sub_map: &CostMap<SubstitutionKey>,
) -> HashMap<String, Vec<(String, f64)>> {
    let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for ((src, tgt), &c) in &sub_map.costs {
        graph.entry(tgt.clone()).or_default().push((src.clone(), c));
    }
    graph
}

/// Decides how to record a substitution pair `(source, target)` given the
/// Dijkstra result for `source`.
///
/// Returns `None` if the pair should be omitted (not in the raw map and no
/// improvement over the default). Otherwise returns the effective cost and chain.
fn classify_sub_pair(
    chain_cost: f64,
    direct_cost_opt: Option<f64>,
    default_cost: f64,
    target: &str,
    prev: &PrevMap,
) -> Option<(f64, EffectiveSubChain)> {
    let in_raw_map = direct_cost_opt.is_some();
    if !in_raw_map && chain_cost >= default_cost {
        return None;
    }
    let direct = direct_cost_opt.unwrap_or(f64::INFINITY);
    if chain_cost < direct {
        let steps = reconstruct_steps_to(target, prev);
        Some((chain_cost, EffectiveSubChain::Via { steps }))
    } else {
        Some((direct, EffectiveSubChain::Direct))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SingleTokenCostMap, SubstitutionCostMap};

    fn make_del_map(entries: &[(&str, f64)]) -> CostMap<SingleTokenKey> {
        let costs: SingleTokenCostMap = entries.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        CostMap::<SingleTokenKey>::new(costs, 1.0)
    }

    fn make_ins_map(entries: &[(&str, f64)]) -> CostMap<SingleTokenKey> {
        make_del_map(entries)
    }

    fn make_sub_map(entries: &[((&str, &str), f64)]) -> CostMap<SubstitutionKey> {
        let costs: SubstitutionCostMap = entries
            .iter()
            .map(|((a, b), v)| ((a.to_string(), b.to_string()), *v))
            .collect();
        CostMap::<SubstitutionKey>::new(costs, 1.0, false)
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < f64::EPSILON * 100.0
    }

    // ── Deletion ──────────────────────────────────────────────────────────────

    #[test]
    fn direct_deletion_without_substitution_map() {
        let eff =
            compute_effective_deletion_costs(&make_del_map(&[("a", 0.3)]), &make_sub_map(&[]));
        assert!(approx_eq(eff.get_cost("a"), 0.3));
        assert_eq!(eff.get_chain("a"), EffectiveOpChain::Direct);
        assert!(approx_eq(eff.get_cost("z"), 1.0));
        assert_eq!(eff.get_chain("z"), EffectiveOpChain::Direct);
    }

    #[test]
    fn one_hop_deletion_chain() {
        // Issue #12: sub("6"->"G", 0.5) + del("G", 0.01) = 0.51 < direct del("6", 1.0)
        let eff = compute_effective_deletion_costs(
            &make_del_map(&[("G", 0.01)]),
            &make_sub_map(&[(("6", "G"), 0.5)]),
        );

        // "6": chain is strictly cheaper — must be present with Via chain
        assert!(approx_eq(eff.get_cost("6"), 0.51));
        assert_eq!(
            eff.get_chain("6"),
            EffectiveOpChain::Via {
                steps: vec![("6".to_string(), "G".to_string(), 0.5)],
                terminal_cost: 0.01,
            }
        );
        assert!(eff.has_key("6"));

        // "G": explicit in del_map, direct — must not be wrapped in a chain
        assert!(approx_eq(eff.get_cost("G"), 0.01));
        assert_eq!(eff.get_chain("G"), EffectiveOpChain::Direct);
        assert!(eff.has_key("G"));

        // unknown token falls back to default cost, Direct chain, not registered
        assert!(approx_eq(eff.get_cost("X"), 1.0));
        assert_eq!(eff.get_chain("X"), EffectiveOpChain::Direct);
        assert!(!eff.has_key("X"));

        // all tokens are single-character
        assert_eq!(eff.max_token_length, 1);
    }

    #[test]
    fn three_hop_deletion_chain() {
        // A->B (0.3), B->C (0.2), del(C)=0.01 -> chain(A) = 0.51 < 1.0
        let eff = compute_effective_deletion_costs(
            &make_del_map(&[("C", 0.01)]),
            &make_sub_map(&[(("A", "B"), 0.3), (("B", "C"), 0.2)]),
        );
        assert!(approx_eq(eff.get_cost("A"), 0.51));
        assert_eq!(
            eff.get_chain("A"),
            EffectiveOpChain::Via {
                steps: vec![
                    ("A".to_string(), "B".to_string(), 0.3),
                    ("B".to_string(), "C".to_string(), 0.2),
                ],
                terminal_cost: 0.01,
            }
        );
        // B also improves: 0.2 + 0.01 = 0.21 < 1.0
        assert!(approx_eq(eff.get_cost("B"), 0.21));
    }

    #[test]
    fn direct_deletion_preserved_when_chain_is_more_expensive() {
        // sub("a"->"b") = 0.5, del("b") = 0.8 -> chain = 1.3 > del("a") = 0.2
        let eff = compute_effective_deletion_costs(
            &make_del_map(&[("a", 0.2), ("b", 0.8)]),
            &make_sub_map(&[(("a", "b"), 0.5)]),
        );
        assert!(approx_eq(eff.get_cost("a"), 0.2));
        assert_eq!(eff.get_chain("a"), EffectiveOpChain::Direct);
    }

    #[test]
    fn multiple_substitution_targets_best_chain_wins() {
        // sub("X"->"A") = 0.4, del("A") = 0.3 -> 0.7
        // sub("X"->"B") = 0.1, del("B") = 0.5 -> 0.6 ← winner
        let eff = compute_effective_deletion_costs(
            &make_del_map(&[("A", 0.3), ("B", 0.5)]),
            &make_sub_map(&[(("X", "A"), 0.4), (("X", "B"), 0.1)]),
        );
        assert!(approx_eq(eff.get_cost("X"), 0.6));
        assert_eq!(
            eff.get_chain("X"),
            EffectiveOpChain::Via {
                steps: vec![("X".to_string(), "B".to_string(), 0.1)],
                terminal_cost: 0.5,
            }
        );
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    #[test]
    fn one_hop_insertion_chain() {
        // ins("x") = 0.1, sub("x"->"y") = 0.2 -> chain ins("y") = 0.3 < 1.0
        let eff = compute_effective_insertion_costs(
            &make_ins_map(&[("x", 0.1)]),
            &make_sub_map(&[(("x", "y"), 0.2)]),
        );
        assert!(approx_eq(eff.get_cost("y"), 0.3));
        assert_eq!(
            eff.get_chain("y"),
            EffectiveOpChain::Via {
                steps: vec![("x".to_string(), "y".to_string(), 0.2)],
                terminal_cost: 0.1,
            }
        );
    }

    #[test]
    fn three_hop_insertion_chain() {
        // ins(A)=0.05, sub(A->B)=0.3, sub(B->C)=0.2 -> chain(C)=0.55 < 1.0
        let eff = compute_effective_insertion_costs(
            &make_ins_map(&[("A", 0.05)]),
            &make_sub_map(&[(("A", "B"), 0.3), (("B", "C"), 0.2)]),
        );
        assert!(approx_eq(eff.get_cost("C"), 0.55));
        assert_eq!(
            eff.get_chain("C"),
            EffectiveOpChain::Via {
                steps: vec![
                    ("A".to_string(), "B".to_string(), 0.3),
                    ("B".to_string(), "C".to_string(), 0.2),
                ],
                terminal_cost: 0.05,
            }
        );
    }

    #[test]
    fn direct_insertion_preserved_when_chain_is_more_expensive() {
        // ins("y") = 0.1, ins("x") = 0.9, sub("x"->"y") = 0.5 -> chain = 1.4 > 0.1
        let eff = compute_effective_insertion_costs(
            &make_ins_map(&[("y", 0.1), ("x", 0.9)]),
            &make_sub_map(&[(("x", "y"), 0.5)]),
        );
        assert!(approx_eq(eff.get_cost("y"), 0.1));
        assert_eq!(eff.get_chain("y"), EffectiveOpChain::Direct);
    }

    // ── has_key semantics ─────────────────────────────────────────────────────

    #[test]
    fn has_key_only_when_explicit_or_chain_improves() {
        // sub("b"->"c", 0.2) + del("c", 1.0) = 1.2 > default 1.0 -> "b" NOT added
        let eff = compute_effective_deletion_costs(
            &make_del_map(&[("a", 0.5)]),
            &make_sub_map(&[(("b", "c"), 0.2)]),
        );
        assert!(eff.has_key("a"));
        assert!(!eff.has_key("b")); // chain not cheaper
        assert!(!eff.has_key("z"));

        // sub("6"->"G", 0.5) + del("G", 0.01) = 0.51 < 1.0 -> "6" IS added
        let eff2 = compute_effective_deletion_costs(
            &make_del_map(&[("G", 0.01)]),
            &make_sub_map(&[(("6", "G"), 0.5)]),
        );
        assert!(eff2.has_key("6"));
    }

    #[test]
    fn max_token_length_reflects_longest_key() {
        let eff =
            compute_effective_deletion_costs(&make_del_map(&[("ab", 0.5)]), &make_sub_map(&[]));
        assert_eq!(eff.max_token_length, 2);
    }

    // ── Substitution ──────────────────────────────────────────────────────────

    #[test]
    fn direct_substitution() {
        let eff = compute_effective_substitution_costs(&make_sub_map(&[(("a", "b"), 0.3)]));
        assert!(approx_eq(eff.get_cost("a", "b"), 0.3));
        assert_eq!(eff.get_chain("a", "b"), EffectiveSubChain::Direct);
        assert!(eff.has_key("a", "b"));
        // Unknown pair falls back to default.
        assert!(approx_eq(eff.get_cost("a", "c"), 1.0));
        assert!(!eff.has_key("a", "c"));
    }

    #[test]
    fn two_hop_substitution_chain() {
        // sub(a->b)=0.1 + sub(b->c)=0.1 -> eff_sub(a->c)=0.2 < default 1.0
        let eff = compute_effective_substitution_costs(&make_sub_map(&[
            (("a", "b"), 0.1),
            (("b", "c"), 0.1),
        ]));
        assert!(approx_eq(eff.get_cost("a", "c"), 0.2));
        assert_eq!(
            eff.get_chain("a", "c"),
            EffectiveSubChain::Via {
                steps: vec![
                    ("a".to_string(), "b".to_string(), 0.1),
                    ("b".to_string(), "c".to_string(), 0.1),
                ],
            }
        );
        assert!(eff.has_key("a", "c"));
        // Direct pair is still Direct.
        assert_eq!(eff.get_chain("a", "b"), EffectiveSubChain::Direct);
        assert!(approx_eq(eff.get_cost("a", "b"), 0.1));
    }

    #[test]
    fn direct_substitution_beats_chain() {
        // sub(a->b)=0.1 (direct), sub(a->c)=0.3, sub(c->b)=0.1
        // chain(a->b) via c = 0.4 > direct 0.1 -> Direct wins
        let eff = compute_effective_substitution_costs(&make_sub_map(&[
            (("a", "b"), 0.1),
            (("a", "c"), 0.3),
            (("c", "b"), 0.1),
        ]));
        assert!(approx_eq(eff.get_cost("a", "b"), 0.1));
        assert_eq!(eff.get_chain("a", "b"), EffectiveSubChain::Direct);
    }

    #[test]
    fn chain_substitution_beats_direct() {
        // sub(a->b)=0.5 (direct), sub(a->c)=0.1, sub(c->b)=0.1
        // chain(a->b) via c = 0.2 < direct 0.5 -> Via wins
        let eff = compute_effective_substitution_costs(&make_sub_map(&[
            (("a", "b"), 0.5),
            (("a", "c"), 0.1),
            (("c", "b"), 0.1),
        ]));
        assert!(approx_eq(eff.get_cost("a", "b"), 0.2));
        assert_eq!(
            eff.get_chain("a", "b"),
            EffectiveSubChain::Via {
                steps: vec![
                    ("a".to_string(), "c".to_string(), 0.1),
                    ("c".to_string(), "b".to_string(), 0.1),
                ],
            }
        );
    }

    #[test]
    fn chain_not_added_when_no_improvement_over_default() {
        // sub(a->b)=0.6, sub(b->c)=0.6 -> chain(a->c)=1.2 >= default 1.0 -> NOT added
        let eff = compute_effective_substitution_costs(&make_sub_map(&[
            (("a", "b"), 0.6),
            (("b", "c"), 0.6),
        ]));
        assert!(!eff.has_key("a", "c"));
        assert!(approx_eq(eff.get_cost("a", "c"), 1.0)); // default
    }
}
