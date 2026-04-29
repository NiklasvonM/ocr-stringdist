//! Effective edit costs under substitution-only transitivity and token-graph closure.
//!
//! The Python-facing calculator uses [`compute_effective_costs_unified`], which
//! seeds one token graph from raw insertion/deletion/substitution maps and then
//! runs a single Floyd-Warshall closure. The older Dijkstra-based constructors
//! remain as small building blocks for tests and for substitution-chain
//! provenance in Rust-only callers.
//!
//! Effective costs are stored in [`EffectiveSingleTokenCosts`] (del/ins) and
//! [`EffectiveSubstitutionCosts`] (sub), computed once at construction (for example when
//! building the Rust/Python calculator).

use crate::cost_map::CostMap;
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::custom_levenshtein_distance_precomputed;
use std::collections::{HashMap, HashSet};

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

// Unified token graph solver

/// Interned identifier for a token graph node.
///
/// A node represents either a configured/derived token or the distinguished
/// epsilon node (`""`). Keeping this as a newtype instead of a bare `usize`
/// makes graph indexing sites explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct NodeId(usize);

impl NodeId {
    #[inline]
    fn index(self) -> usize {
        self.0
    }
}

/// All base costs used to seed the unified token graph.
///
/// These are direct wrappers around the raw cost maps. They intentionally do
/// not apply transitive closure; [`CostSolver`] owns the single closure pass.
struct BaseEffectiveCosts {
    sub: EffectiveSubstitutionCosts,
    ins: EffectiveSingleTokenCosts,
    del: EffectiveSingleTokenCosts,
}

fn raw_effective_substitution_costs(sub_map: &CostMap<SubstitutionKey>) -> EffectiveSubstitutionCosts {
    let entries = sub_map
        .costs
        .iter()
        .map(|((source, target), &cost)| {
            (
                (source.clone(), target.clone()),
                (cost, EffectiveSubChain::Direct),
            )
        })
        .collect();

    EffectiveSubstitutionCosts {
        max_token_length: sub_map
            .costs
            .keys()
            .flat_map(|(source, target)| [source.chars().count(), target.chars().count()])
            .max()
            .unwrap_or(1)
            .max(1),
        entries,
        default_cost: sub_map.default_cost(),
    }
}

fn raw_effective_single_token_costs(
    map: &CostMap<SingleTokenKey>,
) -> EffectiveSingleTokenCosts {
    let entries = map
        .costs
        .iter()
        .map(|(token, &cost)| (token.clone(), (cost, EffectiveOpChain::Direct)))
        .collect();

    EffectiveSingleTokenCosts {
        max_token_length: map
            .costs
            .keys()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1)
            .max(1),
        entries,
        default_cost: map.default_cost(),
    }
}

/// Unified state-transition solver for effective edit costs.
///
/// The graph has one node per relevant token plus a distinguished epsilon node.
/// Its dense adjacency matrix is initialized with direct weighted token-to-token
/// distances under the base effective costs, then closed with Floyd-Warshall.
///
/// After closure, every operation is a projection from the same matrix:
/// - substitution `a -> b`: `dist[a][b]`
/// - deletion `a`: `dist[a][epsilon]`
/// - insertion `b`: `dist[epsilon][b]`
struct CostSolver<'a> {
    sub_map: &'a CostMap<SubstitutionKey>,
    ins_map: &'a CostMap<SingleTokenKey>,
    del_map: &'a CostMap<SingleTokenKey>,
    base: BaseEffectiveCosts,
    token_to_id: HashMap<String, NodeId>,
    tokens: Vec<String>,
    epsilon_id: NodeId,
    dist: Vec<Vec<f64>>,
    next: Vec<Vec<Option<NodeId>>>,
}

impl<'a> CostSolver<'a> {
    fn new(
        sub_map: &'a CostMap<SubstitutionKey>,
        ins_map: &'a CostMap<SingleTokenKey>,
        del_map: &'a CostMap<SingleTokenKey>,
    ) -> Self {
        let base = BaseEffectiveCosts {
            sub: raw_effective_substitution_costs(sub_map),
            ins: raw_effective_single_token_costs(ins_map),
            del: raw_effective_single_token_costs(del_map),
        };

        let tokens = collect_solver_tokens(sub_map, ins_map, del_map, &base);
        let token_to_id: HashMap<String, NodeId> = tokens
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), NodeId(i)))
            .collect();
        let epsilon_id = token_to_id[""];
        let (dist, next) = Self::seed_distances(&tokens, &base);

        Self {
            sub_map,
            ins_map,
            del_map,
            base,
            token_to_id,
            tokens,
            epsilon_id,
            dist,
            next,
        }
    }

    fn compute_effective_costs(mut self) -> (
        EffectiveSubstitutionCosts,
        EffectiveSingleTokenCosts,
        EffectiveSingleTokenCosts,
    ) {
        self.close_all_pairs();
        (
            self.effective_substitutions(),
            self.effective_deletions(),
            self.effective_insertions(),
        )
    }

    /// Initialize graph edges from direct weighted token-to-token distances.
    fn seed_distances(
        tokens: &[String],
        base: &BaseEffectiveCosts,
    ) -> (Vec<Vec<f64>>, Vec<Vec<Option<NodeId>>>) {
        let n = tokens.len();
        let mut dist = vec![vec![f64::INFINITY; n]; n];
        let mut next = vec![vec![None; n]; n];
        for source in 0..n {
            dist[source][source] = 0.0;
            next[source][source] = Some(NodeId(source));
            for target in 0..n {
                if source == target {
                    continue;
                }
                dist[source][target] = custom_levenshtein_distance_precomputed(
                    &tokens[source],
                    &tokens[target],
                    &base.sub,
                    &base.ins,
                    &base.del,
                );
                if dist[source][target].is_finite() {
                    next[source][target] = Some(NodeId(target));
                }
            }
        }
        (dist, next)
    }

    /// Floyd-Warshall all-pairs shortest paths over the unified graph.
    #[allow(clippy::needless_range_loop)] // Indexed `target` avoids simultaneous borrows of `dist`.
    fn close_all_pairs(&mut self) {
        let n = self.dist.len();
        for via in 0..n {
            for source in 0..n {
                let source_to_via = self.dist[source][via];
                if !source_to_via.is_finite() {
                    continue;
                }
                for target in 0..n {
                    let candidate = source_to_via + self.dist[via][target];
                    if candidate < self.dist[source][target] {
                        self.dist[source][target] = candidate;
                        self.next[source][target] = self.next[source][via];
                    }
                }
            }
        }
    }

    fn id(&self, token: &str) -> NodeId {
        self.token_to_id[token]
    }

    fn cost(&self, source: NodeId, target: NodeId) -> f64 {
        self.dist[source.index()][target.index()]
    }

    fn path(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        self.next[source.index()][target.index()]?;
        let mut path = vec![source];
        let mut current = source;
        while current != target {
            current = self.next[current.index()][target.index()]?;
            path.push(current);
        }
        Some(path)
    }

    fn token(&self, node: NodeId) -> &str {
        &self.tokens[node.index()]
    }

    fn effective_substitutions(&self) -> EffectiveSubstitutionCosts {
        let mut entries: HashMap<(String, String), (f64, EffectiveSubChain)> = HashMap::new();
        for source in self.non_epsilon_tokens() {
            for target in self.non_epsilon_tokens() {
                if source == target {
                    continue;
                }
                let best = self.cost(self.id(source), self.id(target));
                let raw_direct = self
                    .sub_map
                    .costs
                    .get(&(source.clone(), target.clone()))
                    .copied()
                    .unwrap_or(self.base.sub.default_cost);
                let in_raw_map = self
                    .sub_map
                    .costs
                    .contains_key(&(source.clone(), target.clone()));

                if !in_raw_map && best >= self.base.sub.default_cost {
                    continue;
                }

                if best < raw_direct {
                    let chain = self
                        .substitution_chain(self.id(source), self.id(target))
                        .unwrap_or(EffectiveSubChain::Direct);
                    entries.insert((source.clone(), target.clone()), (best, chain));
                } else {
                    entries.insert(
                        (source.clone(), target.clone()),
                        (raw_direct, EffectiveSubChain::Direct),
                    );
                }
            }
        }

        EffectiveSubstitutionCosts {
            max_token_length: max_pair_token_len(&entries),
            entries,
            default_cost: self.base.sub.default_cost,
        }
    }

    fn effective_deletions(&self) -> EffectiveSingleTokenCosts {
        let mut entries: HashMap<String, (f64, EffectiveOpChain)> = HashMap::new();
        for token in self.non_epsilon_tokens() {
            let node = self.id(token);
            let best = self.cost(node, self.epsilon_id);
            let direct = self.base.del.get_cost(token);
            if self.del_map.has_key(token) || self.base.del.has_key(token) || best < direct {
                let chain = if best < direct {
                    self.deletion_chain(node).unwrap_or(EffectiveOpChain::Direct)
                } else {
                    self.base.del.get_chain(token)
                };
                entries.insert(token.clone(), (best, chain));
            }
        }

        EffectiveSingleTokenCosts {
            max_token_length: max_single_token_len(&entries),
            entries,
            default_cost: self.base.del.default_cost,
        }
    }

    fn substitution_chain(&self, source: NodeId, target: NodeId) -> Option<EffectiveSubChain> {
        let path = self.path(source, target)?;
        if path.len() <= 2 {
            return Some(EffectiveSubChain::Direct);
        }

        let mut steps = Vec::new();
        for edge in path.windows(2) {
            let from = self.token(edge[0]);
            let to = self.token(edge[1]);
            let cost = self.raw_substitution_cost(from, to)?;
            steps.push((from.to_string(), to.to_string(), cost));
        }

        Some(EffectiveSubChain::Via { steps })
    }

    fn deletion_chain(&self, source: NodeId) -> Option<EffectiveOpChain> {
        let path = self.path(source, self.epsilon_id)?;
        if path.len() <= 2 {
            return Some(EffectiveOpChain::Direct);
        }

        let terminal = *path.get(path.len() - 2)?;
        let terminal_token = self.token(terminal);
        let terminal_cost = self.del_map.get_cost(terminal_token);
        if !self.del_map.has_key(terminal_token) {
            return None;
        }

        let mut steps = Vec::new();
        for edge in path[..path.len() - 1].windows(2) {
            let from = self.token(edge[0]);
            let to = self.token(edge[1]);
            let cost = self.raw_substitution_cost(from, to)?;
            steps.push((from.to_string(), to.to_string(), cost));
        }
        Some(EffectiveOpChain::Via {
            steps,
            terminal_cost,
        })
    }

    fn insertion_chain(&self, target: NodeId) -> Option<EffectiveOpChain> {
        let path = self.path(self.epsilon_id, target)?;
        if path.len() <= 2 {
            return Some(EffectiveOpChain::Direct);
        }

        let initial = *path.get(1)?;
        let initial_token = self.token(initial);
        let terminal_cost = self.ins_map.get_cost(initial_token);
        if !self.ins_map.has_key(initial_token) {
            return None;
        }

        let mut steps = Vec::new();
        for edge in path[1..].windows(2) {
            let from = self.token(edge[0]);
            let to = self.token(edge[1]);
            let cost = self.raw_substitution_cost(from, to)?;
            steps.push((from.to_string(), to.to_string(), cost));
        }
        Some(EffectiveOpChain::Via {
            steps,
            terminal_cost,
        })
    }

    fn raw_substitution_cost(&self, source: &str, target: &str) -> Option<f64> {
        self.sub_map
            .costs
            .get(&(source.to_owned(), target.to_owned()))
            .copied()
    }

    fn effective_insertions(&self) -> EffectiveSingleTokenCosts {
        let mut entries: HashMap<String, (f64, EffectiveOpChain)> = HashMap::new();
        for token in self.non_epsilon_tokens() {
            let node = self.id(token);
            let best = self.cost(self.epsilon_id, node);
            let direct = self.base.ins.get_cost(token);
            if self.ins_map.has_key(token) || self.base.ins.has_key(token) || best < direct {
                let chain = if best < direct {
                    self.insertion_chain(node).unwrap_or(EffectiveOpChain::Direct)
                } else {
                    self.base.ins.get_chain(token)
                };
                entries.insert(token.clone(), (best, chain));
            }
        }

        EffectiveSingleTokenCosts {
            max_token_length: max_single_token_len(&entries),
            entries,
            default_cost: self.base.ins.default_cost,
        }
    }

    fn non_epsilon_tokens(&self) -> impl Iterator<Item = &String> {
        self.tokens.iter().filter(|token| !token.is_empty())
    }
}

fn collect_solver_tokens(
    sub_map: &CostMap<SubstitutionKey>,
    ins_map: &CostMap<SingleTokenKey>,
    del_map: &CostMap<SingleTokenKey>,
    base: &BaseEffectiveCosts,
) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
    tokens.insert(String::new()); // epsilon
    tokens.extend(ins_map.costs.keys().cloned());
    tokens.extend(del_map.costs.keys().cloned());
    tokens.extend(base.ins.entries.keys().cloned());
    tokens.extend(base.del.entries.keys().cloned());
    tokens.extend(
        sub_map
            .costs
            .keys()
            .flat_map(|(source, target)| [source.clone(), target.clone()]),
    );
    tokens.extend(
        base.sub
            .entries
            .keys()
            .flat_map(|(source, target)| [source.clone(), target.clone()]),
    );

    let originals: Vec<String> = tokens.iter().cloned().collect();
    for token in originals {
        let chars: Vec<char> = token.chars().collect();
        for start in 0..chars.len() {
            for end in (start + 1)..=chars.len() {
                tokens.insert(chars[start..end].iter().collect());
            }
        }
    }

    tokens.into_iter().collect()
}

fn max_pair_token_len(entries: &HashMap<(String, String), (f64, EffectiveSubChain)>) -> usize {
    entries
        .keys()
        .flat_map(|(source, target)| [source.chars().count(), target.chars().count()])
        .max()
        .unwrap_or(1)
        .max(1)
}

fn max_single_token_len(entries: &HashMap<String, (f64, EffectiveOpChain)>) -> usize {
    entries
        .keys()
        .map(|token| token.chars().count())
        .max()
        .unwrap_or(1)
        .max(1)
}

/// Computes all effective costs with one unified state-transition graph.
pub(crate) fn compute_effective_costs_unified(
    sub_map: &CostMap<SubstitutionKey>,
    ins_map: &CostMap<SingleTokenKey>,
    del_map: &CostMap<SingleTokenKey>,
) -> (
    EffectiveSubstitutionCosts,
    EffectiveSingleTokenCosts,
    EffectiveSingleTokenCosts,
) {
    CostSolver::new(sub_map, ins_map, del_map).compute_effective_costs()
}
