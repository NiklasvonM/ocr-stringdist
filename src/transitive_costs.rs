//! Effective edit costs from a unified token graph.
//!
//! [`compute_effective_costs`] seeds one graph from the raw sub/ins/del maps,
//! closes it with Floyd-Warshall, and projects distances back: `dist[a][b]` for
//! substitutions, `dist[a][ε]` for deletions, `dist[ε][b]` for insertions.

use crate::cost_map::CostMap;
use crate::explanation::EditOperation;
use crate::types::{SingleTokenKey, SubstitutionKey};
use crate::weighted_levenshtein::explain_custom_levenshtein_precomputed;
use std::collections::{HashMap, HashSet};

const NO_NEXT_NODE: u32 = u32::MAX;

// Configured tokens up to this length are expanded into all of their substrings
// so closure can route through intermediate strings (e.g. `A -> AA -> AAA`).
// Capped because Floyd-Warshall is O(N³) over the resulting node set.
const MAX_SUBTOKEN_EXPANSION_CHARS: usize = 16;

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

    /// A cheaper path exists through mixed edit operations.
    EditPath { operations: Vec<EditOperation> },
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
        self.get_explicit_cost(token).unwrap_or(self.default_cost)
    }

    /// Cost of an explicit entry, or `None` if `token` is not in the map.
    #[inline]
    pub fn get_explicit_cost(&self, token: &str) -> Option<f64> {
        self.entries.get(token).map(|(c, _)| *c)
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

    /// A cheaper path was found through mixed edit operations.
    ///
    /// This is needed when an effective substitution path contains insertions
    /// or deletions between graph nodes, for example `A -> AAA -> B`.
    EditPath { operations: Vec<EditOperation> },
}

/// Precomputed effective substitution costs (all-pairs shortest paths).
///
/// Replaces the raw [`CostMap<SubstitutionKey>`] inside the DP so the algorithm
/// automatically uses the globally cheapest substitution path.
#[derive(Debug)]
pub struct EffectiveSubstitutionCosts {
    /// Entries are indexed by source token, then by target token.
    entries: HashMap<String, HashMap<String, (f64, EffectiveSubChain)>>,
    default_cost: f64,
    pub max_token_length: usize,
}

impl EffectiveSubstitutionCosts {
    #[inline]
    pub fn get_cost(&self, source: &str, target: &str) -> f64 {
        self.get_explicit_cost(source, target)
            .unwrap_or(self.default_cost)
    }

    /// Cost of an explicit entry, or `None` if `(source, target)` is not in the map.
    #[inline]
    pub fn get_explicit_cost(&self, source: &str, target: &str) -> Option<f64> {
        self.entries
            .get(source)
            .and_then(|targets| targets.get(target))
            .map(|(c, _)| *c)
    }

    #[inline]
    pub fn get_chain(&self, source: &str, target: &str) -> EffectiveSubChain {
        self.entries
            .get(source)
            .and_then(|targets| targets.get(target))
            .map(|(_, ch)| ch.clone())
            .unwrap_or(EffectiveSubChain::Direct)
    }

    #[inline]
    pub fn has_key(&self, source: &str, target: &str) -> bool {
        self.entries
            .get(source)
            .is_some_and(|targets| targets.contains_key(target))
    }
}

// Unified token graph solver

/// Interned identifier for a token graph node.
///
/// A node represents either a configured/derived token or the distinguished
/// epsilon node (`""`). Keeping this as a newtype instead of a bare `usize`
/// makes graph indexing sites explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct NodeId(u32);

impl NodeId {
    fn new(index: usize) -> Self {
        assert!(index < NO_NEXT_NODE as usize, "too many token graph nodes");
        Self(index as u32)
    }

    #[inline]
    fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug)]
struct EdgeResolution {
    operations: Vec<EditOperation>,
    substitution_steps: Vec<(String, String, f64)>,
    all_edges_are_raw_substitutions: bool,
}

/// Dense square matrix stored in row-major order.
///
/// Floyd-Warshall touches the matrix in tight nested loops; a flat vector avoids
/// the pointer chasing and per-row allocations of `Vec<Vec<T>>`.
#[derive(Debug)]
struct Matrix<T> {
    width: usize,
    cells: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    fn filled(width: usize, value: T) -> Self {
        Self {
            width,
            cells: vec![value; width * width],
        }
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

/// Bundle of effective sub/ins/del costs.
///
/// Used both as direct wrappers around the raw cost maps (input to closure) and
/// as the closed-graph projection returned by [`compute_effective_costs`].
#[derive(Debug)]
pub struct EffectiveCosts {
    pub sub: EffectiveSubstitutionCosts,
    pub ins: EffectiveSingleTokenCosts,
    pub del: EffectiveSingleTokenCosts,
}

impl EffectiveCosts {
    /// Direct wrappers around the raw cost maps, no closure applied.
    fn raw(
        sub_map: &CostMap<SubstitutionKey>,
        ins_map: &CostMap<SingleTokenKey>,
        del_map: &CostMap<SingleTokenKey>,
    ) -> Self {
        Self {
            sub: raw_effective_substitution_costs(sub_map),
            ins: raw_effective_single_token_costs(ins_map),
            del: raw_effective_single_token_costs(del_map),
        }
    }
}

fn raw_effective_substitution_costs(
    sub_map: &CostMap<SubstitutionKey>,
) -> EffectiveSubstitutionCosts {
    let mut entries: HashMap<String, HashMap<String, (f64, EffectiveSubChain)>> = HashMap::new();
    for ((source, target), &cost) in &sub_map.costs {
        entries
            .entry(source.clone())
            .or_default()
            .insert(target.clone(), (cost, EffectiveSubChain::Direct));
    }

    EffectiveSubstitutionCosts {
        max_token_length: sub_map
            .costs
            .keys()
            .flat_map(|(source, target)| [source.chars().count(), target.chars().count()])
            .max()
            .unwrap_or(0)
            .max(1),
        entries,
        default_cost: sub_map.default_cost(),
    }
}

fn raw_effective_single_token_costs(map: &CostMap<SingleTokenKey>) -> EffectiveSingleTokenCosts {
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
            .unwrap_or(0)
            .max(1),
        entries,
        default_cost: map.default_cost(),
    }
}

/// Adds or improves one directed graph edge and records its first hop.
fn set_seed_edge(
    dist: &mut Matrix<f64>,
    next: &mut Matrix<u32>,
    source: NodeId,
    target: NodeId,
    cost: f64,
) {
    if cost < *dist.get(source, target) {
        dist.set(source, target, cost);
        next.set(source, target, target.raw());
    }
}

/// Adds token-to-token edges that can be made by one explicit insertion/deletion.
///
/// This captures paths like `A -> AA` or `AB -> A` without running full
/// Levenshtein DP for every token pair during graph construction.
fn seed_embedded_single_token_edges(
    tokens: &[String],
    token_to_id: &HashMap<String, NodeId>,
    base: &EffectiveCosts,
    dist: &mut Matrix<f64>,
    next: &mut Matrix<u32>,
) {
    let insertions: Vec<(&str, f64)> = base
        .ins
        .entries
        .iter()
        .map(|(token, (cost, _))| (token.as_str(), *cost))
        .collect();
    let deletions: Vec<(&str, f64)> = base
        .del
        .entries
        .iter()
        .map(|(token, (cost, _))| (token.as_str(), *cost))
        .collect();

    for source in tokens {
        let source_id = token_to_id[source.as_str()];
        for (inserted, cost) in &insertions {
            for target in insert_token_variants(source, inserted) {
                if let Some(&target_id) = token_to_id.get(target.as_str()) {
                    set_seed_edge(dist, next, source_id, target_id, *cost);
                }
            }
        }

        for (deleted, cost) in &deletions {
            for target in delete_token_variants(source, deleted) {
                if let Some(&target_id) = token_to_id.get(target.as_str()) {
                    set_seed_edge(dist, next, source_id, target_id, *cost);
                }
            }
        }
    }
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
fn delete_token_variants(source: &str, deleted: &str) -> Vec<String> {
    source
        .match_indices(deleted)
        .map(|(idx, _)| {
            let end = idx + deleted.len();
            let mut target = String::with_capacity(source.len() - deleted.len());
            target.push_str(&source[..idx]);
            target.push_str(&source[end..]);
            target
        })
        .collect()
}

/// Unified state-transition solver for effective edit costs.
///
/// The graph has one node per relevant token plus a distinguished epsilon node.
/// Its dense adjacency matrix is initialized with direct weighted token-to-token
/// distances under the base effective costs, then closed with Floyd-Warshall.
struct CostSolver {
    base: EffectiveCosts,
    token_to_id: HashMap<String, NodeId>,
    tokens: Vec<String>,
    epsilon_id: NodeId,
    dist: Matrix<f64>,
    next: Matrix<u32>,
}

impl CostSolver {
    fn new(
        sub_map: &CostMap<SubstitutionKey>,
        ins_map: &CostMap<SingleTokenKey>,
        del_map: &CostMap<SingleTokenKey>,
    ) -> Self {
        let base = EffectiveCosts::raw(sub_map, ins_map, del_map);

        let tokens = collect_solver_tokens(sub_map, ins_map, del_map);
        let token_to_id: HashMap<String, NodeId> = tokens
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), NodeId::new(i)))
            .collect();
        let epsilon_id = token_to_id[""];
        let (dist, next) = Self::seed_distances(&tokens, &token_to_id, &base);

        Self {
            base,
            token_to_id,
            tokens,
            epsilon_id,
            dist,
            next,
        }
    }

    fn compute_effective_costs(mut self) -> EffectiveCosts {
        self.close_all_pairs();
        EffectiveCosts {
            sub: self.effective_substitutions(),
            ins: self.effective_insertions(),
            del: self.effective_deletions(),
        }
    }

    /// Initializes direct graph edges from raw edit operations.
    ///
    /// This deliberately avoids all-pairs weighted DP. The closure pass can
    /// discover multi-step paths from raw substitutions, epsilon insertions/
    /// deletions, and the targeted embedded single-token edges.
    fn seed_distances(
        tokens: &[String],
        token_to_id: &HashMap<String, NodeId>,
        base: &EffectiveCosts,
    ) -> (Matrix<f64>, Matrix<u32>) {
        let n = tokens.len();
        let mut dist = Matrix::filled(n, f64::INFINITY);
        let mut next = Matrix::filled(n, NO_NEXT_NODE);
        let epsilon = token_to_id[""];

        for node in 0..n {
            let node = NodeId::new(node);
            set_seed_edge(&mut dist, &mut next, node, node, 0.0);
        }

        for (source, targets) in &base.sub.entries {
            let source_id = token_to_id[source.as_str()];
            for (target, (cost, _)) in targets {
                let target_id = token_to_id[target.as_str()];
                set_seed_edge(&mut dist, &mut next, source_id, target_id, *cost);
            }
        }

        for (token, (cost, _)) in &base.ins.entries {
            set_seed_edge(
                &mut dist,
                &mut next,
                epsilon,
                token_to_id[token.as_str()],
                *cost,
            );
        }

        for (token, (cost, _)) in &base.del.entries {
            set_seed_edge(
                &mut dist,
                &mut next,
                token_to_id[token.as_str()],
                epsilon,
                *cost,
            );
        }

        seed_embedded_single_token_edges(tokens, token_to_id, base, &mut dist, &mut next);
        (dist, next)
    }

    /// Floyd-Warshall all-pairs shortest paths over the unified graph.
    fn close_all_pairs(&mut self) {
        let n = self.dist.width;
        for via in 0..n {
            let via = NodeId::new(via);
            for source in 0..n {
                let source = NodeId::new(source);
                let source_to_via = *self.dist.get(source, via);
                if !source_to_via.is_finite() {
                    continue;
                }
                for target in 0..n {
                    let target = NodeId::new(target);
                    let candidate = source_to_via + *self.dist.get(via, target);
                    if candidate < *self.dist.get(source, target) {
                        self.dist.set(source, target, candidate);
                        self.next.set(source, target, *self.next.get(source, via));
                    }
                }
            }
        }
    }

    fn id(&self, token: &str) -> NodeId {
        self.token_to_id[token]
    }

    fn cost(&self, source: NodeId, target: NodeId) -> f64 {
        *self.dist.get(source, target)
    }

    fn path(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        if *self.next.get(source, target) == NO_NEXT_NODE {
            return None;
        }
        let mut path = vec![source];
        let mut current = source;
        while current != target {
            let next = *self.next.get(current, target);
            if next == NO_NEXT_NODE {
                return None;
            }
            current = NodeId(next);
            path.push(current);
        }
        Some(path)
    }

    fn token(&self, node: NodeId) -> &str {
        &self.tokens[node.index()]
    }

    /// Projects closed token-to-token distances into effective substitutions.
    fn effective_substitutions(&self) -> EffectiveSubstitutionCosts {
        let mut entries: HashMap<String, HashMap<String, (f64, EffectiveSubChain)>> =
            HashMap::new();
        for source in self.non_epsilon_tokens() {
            for target in self.non_epsilon_tokens() {
                if source == target {
                    continue;
                }
                let best = self.cost(self.id(source), self.id(target));
                let raw_direct = self.raw_substitution_cost(source, target);
                let in_raw_map = raw_direct.is_some();
                let direct_cost = raw_direct.unwrap_or(self.base.sub.default_cost);

                if !in_raw_map && best >= self.base.sub.default_cost {
                    continue;
                }

                let entry = if best < direct_cost {
                    let chain = self
                        .substitution_chain(self.id(source), self.id(target))
                        .unwrap_or(EffectiveSubChain::Direct);
                    (best, chain)
                } else {
                    (direct_cost, EffectiveSubChain::Direct)
                };

                entries
                    .entry(source.to_owned())
                    .or_default()
                    .insert(target.to_owned(), entry);
            }
        }

        EffectiveSubstitutionCosts {
            max_token_length: max_pair_token_len(&entries),
            entries,
            default_cost: self.base.sub.default_cost,
        }
    }

    /// Projects token-to-epsilon distances into effective deletions.
    fn effective_deletions(&self) -> EffectiveSingleTokenCosts {
        let mut entries: HashMap<String, (f64, EffectiveOpChain)> = HashMap::new();
        for token in self.non_epsilon_tokens() {
            let node = self.id(token);
            let best = self.cost(node, self.epsilon_id);
            let direct = self.base.del.get_cost(token);
            if self.base.del.has_key(token) || best < direct {
                let chain = if best < direct {
                    self.deletion_chain(node)
                        .unwrap_or(EffectiveOpChain::Direct)
                } else {
                    self.base.del.get_chain(token)
                };
                entries.insert(token.to_owned(), (best, chain));
            }
        }

        EffectiveSingleTokenCosts {
            max_token_length: max_single_token_len(&entries),
            entries,
            default_cost: self.base.del.default_cost,
        }
    }

    /// Builds the explanation chain for an effective substitution.
    fn substitution_chain(&self, source: NodeId, target: NodeId) -> Option<EffectiveSubChain> {
        let path = self.path(source, target)?;
        if path.len() <= 2 {
            return Some(EffectiveSubChain::Direct);
        }

        let resolution = self.resolve_path_edges(&path);

        if resolution.all_edges_are_raw_substitutions {
            Some(EffectiveSubChain::Via {
                steps: resolution.substitution_steps,
            })
        } else {
            Some(EffectiveSubChain::EditPath {
                operations: resolution.operations,
            })
        }
    }

    /// Builds the explanation chain for an effective deletion.
    fn deletion_chain(&self, source: NodeId) -> Option<EffectiveOpChain> {
        let path = self.path(source, self.epsilon_id)?;
        if path.len() <= 2 {
            return Some(EffectiveOpChain::Direct);
        }

        let mut resolution = self.resolve_path_edges(&path[..path.len() - 1]);

        let terminal = *path.get(path.len() - 2)?;
        let terminal_token = self.token(terminal);
        let terminal_cost = self.base.del.get_explicit_cost(terminal_token)?;

        if resolution.all_edges_are_raw_substitutions {
            Some(EffectiveOpChain::Via {
                steps: resolution.substitution_steps,
                terminal_cost,
            })
        } else {
            resolution.operations.push(EditOperation::Delete {
                source: terminal_token.to_string(),
                cost: terminal_cost,
            });
            Some(EffectiveOpChain::EditPath {
                operations: resolution.operations,
            })
        }
    }

    /// Builds the explanation chain for an effective insertion.
    fn insertion_chain(&self, target: NodeId) -> Option<EffectiveOpChain> {
        let path = self.path(self.epsilon_id, target)?;
        if path.len() <= 2 {
            return Some(EffectiveOpChain::Direct);
        }

        let initial = *path.get(1)?;
        let initial_token = self.token(initial);
        let terminal_cost = self.base.ins.get_explicit_cost(initial_token)?;

        let mut operations = vec![EditOperation::Insert {
            target: initial_token.to_string(),
            cost: terminal_cost,
        }];
        let mut resolution = self.resolve_path_edges(&path[1..]);

        if resolution.all_edges_are_raw_substitutions {
            Some(EffectiveOpChain::Via {
                steps: resolution.substitution_steps,
                terminal_cost,
            })
        } else {
            operations.append(&mut resolution.operations);
            Some(EffectiveOpChain::EditPath { operations })
        }
    }

    /// Converts graph edges back into user-visible edit operations.
    fn resolve_path_edges(&self, path: &[NodeId]) -> EdgeResolution {
        let mut operations = Vec::new();
        let mut substitution_steps = Vec::new();
        let mut all_edges_are_raw_substitutions = true;

        for edge in path.windows(2) {
            let from = self.token(edge[0]);
            let to = self.token(edge[1]);
            if let Some(cost) = self.raw_substitution_cost(from, to) {
                substitution_steps.push((from.to_owned(), to.to_owned(), cost));
                operations.push(EditOperation::Substitute {
                    source: from.to_owned(),
                    target: to.to_owned(),
                    cost,
                });
            } else {
                all_edges_are_raw_substitutions = false;
                operations.extend(
                    explain_custom_levenshtein_precomputed(from, to, &self.base)
                        .into_iter()
                        .filter(|op| !matches!(op, EditOperation::Match { .. })),
                );
            }
        }

        EdgeResolution {
            operations,
            substitution_steps,
            all_edges_are_raw_substitutions,
        }
    }

    /// Raw substitution edge cost (pre-closure), or `None` if not configured.
    fn raw_substitution_cost(&self, source: &str, target: &str) -> Option<f64> {
        self.base.sub.get_explicit_cost(source, target)
    }

    /// Projects epsilon-to-token distances into effective insertions.
    fn effective_insertions(&self) -> EffectiveSingleTokenCosts {
        let mut entries: HashMap<String, (f64, EffectiveOpChain)> = HashMap::new();
        for token in self.non_epsilon_tokens() {
            let node = self.id(token);
            let best = self.cost(self.epsilon_id, node);
            let direct = self.base.ins.get_cost(token);
            if self.base.ins.has_key(token) || best < direct {
                let chain = if best < direct {
                    self.insertion_chain(node)
                        .unwrap_or(EffectiveOpChain::Direct)
                } else {
                    self.base.ins.get_chain(token)
                };
                entries.insert(token.to_owned(), (best, chain));
            }
        }

        EffectiveSingleTokenCosts {
            max_token_length: max_single_token_len(&entries),
            entries,
            default_cost: self.base.ins.default_cost,
        }
    }

    fn non_epsilon_tokens(&self) -> impl Iterator<Item = &str> {
        self.tokens
            .iter()
            .filter(|token| !token.is_empty())
            .map(String::as_str)
    }
}

/// Collects graph nodes and bounded substrings needed for transitive closure.
fn collect_solver_tokens(
    sub_map: &CostMap<SubstitutionKey>,
    ins_map: &CostMap<SingleTokenKey>,
    del_map: &CostMap<SingleTokenKey>,
) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
    tokens.insert(String::new()); // epsilon
    tokens.extend(ins_map.costs.keys().cloned());
    tokens.extend(del_map.costs.keys().cloned());
    tokens.extend(
        sub_map
            .costs
            .keys()
            .flat_map(|(source, target)| [source.clone(), target.clone()]),
    );

    let originals: Vec<String> = tokens.iter().cloned().collect();
    for token in originals {
        let char_count = token.chars().count();
        if char_count > MAX_SUBTOKEN_EXPANSION_CHARS {
            continue;
        }

        let mut boundaries: Vec<usize> = token.char_indices().map(|(idx, _)| idx).collect();
        boundaries.push(token.len());
        for start in 0..char_count {
            for end in (start + 1)..=char_count {
                tokens.insert(token[boundaries[start]..boundaries[end]].to_owned());
            }
        }
    }

    tokens.into_iter().collect()
}

/// Longest token length in effective substitution entries.
fn max_pair_token_len(
    entries: &HashMap<String, HashMap<String, (f64, EffectiveSubChain)>>,
) -> usize {
    entries
        .iter()
        .flat_map(|(source, targets)| {
            targets
                .keys()
                .flat_map(move |target| [source.chars().count(), target.chars().count()])
        })
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Longest token length in effective insertion/deletion entries.
fn max_single_token_len(entries: &HashMap<String, (f64, EffectiveOpChain)>) -> usize {
    entries
        .keys()
        .map(|token| token.chars().count())
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Computes all effective costs with one unified state-transition graph.
pub(crate) fn compute_effective_costs(
    sub_map: &CostMap<SubstitutionKey>,
    ins_map: &CostMap<SingleTokenKey>,
    del_map: &CostMap<SingleTokenKey>,
) -> EffectiveCosts {
    CostSolver::new(sub_map, ins_map, del_map).compute_effective_costs()
}
