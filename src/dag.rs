//! [Directed Acyclic
//! Graphs](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAGs)
//! represented as [Strictly Upper Triangular
//! matrices](https://mathworld.wolfram.com/StrictlyUpperTriangularMatrix.html).
//!
//! A create for working with DAGs where it is known upfront (i.e. statically)
//! that graphs are directed and there are no cycles.

//! There are several assumptions imposed on *your* code:
//!
//! 1. DAG vertices are integer numbers (`usize`) which is used to trivially
//!    test whether adding an edge would form a cycle: edges are only allowed to
//!    go "forward", i.e. from `u` to `v` iff `u < v`.  Otherwise we panic.
//! 1. Vertices numbering starts at 0.
//! 1. The number of vertices is determined at construction time and
//!    growing/shrinking generally requires a new graph to be constructed.
//!
//! In exchange for these assumptions you get these useful properties:
//! * **Correctness**: Cycles (an illegal state) are unrepresentable.
//! * **Compactness**: Edges are just bits in a bit set.  The implementation
//!   uses just `(|V|*|V|-|V|)/2` *bits* of memory + a constant.
//! * **CPU cache locality**: Edges are stored in a [row-major packed
//!   representation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
//!   so that iteration over the neighbours of a vertex is just an iteration
//!   over *consecutive* bits in a bit set.
//! * **Low cognitive overhead**: No need to deal with type-level shenenigans to
//!   get basic tasks done.
//! * **Asymptotic complexity reduction**: Generating a random DAG is a `O(|E|)`
//!   operation.  That was actually the original motivation for writing this
//!   crate.  It can be used with
//!   [quickcheck](https://crates.io/crates/quickcheck) efficiently.  In fact,
//!   [`DirectedAcyclicGraph`] implements [`quickcheck::Arbitrary`] (with
//!   meaningful shrinking).
//!
//! ## Anti-features
//!
//! * No support for storing anything in the vertices.
//! * No support for assigning weights to either edges or vertices.
//! * No support for enumerating *incoming* edges of a vertex, only *outgoing*
//!   ones.
//! * No serde impls.  Simply serialize/deserialize the list of edges with a
//!   library of your choosing.
//!
//! # Entry points
//!
//! See either [`DirectedAcyclicGraph::empty`],
//! [`DirectedAcyclicGraph::from_edges_iter`], or
//! [`DirectedAcyclicGraph::from_adjacency_matrix`] for the "entry point" to
//! this crate.

use std::collections::VecDeque;
use std::io::Write;

use fixedbitset::FixedBitSet;
use proptest::prelude::*;
use quickcheck::{Arbitrary, Gen};
use rand::{prelude::StdRng, Rng, SeedableRng};
use rand_distr::{Bernoulli, Distribution};

use crate::TraversableDirectedGraph;
use crate::strictly_upper_triangular_logical_matrix::{
    self, strictly_upper_triangular_matrix_capacity, StrictlyUpperTriangularLogicalMatrix,
};

/// A mutable, single-threaded directed acyclic graph.
#[derive(Clone)]
pub struct DirectedAcyclicGraph {
    adjacency_matrix: StrictlyUpperTriangularLogicalMatrix,
}

impl std::fmt::Debug for DirectedAcyclicGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones: Vec<(usize, usize)> = self.iter_edges().collect();
        write!(
            f,
            "DirectedAcyclicGraph::from_edges_iter({}, vec!{:?}.iter().cloned())",
            self.get_vertex_count(),
            ones
        )?;
        Ok(())
    }
}

impl TraversableDirectedGraph for DirectedAcyclicGraph {
    fn extend_with_children(&self, children: &mut Vec<usize>, u: usize) {
        self.extend_with_children(children, u)
    }

    fn extend_with_parents(&self, parents: &mut Vec<usize>, v: usize) {
        self.extend_with_parents(parents, v)
    }
}

impl DirectedAcyclicGraph {
    pub fn empty(vertex_count: usize) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularLogicalMatrix::zeroed(vertex_count),
        }
    }

    /// Constructs a DAG from an iterator of edges.
    ///
    /// Requires `u < vertex_count && v < vertex_count && u < v` for every edge
    /// `(u, v)` in `edges`.  Panics otherwise.
    pub fn from_edges_iter<I: Iterator<Item = (usize, usize)>>(
        vertex_count: usize,
        edges: I,
    ) -> Self {
        let adjacency_matrix = StrictlyUpperTriangularLogicalMatrix::from_iter(vertex_count, edges);
        Self { adjacency_matrix }
    }

    /// Generate a random DAG sampling edges from `edges_distribution`.
    ///
    /// For example, to generate a random DAG with a probability `1/2` of having
    /// an edge between any two of `321` vertices, do:
    ///
    /// ```
    /// use dograph::dag::DirectedAcyclicGraph;
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use rand::distributions::Bernoulli;
    ///
    /// let mut rng = StdRng::seed_from_u64(123);
    /// let dag = DirectedAcyclicGraph::random(321, &mut rng, Bernoulli::new(0.5).unwrap());
    /// ```
    pub fn random<R: Rng, D: Distribution<bool> + Copy>(
        vertex_count: usize,
        rng: &mut R,
        edges_distribution: D,
    ) -> Self {
        let mut dag = DirectedAcyclicGraph::empty(vertex_count);
        for u in 0..vertex_count {
            for v in (u + 1)..vertex_count {
                dag.set_edge(u, v, rng.sample(edges_distribution));
            }
        }
        dag
    }

    pub fn from_random_edges(vertex_count: usize, edges: &[bool]) -> Self {
        assert_eq!(
            edges.len(),
            strictly_upper_triangular_matrix_capacity(vertex_count)
        );
        let mut bitset = FixedBitSet::with_capacity(edges.len());
        for (index, value) in edges.iter().enumerate() {
            bitset.set(index, *value);
        }
        let matrix = StrictlyUpperTriangularLogicalMatrix::from_bitset(vertex_count, bitset);
        let dag = DirectedAcyclicGraph::from_adjacency_matrix(matrix);
        dag
    }

    /// Construct a DAG from an pre-computed adjacency matrix.
    pub fn from_adjacency_matrix(adjacency_matrix: StrictlyUpperTriangularLogicalMatrix) -> Self {
        Self { adjacency_matrix }
    }

    #[inline]
    pub fn get_vertex_count(&self) -> usize {
        self.adjacency_matrix.size()
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn get_edge(&self, u: usize, v: usize) -> bool {
        assert!(u < self.get_vertex_count());
        assert!(v < self.get_vertex_count());
        assert!(u < v);
        self.adjacency_matrix.get(u, v)
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn set_edge(&mut self, u: usize, v: usize, exists: bool) {
        assert!(u < self.get_vertex_count());
        assert!(v < self.get_vertex_count());
        assert!(u < v);
        self.adjacency_matrix.set(u, v, exists);
    }

    /// Iterates over the edges in an order that favors CPU cache locality.
    pub fn iter_edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.adjacency_matrix.iter_ones()
    }

    /// Iterates over vertices `v` such that there's an edge `(u, v)` in the
    /// DAG.
    pub fn iter_children(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjacency_matrix.iter_ones_at_row(u)
    }

    pub fn extend_with_children(&self, children: &mut Vec<usize>, u: usize) {
        children.extend(self.adjacency_matrix.iter_ones_at_row(u))
    }

    pub fn extend_with_parents(&self, parents: &mut Vec<usize>, v: usize) {
        for u in 0..v {
            if self.get_edge(u, v) {
                parents.push(u);
            }
        }
    }

    /// Consume self and return the underlying adjacency matrix.
    pub fn into_adjacency_matrix(self) -> StrictlyUpperTriangularLogicalMatrix {
        self.adjacency_matrix
    }

    /// Visit all vertices reachable from `vertex` in a depth-first-search (DFS)
    /// order.
    pub fn iter_descendants_dfs(&self, start_vertex: usize) -> Box<dyn Iterator<Item=usize> + '_> {
        let iter = crate::digraph::DfsDescendantsIterator {
            digraph: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: vec![start_vertex],
        };
        let iter = iter.filter(move |vertex| *vertex != start_vertex);
        Box::new(iter)
    }

    pub fn iter_ancestors_dfs(&self, start_vertex: usize) -> Box<dyn Iterator<Item=usize> + '_> {
        let iter = crate::digraph::DfsAncestorsIterator {
            digraph: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: vec![start_vertex],
        };
        let iter = iter.filter(move |vertex| *vertex != start_vertex);
        Box::new(iter)
    }

    /// Visit all vertices of a DAG in a depth-first-search (DFS) order.
    pub fn iter_vertices_dfs(&self) -> Box<dyn Iterator<Item=usize> + '_> {
        let iter = crate::digraph::DfsDescendantsIterator {
            digraph: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: self.get_vertices_without_incoming_edges(),
        };
        Box::new(iter)
    }

    /// Visit all vertices of a DAG in a depth-first-search postorder, i.e. emitting
    /// vertices only after all their descendants have been emitted first.
    pub fn iter_vertices_dfs_post_order(&self) -> Box<dyn Iterator<Item=usize> + '_> {
        let iter = crate::digraph::DfsPostOrderVerticesIterator {
            digraph: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: self.get_vertices_without_incoming_edges(),
        };
        Box::new(iter)
    }

    /// Visit nodes in a depth-first-search (DFS) emitting edges in postorder, i.e.
    /// each node is visited after all its descendants have been already visited.
    ///
    /// Note that when a DAG represents a [partially ordered
    /// set](https://en.wikipedia.org/wiki/Partially_ordered_set), this function iterates over pairs of
    /// that poset.  It may be necessary to first compute either a [`crate::transitive_reduction`] of a
    /// DAG, to only get the minimal set of pairs spanning the entire poset, or a
    /// [`crate::transitive_closure`] to get all the pairs of that poset.
    pub fn iter_edges_dfs_post_order(&self) -> Box<dyn Iterator<Item=(usize, usize)> + '_> {
        let iter = crate::digraph::DfsPostOrderEdgesIterator {
            digraph: self,
            inner: self.iter_vertices_dfs_post_order(),
            seen_vertices: FixedBitSet::with_capacity(self.get_vertex_count()),
            buffer: Default::default(),
        };
        Box::new(iter)
    }

    /// Visit all vertices reachable from `vertex` in a depth-first-search
    /// postorder, i.e. emitting vertices only after all their descendants have been
    /// emitted first.
    pub fn iter_descendants_dfs_post_order(&self, vertex: usize) -> Box<dyn Iterator<Item=usize> + '_> {
        let iter = crate::digraph::DfsPostOrderVerticesIterator {
            digraph: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: vec![vertex],
        };
        Box::new(iter)
    }

    /// Combines [`iter_vertices_dfs_post_order`], [`Iterator::collect()`] and
    /// [`slice::reverse()`] to get a topologically ordered sequence of vertices of a
    /// DAG.
    pub fn get_topologically_ordered_vertices(&self) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::with_capacity(self.get_vertex_count());
        result.extend(self.iter_vertices_dfs_post_order());
        result.reverse();
        result
    }

    /// Computes a mapping: vertex -> set of vertices that are descendants of vertex.
    pub fn get_descendants(&self) -> Vec<FixedBitSet> {
        let mut descendants: Vec<FixedBitSet> = vec![FixedBitSet::default(); self.get_vertex_count()];

        for u in (0..self.get_vertex_count()).rev() {
            let mut u_descendants = FixedBitSet::default();
            for v in self.iter_children(u) {
                u_descendants.union_with(&descendants[v]);
                u_descendants.grow(v + 1);
                u_descendants.set(v, true);
            }
            descendants[u] = u_descendants;
        }

        descendants
    }

    /// Returns a new DAG that is a [transitive
    /// reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of a DAG.
    pub fn transitive_reduction(&self) -> DirectedAcyclicGraph {
        let mut result = self.clone();

        let descendants = self.get_descendants();
        for u in 0..self.get_vertex_count() {
            for v in self.iter_children(u) {
                for w in descendants[v].ones() {
                    if w == v {
                        continue;
                    }
                    result.set_edge(u, w, false);
                }
            }
        }
        result
    }

    /// Returns a new DAG that is a [transitive
    /// closure](https://en.wikipedia.org/wiki/Transitive_closure) of a DAG.
    pub fn transitive_closure(&self) -> DirectedAcyclicGraph {
        let mut result = self.clone();

        // http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci335/lecture_notes/chapter08.pdf

        let descendants = self.get_descendants();
        for u in 0..self.get_vertex_count() {
            for v in descendants[u].ones() {
                result.set_edge(u, v, true);
            }
        }

        result
    }

    /// Returns a set "seed" vertices of a DAG from which a traversal may start so
    /// that the process covers all vertices in the graph.
    pub fn get_vertices_without_incoming_edges(&self) -> Vec<usize> {
        let incoming_edges_count = {
            let mut incoming_edges_count: Vec<usize> = vec![0; self.get_vertex_count()];
            for (_, v) in self.iter_edges() {
                incoming_edges_count[v] += 1;
            }
            incoming_edges_count
        };

        let vertices_without_incoming_edges: Vec<usize> = incoming_edges_count
            .into_iter()
            .enumerate()
            .filter(|(_, indegree)| *indegree == 0)
            .map(|(vertex, _)| vertex)
            .collect();

        vertices_without_incoming_edges
    }


    /// Visit all vertices reachable from `vertex` in a breadth-first-search (BFS)
    /// order.
    pub fn iter_descendants_bfs(&self, vertex: usize) -> BfsVerticesIterator {
        BfsVerticesIterator {
            dag: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: vec![vertex].into(),
        }
    }

    /// Visit all vertices of a DAG in a breadth-first-search (BFS) order.
    pub fn iter_vertices_bfs(&self) -> BfsVerticesIterator {
        BfsVerticesIterator {
            dag: self,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: self.get_vertices_without_incoming_edges().into(),
        }
    }

    /// Outputs the DAG in the [Graphviz DOT](https://graphviz.org/) format.
    pub fn to_dot<W: Write>(
        &self,
        output: &mut W,
    ) -> std::result::Result<(), std::io::Error> {
        writeln!(output, "digraph dag_{} {{", self.get_vertex_count())?;

        for elem in 0..self.get_vertex_count() {
            writeln!(output, "\t_{}[label=\"{}\"];", elem, elem)?;
        }

        writeln!(output, "\n")?;

        for (left, right) in self.iter_edges() {
            writeln!(output, "\t_{} -> _{};", left, right)?;
        }

        writeln!(output, "}}")?;
        Ok(())
    }

    pub fn to_dot_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> std::result::Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        self.to_dot(&mut file)?;
        Ok(())
    }
}

/// Break a DAG into two halves at the vertex `vertex`.  Used as a shrinking
/// strategy for DAGs in the [`quickcheck::Arbitrary`] impl.
///
/// Note that if there are any edges between the left and the right DAGs, they
/// get broken.
///
/// The right DAG is equal in the number of vertices to the left one if `|V|` is
/// an even number or one bigger if `|V|` is odd.
///
/// We don't try to be clever here and compute something expensive like
/// connected components as the property that fails and is undergoing shrinking
/// might as well be "a graph has more than one connected component", in which
/// case that clever shrinking algorithm would be useless.
pub fn break_at(
    dag: &DirectedAcyclicGraph,
    vertex: usize,
) -> (DirectedAcyclicGraph, DirectedAcyclicGraph) {
    let vertex_count = dag.get_vertex_count();
    assert!(vertex < vertex_count);

    let left_vertex_count = vertex;
    let mut left = DirectedAcyclicGraph::empty(left_vertex_count);
    for u in 0..left_vertex_count {
        for v in (u + 1)..left_vertex_count {
            left.set_edge(u, v, dag.get_edge(u, v));
        }
    }

    let right_vertex_count = vertex_count - left_vertex_count;
    let mut right = DirectedAcyclicGraph::empty(right_vertex_count);
    for u in left_vertex_count..vertex_count {
        for v in (u + 1)..vertex_count {
            right.set_edge(
                u - left_vertex_count,
                v - left_vertex_count,
                dag.get_edge(u, v),
            );
        }
    }

    (left, right)
}

impl Arbitrary for DirectedAcyclicGraph {
    fn arbitrary(g: &mut Gen) -> Self {
        let vertex_count = g.size();
        let seed = u64::arbitrary(g);
        let mut rng = StdRng::seed_from_u64(seed);
        let dag =
            DirectedAcyclicGraph::random(vertex_count, &mut rng, Bernoulli::new(0.75).unwrap());
        dag
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = DirectedAcyclicGraph>> {
        let vertex_count = self.get_vertex_count();
        if vertex_count < 2 {
            return Box::new(vec![].into_iter());
        }
        let (left, right) = break_at(self, vertex_count / 2);
        Box::new(vec![left, right].into_iter())
    }
}

pub fn arb_dag(max_vertex_count: usize) -> BoxedStrategy<DirectedAcyclicGraph> {
    (1..max_vertex_count)
        .prop_flat_map(|vertex_count| {
            let max_edges_count =
                strictly_upper_triangular_logical_matrix::strictly_upper_triangular_matrix_capacity(
                    vertex_count,
                );
            proptest::bits::bool_vec::between(0, max_edges_count).prop_flat_map(move |boolvec| {
                let dag = DirectedAcyclicGraph::from_random_edges(vertex_count, &boolvec);
                Just(dag).boxed()
            })
        })
        .boxed()
}

/// See [`iter_vertices_bfs`].
pub struct BfsVerticesIterator<'a> {
    dag: &'a DirectedAcyclicGraph,
    visited: FixedBitSet,
    to_visit: VecDeque<usize>,
}

impl<'a> Iterator for BfsVerticesIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.to_visit.pop_front() {
            if self.visited[u] {
                continue;
            }
            self.visited.insert(u);
            self.to_visit
                .extend(self.dag.iter_children(u).filter(|v| !self.visited[*v]));
            return Some(u);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use super::*;

    use quickcheck::{quickcheck, Arbitrary, Gen};

    #[test]
    #[should_panic = "assertion failed: u < v"]
    fn negative_test_smallest_dag() {
        let mut dag = DirectedAcyclicGraph::empty(2);
        assert_eq!(dag.get_edge(0, 0), false);
        dag.set_edge(0, 0, true);
    }

    #[test]
    fn divisibility_poset_of_12_ordered_pairs() {
        let divisibility_poset_pairs = vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 4),
            (2, 6),
            (2, 8),
            (2, 10),
            (2, 12),
            (3, 6),
            (3, 9),
            (3, 12),
            (4, 8),
            (4, 12),
            (5, 10),
            (6, 12),
        ];
        let dag =
            DirectedAcyclicGraph::from_edges_iter(12 + 1, divisibility_poset_pairs.into_iter());
        let dag = dag.transitive_reduction();

        let dag_pairs: HashSet<(usize, usize)> =
            HashSet::from_iter(dag.iter_edges_dfs_post_order());
        let expected = HashSet::from_iter(vec![
            (3, 9),
            (2, 6),
            (6, 12),
            (1, 7),
            (1, 11),
            (5, 10),
            (3, 6),
            (2, 10),
            (1, 2),
            (4, 12),
            (2, 4),
            (4, 8),
            (1, 5),
            (1, 3),
        ]);
        assert_eq!(dag_pairs, expected);
    }

    // This mostly ensures `iter_edges()` really returns *all* the edges.
    fn prop_dag_reduces_to_nothing(mut dag: DirectedAcyclicGraph) -> bool {
        let mut edges: Vec<(usize, usize)> = dag.iter_edges().collect();
        while let Some((left, right)) = edges.pop() {
            dag.set_edge(left, right, false);
        }
        let edges: Vec<(usize, usize)> = dag.iter_edges().collect();
        edges.is_empty()
    }

    quickcheck! {
        fn random_dag_reduces_to_nothing(dag: DirectedAcyclicGraph) -> bool {
            println!("{:?}", dag);
            prop_dag_reduces_to_nothing(dag)
        }
    }

    /// Does not include the trivial divisors: k | k for every integer k.
    #[derive(Clone, Debug)]
    struct IntegerDivisibilityPoset {
        number: usize,
        divisors_of: BTreeMap<usize, Vec<usize>>,
    }

    impl IntegerDivisibilityPoset {
        fn get_divisors(number: usize) -> BTreeMap<usize, Vec<usize>> {
            let mut result: BTreeMap<usize, Vec<usize>> = Default::default();
            let mut numbers: Vec<usize> = vec![number];
            while let Some(n) = numbers.pop() {
                let divisors_of_n: Vec<usize> =
                    (1..n / 2 + 1).filter(|d| n % d == 0).rev().collect();
                for divisor in &divisors_of_n {
                    if !result.contains_key(&divisor) {
                        numbers.push(*divisor);
                    }
                }
                result.insert(n, divisors_of_n);
            }
            result
        }

        fn of_number(number: usize) -> Self {
            IntegerDivisibilityPoset {
                number,
                divisors_of: Self::get_divisors(number),
            }
        }

        fn get_pairs(&self) -> Vec<(usize, usize)> {
            let mut result = Vec::new();

            for divisor in self.divisors_of.keys() {
                result.extend(
                    self.divisors_of[&divisor]
                        .iter()
                        .map(|dividend| (*dividend, *divisor)),
                );
            }

            result
        }
    }

    impl Arbitrary for IntegerDivisibilityPoset {
        fn arbitrary(g: &mut Gen) -> Self {
            let range: Vec<usize> = (3..g.size()).collect();
            IntegerDivisibilityPoset::of_number(*g.choose(&range).unwrap())
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = IntegerDivisibilityPoset>> {
            if self.number == 1 {
                return Box::new(vec![].into_iter());
            }
            let new_number = self.number / 2;
            let smaller_number: usize = *self
                .divisors_of
                .keys()
                .filter(|&k| *k <= new_number)
                .max()
                .unwrap();
            Box::new(vec![IntegerDivisibilityPoset::of_number(smaller_number)].into_iter())
        }
    }

    fn prop_integer_divisibility_poset_isomorphism(
        integer_divisibility_poset: IntegerDivisibilityPoset,
    ) -> bool {
        println!(
            "{:10} {:?}",
            integer_divisibility_poset.number, integer_divisibility_poset.divisors_of
        );

        let pairs = integer_divisibility_poset.get_pairs();

        let dag = DirectedAcyclicGraph::from_edges_iter(
            integer_divisibility_poset.number + 1,
            pairs.iter().cloned(),
        );

        for (left, right) in pairs {
            assert!(dag.get_edge(left, right), "({}, {})", left, right);
        }

        for (left, right) in dag.iter_edges() {
            assert!(right % left == 0, "({}, {})", left, right);
        }

        true
    }

    #[test]
    fn integer_divisibility_poset_isomorphism() {
        let gen = quickcheck::Gen::new(1000);
        quickcheck::QuickCheck::new().gen(gen).quickcheck(
            prop_integer_divisibility_poset_isomorphism as fn(IntegerDivisibilityPoset) -> bool,
        );
    }

    #[test]
    fn divisibility_poset_12_children() {
        let divisibility_poset_pairs = vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 4),
            (2, 6),
            (2, 8),
            (2, 10),
            (2, 12),
            (3, 6),
            (3, 9),
            (3, 12),
            (4, 8),
            (4, 12),
            (5, 10),
            (6, 12),
        ];
        let dag =
            DirectedAcyclicGraph::from_edges_iter(12 + 1, divisibility_poset_pairs.into_iter());
        assert_eq!(dag.iter_children(12).collect::<Vec<usize>>(), vec![]);
        assert_eq!(dag.iter_children(11).collect::<Vec<usize>>(), vec![]);
        assert_eq!(dag.iter_children(9).collect::<Vec<usize>>(), vec![]);
        assert_eq!(dag.iter_children(8).collect::<Vec<usize>>(), vec![]);
        assert_eq!(dag.iter_children(7).collect::<Vec<usize>>(), vec![]);
        assert_eq!(dag.iter_children(6).collect::<Vec<usize>>(), vec![12]);
        assert_eq!(dag.iter_children(5).collect::<Vec<usize>>(), vec![10]);
        assert_eq!(dag.iter_children(4).collect::<Vec<usize>>(), vec![8, 12]);
        assert_eq!(dag.iter_children(3).collect::<Vec<usize>>(), vec![6, 9, 12]);
        assert_eq!(
            dag.iter_children(2).collect::<Vec<usize>>(),
            vec![4, 6, 8, 10, 12]
        );
        assert_eq!(
            dag.iter_children(1).collect::<Vec<usize>>(),
            vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    #[test]
    fn divisibility_poset_of_12_descendants() {
        let divisibility_poset_pairs = vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 4),
            (2, 6),
            (2, 8),
            (2, 10),
            (2, 12),
            (3, 6),
            (3, 9),
            (3, 12),
            (4, 8),
            (4, 12),
            (5, 10),
            (6, 12),
        ];
        let dag =
            DirectedAcyclicGraph::from_edges_iter(12 + 1, divisibility_poset_pairs.into_iter());
        let descendants = dag.get_descendants();
        assert_eq!(descendants[12], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[11], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[10], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[9], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[8], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[7], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[6].ones().collect::<Vec<usize>>(), vec![12]);
        assert_eq!(descendants[5].ones().collect::<Vec<usize>>(), vec![10]);
        assert_eq!(descendants[4].ones().collect::<Vec<usize>>(), vec![8, 12]);
        assert_eq!(
            descendants[3].ones().collect::<Vec<usize>>(),
            vec![6, 9, 12]
        );
        assert_eq!(
            descendants[2].ones().collect::<Vec<usize>>(),
            vec![4, 6, 8, 10, 12]
        );
        assert_eq!(
            descendants[1].ones().collect::<Vec<usize>>(),
            vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    fn prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
        dag: DirectedAcyclicGraph,
    ) -> bool {
        println!("{:?}", dag);
        let transitive_closure: HashSet<(usize, usize)> =
            dag.transitive_closure().iter_edges().collect();
        let transitive_reduction: HashSet<(usize, usize)> =
            dag.transitive_reduction().iter_edges().collect();
        let intersection: HashSet<(usize, usize)> = transitive_closure
            .intersection(&transitive_reduction)
            .cloned()
            .collect();
        intersection == transitive_reduction
    }

    #[test]
    fn transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
    ) {
        quickcheck::QuickCheck::new().quickcheck(prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order as fn(DirectedAcyclicGraph) -> bool);
    }

    #[test]
    fn divisibility_poset_of_12_dfs_descendants() {
        let divisibility_poset_pairs = vec![
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 4),
            (2, 6),
            (2, 8),
            (2, 10),
            (2, 12),
            (3, 6),
            (3, 9),
            (3, 12),
            (4, 8),
            (4, 12),
            (5, 10),
            (6, 12),
        ];
        let dag =
            DirectedAcyclicGraph::from_edges_iter(12 + 1, divisibility_poset_pairs.into_iter());

        assert_eq!(
            dag.iter_descendants_dfs(12).collect::<Vec<usize>>(),
            vec![12]
        );
        assert_eq!(
            dag.iter_descendants_dfs(11).collect::<Vec<usize>>(),
            vec![11]
        );
        assert_eq!(
            dag.iter_descendants_dfs(6).collect::<Vec<usize>>(),
            vec![6, 12]
        );
    }

    fn prop_traversals_equal_modulo_order(dag: DirectedAcyclicGraph) {
        let bfs: HashSet<usize> = dag.iter_vertices_bfs().collect();
        let dfs: HashSet<usize> = dag.iter_vertices_dfs().collect();
        let dfs_post_order: HashSet<usize> = dag.iter_vertices_dfs_post_order().collect();
        assert_eq!(bfs, dfs);
        assert_eq!(dfs_post_order, dfs);
        assert_eq!(dfs_post_order, bfs);
    }

    #[test]
    fn traversals_equal_modulo_order() {
        quickcheck::QuickCheck::new()
            .quickcheck(prop_traversals_equal_modulo_order as fn(DirectedAcyclicGraph));
    }
}
