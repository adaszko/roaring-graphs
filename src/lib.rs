//! [Directed Acyclic
//! Graphs](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAGs)
//! represented as [Strictly Upper Triangular
//! matrices](https://mathworld.wolfram.com/StrictlyUpperTriangularMatrix.html).
//!
//! This crate is best suited for cases when you need to work with graphs and
//! you know upfront they are going to fall within the the DAG category. In such
//! case, the genericity of other graph crates may result in runtime bugs that
//! could have been avoided given a more restrictive graph representation. Some
//! graph algorithms also have better time complexity if you assume you're
//! working with a certain class of graphs.
//!
//! There are several assumptions this crate imposes on *your* code:
//!
//! 1. DAG vertices are integer numbers (`usize`) which is used to trivially
//!    test whether adding an edge would form a cycle.  It is simply stipulated
//!    that an edge can only go from a node `u` to a node `v` when `u < v`.
//!    Otherwise we panic with a [`debug_assert`].
//! 1. Vertices numbering starts at 0.
//! 1. The number of vertices is determined at construction time and
//!    growing/shrinking generally requires a new graph to be constructed.
//!
//! In exchange for these assumptions you get these useful properties:
//! * **Correctness**: It's not possible to represent a graph with the
//!   [`DirectedAcyclicGraph`] data type that's not a DAG, contrary to a fully
//!   general graph representation like adjacency lists or a square matrix. IOW:
//!   Every strictly upper triangular matrix represents *some* valid DAG. At the
//!   same time, every DAG is represented by some strictly upper triangular
//!   matrix.
//! * **Efficiency**: The representation is *compact*: edges are just bits in a
//!   bit set.  The implementation uses just `(n*n-n)/2` *bits* of memory + a
//!   constant, where `n` is the number of vertices.
//! * **Efficiency**: The chosen matrix representation is a [row-major packed
//!   representation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
//!   so that iteration over the edges of a vertex is just an iteration over
//!   *consecutive* bits in a bit set, so it has good CPU cache locality.
//! * **Efficiency**: Generating a random DAG is a linear operation, contrary to
//!   a fully general graph representation.  That was actually the original
//!   motivation for writing this crate.  It can be used with
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
//!
//! # Entry points
//!
//! See either [`DirectedAcyclicGraph::empty`] or
//! [`DirectedAcyclicGraph::from_edges`] for the "entry point" to this crate.

use std::collections::VecDeque;
use std::io::Write;

use fixedbitset::FixedBitSet;
#[cfg(feature = "qc")]
use quickcheck::{Arbitrary, Gen};

mod strictly_upper_triangular_logical_matrix;
pub use strictly_upper_triangular_logical_matrix::EdgesIterator;
pub use strictly_upper_triangular_logical_matrix::NeighboursIterator;
pub use strictly_upper_triangular_logical_matrix::StrictlyUpperTriangularLogicalMatrix;

#[derive(Clone)]
pub struct DirectedAcyclicGraph {
    adjacency_matrix: StrictlyUpperTriangularLogicalMatrix,
}

impl std::fmt::Debug for DirectedAcyclicGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones: Vec<(usize, usize)> = self.iter_edges().collect();
        write!(
            f,
            "DirectedAcyclicGraph::from_edges({}, &{:?})",
            self.get_vertex_count(),
            ones
        )?;
        Ok(())
    }
}

pub struct ReverseTopologicalOrderVerticesIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularLogicalMatrix,
    visited: FixedBitSet,
    to_visit: Vec<usize>,
}

impl<'a> Iterator for ReverseTopologicalOrderVerticesIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // This is basically a recursive topological sort algorithm with an
        // explicit stack instead of recursion for stack safety and without the
        // final reversal because we don't always need it in the caller.
        loop {
            let u = match self.to_visit.last().copied() {
                Some(u) => u,
                None => return None,
            };
            if self.visited[u] {
                self.to_visit.pop();
                continue;
            }
            let unvisited_neighbours: Vec<usize> = self
                .adjacency_matrix
                .iter_neighbours(u)
                .filter(|v| !self.visited[*v])
                .collect();
            if unvisited_neighbours.is_empty() {
                // We have visited all the descendants of u.  We can now emit u
                // from the iterator.
                self.to_visit.pop();
                self.visited.set(u, true);
                return Some(u);
            }
            self.to_visit.extend(unvisited_neighbours);
        }
    }
}

pub struct PosetPairsIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularLogicalMatrix,
    inner: ReverseTopologicalOrderVerticesIterator<'a>,
    seen_vertices: FixedBitSet,
    buffer: VecDeque<(usize, usize)>,
}

impl<'a> Iterator for PosetPairsIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((u, v)) = self.buffer.pop_front() {
                return Some((u, v));
            }

            let u = self.inner.next()?;

            for v in self.adjacency_matrix.iter_neighbours(u) {
                if self.seen_vertices[v] {
                    self.buffer.push_back((u, v));
                }
            }
            self.seen_vertices.set(u, true);
        }
    }
}

impl DirectedAcyclicGraph {
    pub fn empty(vertex_count: usize) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularLogicalMatrix::zeroed(vertex_count),
        }
    }

    /// Constructs a DAG from a list of edges.
    ///
    /// Requires `u < vertex_count && v < vertex_count && u < v` for every edge
    /// `(u, v)` in `edges`.  Panics with [`debug_assert`] otherwise.
    pub fn from_edges(vertex_count: usize, edges: &[(usize, usize)]) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularLogicalMatrix::from_ones(vertex_count, edges),
        }
    }

    #[inline]
    pub fn get_vertex_count(&self) -> usize {
        self.adjacency_matrix.size()
    }

    /// Requires `u < v`.  Panics with [`debug_assert`] otherwise.
    pub fn get_edge(&self, u: usize, v: usize) -> bool {
        debug_assert!(u < self.get_vertex_count());
        debug_assert!(v < self.get_vertex_count());
        debug_assert!(u < v);
        self.adjacency_matrix.get(u, v)
    }

    /// Requires `u < v`.  Panics with [`debug_assert`] otherwise.
    pub fn set_edge(&mut self, u: usize, v: usize, exists: bool) {
        debug_assert!(u < self.get_vertex_count());
        debug_assert!(v < self.get_vertex_count());
        debug_assert!(u < v);
        self.adjacency_matrix.set(u, v, exists);
    }

    pub fn iter_edges(&self) -> EdgesIterator {
        self.adjacency_matrix.iter_ones()
    }

    /// Iterates over the vertices `v` such that there's an edge `(u, v)` in the
    /// DAG.
    pub fn iter_neighbours(&self, v: usize) -> NeighboursIterator {
        self.adjacency_matrix.iter_neighbours(v)
    }

    /// Note that the order of the vertices is reverse topological one.
    pub fn iter_reachable_vertices_starting_at(
        &self,
        u: usize,
    ) -> ReverseTopologicalOrderVerticesIterator {
        ReverseTopologicalOrderVerticesIterator {
            adjacency_matrix: &self.adjacency_matrix,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: vec![u],
        }
    }

    pub fn iter_reverse_topologically_ordered_vertices(
        &self,
    ) -> ReverseTopologicalOrderVerticesIterator {
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

        ReverseTopologicalOrderVerticesIterator {
            adjacency_matrix: &self.adjacency_matrix,
            visited: FixedBitSet::with_capacity(self.get_vertex_count()),
            to_visit: vertices_without_incoming_edges,
        }
    }

    /// When a DAG represents a [partially ordered
    /// set](https://en.wikipedia.org/wiki/Partially_ordered_set), this method
    /// iterates over all the pairs of that poset.
    pub fn iter_poset_pairs(&self) -> PosetPairsIterator {
        PosetPairsIterator {
            adjacency_matrix: &self.adjacency_matrix,
            inner: self.iter_reverse_topologically_ordered_vertices(),
            seen_vertices: FixedBitSet::with_capacity(self.get_vertex_count()),
            buffer: Default::default(),
        }
    }
}

/// Outputs the DAG in the [Graphviz DOT](https://graphviz.org/) format.
pub fn to_dot<W: Write>(
    dag: &DirectedAcyclicGraph,
    output: &mut W,
) -> std::result::Result<(), std::io::Error> {
    writeln!(output, "digraph dag_{} {{", dag.get_vertex_count())?;

    for elem in 0..dag.get_vertex_count() {
        writeln!(output, "\t_{}[label=\"{}\"];", elem, elem)?;
    }

    writeln!(output, "\n")?;

    for (left, right) in dag.iter_edges() {
        writeln!(output, "\t_{} -> _{};", left, right)?;
    }

    writeln!(output, "}}")?;
    Ok(())
}

/// Returns a new DAG that is a [transitive
/// reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of a DAG.
pub fn transitive_reduction(dag: &DirectedAcyclicGraph) -> DirectedAcyclicGraph {
    let mut result = dag.clone();

    for u in 0..dag.get_vertex_count() {
        for v in dag.iter_neighbours(u) {
            for w in dag.iter_reachable_vertices_starting_at(v) {
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
pub fn transitive_closure(dag: &DirectedAcyclicGraph) -> DirectedAcyclicGraph {
    let mut result = dag.clone();

    for u in 0..dag.get_vertex_count() {
        for v in dag.iter_neighbours(u) {
            for w in dag.iter_reachable_vertices_starting_at(v) {
                if w == v {
                    continue;
                }
                result.set_edge(u, w, true);
            }
        }
    }
    result
}

/// Simply calls `collect()` plus `reverse()` on the result of
/// [`DirectedAcyclicGraph::iter_reverse_topologically_ordered_vertices()`].
pub fn get_topologically_ordered_vertices(dag: &DirectedAcyclicGraph) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(dag.get_vertex_count());
    result.extend(dag.iter_reverse_topologically_ordered_vertices());
    result.reverse();
    result
}

#[cfg(feature = "qc")]
impl Arbitrary for DirectedAcyclicGraph {
    fn arbitrary(g: &mut Gen) -> Self {
        let vertex_count = g.size();
        let mut dag = DirectedAcyclicGraph::empty(vertex_count);

        // XXX This could be just FixedBitSet::arbitrary(g) because every
        // strictly upper triangular matrix represents some DAG.
        for u in 0..vertex_count {
            for v in (u + 1)..vertex_count {
                dag.set_edge(u, v, Arbitrary::arbitrary(g));
            }
        }

        dag
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = DirectedAcyclicGraph>> {
        let vertex_count = self.get_vertex_count();

        if vertex_count < 2 {
            return Box::new(vec![].into_iter());
        }

        let left_vertex_count = vertex_count / 2;
        let mut left = DirectedAcyclicGraph::empty(left_vertex_count);
        for u in 0..left_vertex_count {
            for v in (u + 1)..left_vertex_count {
                left.set_edge(u, v, self.get_edge(u, v));
            }
        }

        let right_vertex_count = vertex_count - left_vertex_count;
        let mut right = DirectedAcyclicGraph::empty(right_vertex_count);
        for u in left_vertex_count..vertex_count {
            for v in (left_vertex_count + 1)..vertex_count {
                right.set_edge(
                    u - left_vertex_count,
                    v - left_vertex_count,
                    self.get_edge(u, v),
                );
            }
        }

        Box::new(vec![left, right].into_iter())
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
        let dag = DirectedAcyclicGraph::from_edges(12 + 1, &divisibility_poset_pairs);
        let dag = transitive_reduction(&dag);
        let dag_pairs: HashSet<(usize, usize)> = HashSet::from_iter(dag.iter_poset_pairs());
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

        let dag = DirectedAcyclicGraph::from_edges(integer_divisibility_poset.number + 1, &pairs);

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
}
