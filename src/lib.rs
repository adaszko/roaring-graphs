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
//!    test whether adding an edge would form a cycle: edges are only allowed to
//!    go "forward", i.e. from `u` to `v` iff `u < v`.  Otherwise we panic.
//! 1. Vertices numbering starts at 0.
//! 1. The number of vertices is determined at construction time and
//!    growing/shrinking generally requires a new graph to be constructed.
//!
//! In exchange for these assumptions you get these useful properties:
//! * **Correctness**: It's not possible to have cycles by construction!
//! * **Low memory usage**: The representation is *compact*: edges are just bits
//!   in a bit set.  The implementation uses just `(|V|*|V|-|V|)/2` *bits* of
//!   memory + a constant.
//! * **Good CPU cache locality**: Edges are stored in a [row-major packed
//!   representation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
//!   so that iteration over the neighbours of a vertex is just an iteration
//!   over *consecutive* bits in a bit set.
//! * **Low cognitive overhead**: No need to deal with type-level shenenigans to
//!   get basic tasks done.
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
//! * No serde impls.  Simply serialize/deserialize the list of edges with a
//!   library of your choosing.
//!
//! # Entry points
//!
//! See either [`DirectedAcyclicGraph::empty`] or
//! [`DirectedAcyclicGraph::from_edges_iter`] for the "entry point" to this
//! crate.

use std::io::Write;

use quickcheck::{Arbitrary, Gen};

mod strictly_upper_triangular_logical_matrix;
use rand::{prelude::StdRng, Rng, SeedableRng};
use rand_distr::{Bernoulli, Distribution};
pub use strictly_upper_triangular_logical_matrix::StrictlyUpperTriangularLogicalMatrix;

pub mod algorithm;
pub mod traversal;

#[derive(Clone)]
pub struct DirectedAcyclicGraph {
    adjacency_matrix: StrictlyUpperTriangularLogicalMatrix,
}

impl std::fmt::Debug for DirectedAcyclicGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones: Vec<(usize, usize)> = self.iter_edges().collect();
        write!(
            f,
            "DirectedAcyclicGraph::from_edges({}, vec!{:?}.iter().cloned())",
            self.get_vertex_count(),
            ones
        )?;
        Ok(())
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
/// connected components as the property that fails might as well be "a graph
/// has more than one connected component", in which case that clever shrinking
/// algorithm would be useless.
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
        let dag = algorithm::transitive_reduction(&dag);

        let dag_pairs: HashSet<(usize, usize)> =
            HashSet::from_iter(traversal::iter_edges_dfs_post_order(&dag));
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
}
