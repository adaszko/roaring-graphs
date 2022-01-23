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
//!    Otherwise we panic. [^1]
//! 1. Vertices numbering starts at 0.
//! 1. The number of vertices is determined at construction time and
//!    growing/shrinking generally requires a new graph to be constructed.
//!
//! In exchange for these assumptions you get these useful properties:
//! * It's not possible to represent a graph with the [`DirectedAcyclicGraph`]
//!   data type that's not a DAG, contrary to a fully general graph
//!   representation like adjacency lists or a square matrix.  IOW: Every
//!   strictly upper triangular matrix represents *some* valid DAG.  At the same
//!   time, every DAG is represented by some strictly upper triangular matrix.
//! * The representation is *compact*: edges are just bits in a bit set.
//!   Iteration over the edges of some vertex is just iteration over bits in a
//!   bit set, so it's CPU-cache-friendly. That's nod at [Data Oriented
//!   Design](https://en.wikipedia.org/wiki/Data-oriented_design). [^2]
//! * Generating a random DAG is a linear operation, contrary to a fully general
//!   graph representation.  That was actually the original motivation for
//!   writing this crate.  It can be used with
//!   [quickcheck](https://crates.io/crates/quickcheck) efficiently.  In fact,
//!   [`DirectedAcyclicGraph`] implements [`quickcheck::Arbitrary`] (with
//!   meaningful shrinking).
//!
//! ## Missing features
//!
//! * No support for storing anything in the vertices.  This may be done on the
//!   caller's side with a bidirectional mapping to integer vertices.
//! * No support for assigning weights to either edges or vertices.  Again, this
//!   may be done on the caller's side with a bidirectional mapping.
//!
//!   [^1]: You can always maintain a bidirectional mapping from your domain to
//!    integers if you need some other type.
//!
//!   [^2]: Note that currently, the implementation uses `|v|^2` bits, instead
//!   of the optimal `(|v|^2 - |v|) / 2` bits.  This will most likely be
//!   optimized in the future.
//!
//! # Entry points
//!
//! See either [`DirectedAcyclicGraph::empty`] or
//! [`DirectedAcyclicGraph::from_edges`] for the "entry point" to this crate.

use std::{io::Write, collections::{HashSet, VecDeque}};

#[cfg(feature = "qc")]
use quickcheck::{Gen, Arbitrary};

mod strictly_upper_triangular_matrix;
pub use strictly_upper_triangular_matrix::StrictlyUpperTriangularMatrix;
pub use strictly_upper_triangular_matrix::NeighboursIterator;
pub use strictly_upper_triangular_matrix::EdgesIterator;


#[derive(Clone)]
pub struct DirectedAcyclicGraph {
    adjacency_matrix: StrictlyUpperTriangularMatrix,
}


impl std::fmt::Debug for DirectedAcyclicGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones: Vec<(usize, usize)> = self.iter_edges().collect();
        write!(
            f,
            "DirectedAcyclicGraph::from_edges({}, vec!{:?})",
            self.vertex_count(),
            ones
        )?;
        Ok(())
    }
}


pub struct OrderedPosetPairsIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularMatrix,
    vertices_with_no_incoming_edges: Vec<usize>,
    to_visit: VecDeque<(usize, usize)>,
    visited: HashSet<(usize, usize)>,
}

impl<'a> Iterator for OrderedPosetPairsIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while let Some((u, v)) = self.to_visit.pop_front() {
                if self.visited.contains(&(u, v)) {
                    continue;
                }
                let neighbours: Vec<usize> = self.adjacency_matrix.iter_neighbours(v).collect();
                self.to_visit.extend(neighbours.into_iter().map(|z| (v, z)));
                self.visited.insert((u, v));
                return Some((u, v));
            }

            if let Some(u) = self.vertices_with_no_incoming_edges.pop() {
                let neighbours: Vec<usize> = self.adjacency_matrix.iter_neighbours(u).collect();
                self.to_visit.extend(neighbours.into_iter().map(|v| (u, v)));
            }
            else {
                return None;
            }
        }
    }
}


impl DirectedAcyclicGraph {
    pub fn empty(vertex_count: usize) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularMatrix::zeroed(vertex_count),
        }
    }

    /// Constructs a DAG from a list of edges.
    ///
    /// Requires `u < vertex_count && v < vertex_count && u < v` for every edge
    /// `(u, v)` in `edges`.  Panics otherwise.
    pub fn from_edges(vertex_count: usize, edges: &[(usize, usize)]) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularMatrix::from_ones(vertex_count, edges),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.adjacency_matrix.size()
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn get_edge(&self, u: usize, v: usize) -> bool {
        assert!(u < self.vertex_count());
        assert!(v < self.vertex_count());
        assert!(u < v);
        self.adjacency_matrix.get(u, v)
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn set_edge(&mut self, u: usize, v: usize, exists: bool) {
        assert!(u < self.vertex_count());
        assert!(v < self.vertex_count());
        assert!(u < v);
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

    /// Outputs the DAG in the [Graphviz DOT](https://graphviz.org/) format.
    pub fn to_dot<W: Write>(&self, output: &mut W) -> std::result::Result<(), std::io::Error> {
        writeln!(output, "digraph dag_{} {{", self.vertex_count())?;

        let elements: Vec<usize> = (0..self.vertex_count()).collect();
        for elem in elements {
            writeln!(output, "\t_{}[label=\"{}\"];", elem, elem)?;
        }

        writeln!(output, "\n")?;

        for (left, right) in self.iter_edges() {
            writeln!(output, "\t_{} -> _{};", left, right)?;
        }

        writeln!(output, "}}")?;
        Ok(())
    }

    /// When the DAG represents a [Partially Ordered
    /// Set](https://en.wikipedia.org/wiki/Partially_ordered_set), it's useful
    /// to enumerate all the Poset pairs in a fashion that preserves the
    /// underlying order.
    pub fn iter_ordered_poset_pairs(&self) -> OrderedPosetPairsIterator {
        let vertex_count = self.vertex_count();

        let mut incoming_edges_count: Vec<usize> = vec![0; vertex_count];
        self.iter_edges().for_each(|(_, right)| {
            incoming_edges_count[right] += 1;
        });

        let vertices_with_no_incoming_edges: Vec<usize> = incoming_edges_count
            .into_iter()
            .enumerate()
            .filter(|(_, indegree)| *indegree == 0)
            .map(|(vertex, _)| vertex)
            .collect();
        let to_visit: VecDeque<(usize, usize)> = VecDeque::with_capacity(vertex_count);
        let visited: HashSet<(usize, usize)> = HashSet::with_capacity(vertex_count);

        OrderedPosetPairsIterator {
            adjacency_matrix: &self.adjacency_matrix,
            vertices_with_no_incoming_edges,
            to_visit,
            visited,
        }
    }
}

#[cfg(feature = "qc")]
impl Arbitrary for DirectedAcyclicGraph {
    fn arbitrary(g: &mut Gen) -> Self {
        let vertex_count = g.size();
        let mut dag = DirectedAcyclicGraph::empty(vertex_count);

        for u in 0..vertex_count {
            for v in (u+1)..vertex_count {
                dag.set_edge(u, v, Arbitrary::arbitrary(g));
            }
        }

        dag
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = DirectedAcyclicGraph>> {
        let vertex_count = self.vertex_count();

        if vertex_count < 2 {
            return Box::new(vec![].into_iter());
        }

        let left_vertex_count = vertex_count / 2;
        let mut left = DirectedAcyclicGraph::empty(left_vertex_count);
        for u in 0..left_vertex_count {
            for v in (u+1)..left_vertex_count {
                left.set_edge(u, v, self.get_edge(u, v));
            }
        }

        let right_vertex_count = vertex_count - left_vertex_count;
        let mut right = DirectedAcyclicGraph::empty(right_vertex_count);
        for u in left_vertex_count..vertex_count {
            for v in (left_vertex_count+1)..vertex_count {
                right.set_edge(u - left_vertex_count, v - left_vertex_count, self.get_edge(u, v));
            }
        }

        Box::new(vec![left, right].into_iter())
    }
}


#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use quickcheck::{quickcheck, Arbitrary, Gen};
    use super::*;

    #[test]
    #[should_panic = "assertion failed: u < v"]
    fn negative_test_smallest_dag() {
        let mut dag = DirectedAcyclicGraph::empty(2);
        assert_eq!(dag.get_edge(0, 0), false);
        dag.set_edge(0, 0, true);
    }

    #[test]
    fn divisibility_poset_of_12_ordered_pairs() {
        let divisibility_poset_pairs: Vec<(usize, usize)> = vec![
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12),
            (2, 4), (2, 6), (2, 8), (2, 10), (2, 12),
            (3, 6), (3, 9), (3, 12),
            (4, 8), (4, 12),
            (5, 10),
            (6, 12)
        ];
        let dag = DirectedAcyclicGraph::from_edges(12+1, &divisibility_poset_pairs);
        let total_order: Vec<(usize, usize)> = dag.iter_ordered_poset_pairs().collect();
        println!("{:?}", total_order);
        assert_eq!(total_order, divisibility_poset_pairs);
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
            let mut numbers: VecDeque<usize> = vec![number].into();
            while let Some(n) = numbers.pop_front() {
                let divisors_of_n: Vec<usize> = (1..n/2+1).filter(|d| n % d == 0).collect::<Vec<usize>>();
                for divisor in &divisors_of_n {
                    if !result.contains_key(&divisor) {
                        numbers.push_back(*divisor);
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

            let divisors: Vec<usize> = self.divisors_of.keys().cloned().collect();
            for divisor in divisors {
                let dividends: Vec<usize> = self.divisors_of[&divisor].to_vec();
                result.extend(dividends.iter().map(|dividend| (*dividend, divisor)));
            }

            result
        }
    }

    impl Arbitrary for IntegerDivisibilityPoset {
        fn arbitrary(g: &mut Gen) -> Self {
            let range: Vec<usize> = (3..g.size()).collect();
            IntegerDivisibilityPoset::of_number(*g.choose(&range).unwrap())
        }

        fn shrink(&self) -> Box<dyn Iterator<Item=IntegerDivisibilityPoset>> {
            if self.number == 1 {
                return Box::new(vec![].into_iter());
            }
            let new_number = self.number / 2;
            let smaller_number: usize = *self.divisors_of.keys().filter(|&k| *k <= new_number).max().unwrap();
            Box::new(vec![IntegerDivisibilityPoset::of_number(smaller_number)].into_iter())
        }
    }

    fn prop_integer_divisibility_poset_isomorphism(integer_divisibility_poset: IntegerDivisibilityPoset) -> bool {
        println!("{:10} {:?}", integer_divisibility_poset.number, integer_divisibility_poset.divisors_of);

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
        quickcheck::QuickCheck::new().gen(gen).quickcheck(prop_integer_divisibility_poset_isomorphism as fn(IntegerDivisibilityPoset) -> bool);
    }
}
