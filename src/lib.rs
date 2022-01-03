//! [Directed Acyclic
//! Graphs](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAGs)
//! represented as [Strictly Upper Triangular
//! matrices](https://mathworld.wolfram.com/StrictlyUpperTriangularMatrix.html).
//!
//! There are several assumptions this crate imposes on *your* code:
//!
//! 1. The number of vertices is determined at construction time and
//!    growing/shrinking is generally an expensive operation.
//! 1. DAG vertices are integers (`usize` to be precise).  Although you can
//!    always maintain a bidirectional mapping from your domain to integers if
//!    you need some other type.
//! 1. Vertices numbering starts at 0.
//! 1. For every edge `(u, v)` in the DAG, it holds that `u < v`.  This ensures
//!    there are no cycles.
//!
//! In exchange for these assumptions you get these useful properties:
//! * It's not possible to represent a graph with the [`DirectedAcyclicGraph`]
//!   data type that's not a DAG, contrary to a fully general graph
//!   representation, so fewer runtime bugs.
//! * The representation is *compact*: edges are just bits in a bit set.  Note
//!   that currently, we use `|v|^2` bits, instead of the optimal `(|v|^2 - |v|)
//!   / 2` bits.  This will most likely be optimized in the future.
//! * The representation is CPU-cache-friendly, so traversals are fast.
//! * Generating a random DAG is a linear operation, contrary to a fully general
//!   graph representations.  That was the original motivation for writing this
//!   crate.  It can be used with
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
//! * Bare minimum of provided graph algorithms: neighbours, traversals.

use std::{io::Write, collections::{HashSet}};
use thiserror::Error;

#[cfg(feature = "qc")]
use quickcheck::{Gen, Arbitrary};

mod strictly_upper_triangular_matrix;
pub use strictly_upper_triangular_matrix::StrictlyUpperTriangularMatrix;
pub use strictly_upper_triangular_matrix::EdgesIterator;


#[derive(Error, Debug)]
pub enum DiagError {
    #[error("I/0 Error")]
    IoError(#[from] std::io::Error),
}


type Result<T> = std::result::Result<T, DiagError>;



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


pub struct NeighboursIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularMatrix,
    left_vertex: usize,
    right_vertex: usize,
    max_right_vertex: usize,
}


impl<'a> Iterator for NeighboursIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.right_vertex <= self.max_right_vertex {
            if self.adjacency_matrix.get(self.left_vertex, self.right_vertex) {
                let result = self.right_vertex;
                self.right_vertex += 1;
                return Some(result);
            }
            self.right_vertex += 1;
        }
        None
    }
}


pub struct TopologicallyOrderedEdgesIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularMatrix,
    vertices_with_no_incoming_edges: Vec<usize>,
    to_visit: Vec<(usize, usize)>,
    visited: HashSet<(usize, usize)>,
}

impl<'a> TopologicallyOrderedEdgesIterator<'a> {
    fn iter_neighbours(&self, u: usize) -> NeighboursIterator {
        NeighboursIterator {
            adjacency_matrix: &self.adjacency_matrix,
            left_vertex: u,
            right_vertex: 0,
            max_right_vertex: self.adjacency_matrix.size() - u - 1,
        }
    }
}

impl<'a> Iterator for TopologicallyOrderedEdgesIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.vertices_with_no_incoming_edges.pop() {
            let neighbours: Vec<usize> = self.iter_neighbours(u).collect();
            for v in neighbours {
                self.to_visit.push((u, v));
            }
            while let Some((u, v)) = self.to_visit.pop() {
                if self.visited.contains(&(u, v)) {
                    continue;
                }
                let neighbours: Vec<usize> = self.iter_neighbours(v).collect();
                for z in neighbours {
                    self.to_visit.push((v, z));
                }
                self.visited.insert((u, v));
                return Some((u, v));
            }
        }
        None
    }
}


impl DirectedAcyclicGraph {
    pub fn empty(vertex_count: usize) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularMatrix::zeroed(vertex_count),
        }
    }

    pub fn from_edges(size: usize, edges: &[(usize, usize)]) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularMatrix::from_ones(size, edges),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.adjacency_matrix.size()
    }

    pub fn get_edge(&self, u: usize, v: usize) -> bool {
        assert!(u < self.vertex_count());
        assert!(v < self.vertex_count());
        assert!(u < v);
        self.adjacency_matrix.get(u, v)
    }

    pub fn set_edge(&mut self, u: usize, v: usize, exists: bool) {
        assert!(u < self.vertex_count());
        assert!(v < self.vertex_count());
        assert!(u < v);
        self.adjacency_matrix.set(u, v, exists);
    }

    pub fn iter_edges(&self) -> EdgesIterator {
        self.adjacency_matrix.iter_ones()
    }

    pub fn iter_neighbours(&self, u: usize) -> NeighboursIterator {
        assert!(u < self.vertex_count());
        NeighboursIterator {
            adjacency_matrix: &self.adjacency_matrix,
            left_vertex: u,
            right_vertex: 0,
            max_right_vertex: (self.vertex_count() - u - 1),
        }
    }

    /// Outputs the DAG in the [Graphviz DOT](https://graphviz.org/) format.
    pub fn to_dot<W: Write>(&self, output: &mut W) -> Result<()> {
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

    /// Convienient function when the DAG represents a [Partially Ordered
    /// Set](https://en.wikipedia.org/wiki/Partially_ordered_set).  It's often
    /// useful to enumerate all the Poset pairs in a fashion that preserves the
    /// underlying order.
    pub fn iter_topologically_ordered_edges(&self) -> TopologicallyOrderedEdgesIterator {
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
        let to_visit: Vec<(usize, usize)> = Vec::with_capacity(vertex_count);
        let visited: HashSet<(usize, usize)> = HashSet::with_capacity(vertex_count);

        TopologicallyOrderedEdgesIterator {
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
    use std::collections::HashMap;

    use quickcheck::{quickcheck, Arbitrary, Gen};
    use super::*;

    #[test]
    #[should_panic = "assertion failed: i < j"]
    fn negative_test_smallest_dag() {
        let mut dag = DirectedAcyclicGraph::empty(2);
        assert_eq!(dag.get_edge(0, 0), false);
        dag.set_edge(0, 0, true);
    }

    #[test]
    fn topological_order_of_divisibility_poset() {
        let edges: Vec<(usize, usize)> = vec![
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12),
            (2, 4), (2, 6), (2, 8), (2, 10), (2, 12),
            (3, 6), (3, 9), (3, 12),
            (4, 8), (4, 12),
            (5, 10),
            (6, 12)
        ].into_iter().map(|(u, v)| (u - 1, v - 1)).collect();
        let dag = DirectedAcyclicGraph::from_edges(12, &edges);
        let topological_order: Vec<(usize, usize)> = dag.iter_topologically_ordered_edges().collect();
        println!("{:?}", topological_order);
    }

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

    #[derive(Clone, Debug)]
    struct DivisibilityPoset {
        seed: usize,
        adjacency_lists: HashMap<usize, Vec<usize>>,
    }

    impl DivisibilityPoset {
        fn get_divisors_adjacency_lists(n: usize) -> HashMap<usize, Vec<usize>> {
            let mut result: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut stack = vec![n];
            while let Some(k) = stack.pop() {
                if result.contains_key(&k) {
                    continue
                }
                let partial = (1..k/2+1).filter(|d| k % d == 0).collect::<Vec<usize>>();
                stack.extend(partial.iter());
                result.insert(k, partial);
            }
            result
        }

        fn new(n: usize) -> Self {
            DivisibilityPoset {
                seed: n,
                adjacency_lists: Self::get_divisors_adjacency_lists(n),
            }
        }

        fn get_max_element(&self) -> usize {
            self.seed
        }

        fn get_pairs(&self) -> Vec<(usize, usize)> {
            let mut result = Vec::new();

            let mut divisors: Vec<usize> = self.adjacency_lists.keys().cloned().collect();
            divisors.sort();
            divisors.reverse();

            for divisor in divisors {
                let mut dividends: Vec<usize> = self.adjacency_lists[&divisor].to_vec();
                dividends.sort();
                dividends.reverse();
                for dividend in dividends {
                    result.push((dividend, divisor))
                }
            }

            result
        }

        fn is_divisor_of(&self, left: usize, right: usize) -> bool {
            if !self.adjacency_lists.contains_key(&left) {
                return false
            }

            let mut visited = HashSet::new();
            let mut stack = vec![left];
            while let Some(divisor) = stack.pop() {
                for dividend in &self.adjacency_lists[&divisor] {
                    if *dividend == right {
                        return true
                    }
                    if !visited.contains(dividend) {
                        stack.push(*dividend)
                    }
                }
                debug_assert!(visited.insert(divisor));
            }

            false
        }
    }

    impl Arbitrary for DivisibilityPoset {
        fn arbitrary(g: &mut Gen) -> Self {
            let range: Vec<usize> = (3..g.size()).collect();
            DivisibilityPoset::new(*g.choose(&range).unwrap())
        }

        fn shrink(&self) -> Box<dyn Iterator<Item=DivisibilityPoset>> {
            let seed = self.seed;
            let divisors: Vec<usize> = self.adjacency_lists.keys().cloned().collect();
            let result = divisors.into_iter().filter(move |d| *d < seed).map(DivisibilityPoset::new);
            Box::new(result)
        }
    }

    fn prop_divisibility_poset_isomorphism(divisibility_poset: &DivisibilityPoset) -> bool {
        let dag = DirectedAcyclicGraph::from_edges(divisibility_poset.get_max_element() + 1, &divisibility_poset.get_pairs());

        for (left, right) in divisibility_poset.get_pairs() {
            assert!(dag.get_edge(left, right));
        }

        for (left, right) in dag.iter_edges() {
            assert!(divisibility_poset.is_divisor_of(right, left));
        }

        true
    }

    quickcheck! {
        fn prop_divisibility_poset_isomorphic_to_containment(divisibility_poset: DivisibilityPoset) -> bool {
            println!("{:10} {} {:?}", divisibility_poset.seed, divisibility_poset.get_max_element(), divisibility_poset.adjacency_lists);
            prop_divisibility_poset_isomorphism(&divisibility_poset)
        }
    }
}
