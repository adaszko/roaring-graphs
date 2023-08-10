//! [Directed Acyclic
//! Graphs](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAGs)
//! represented as [Strictly Upper Triangular
//! matrices](https://mathworld.wolfram.com/StrictlyUpperTriangularMatrix.html).
//!
//! A create for working with DAGs where it is known upfront (i.e. statically)
//! that graphs are directed and there are no cycles.

//! There are several assumptions imposed on *your* code:
//!
//! 1. DAG vertices are integer numbers which is used to trivially
//!    test whether adding an edge would form a cycle: edges are only allowed to
//!    go "forward", i.e. from `u` to `v` iff `u < v`.  Otherwise we panic.
//! 1. Vertices numbering starts at 0.
//! 1. The number of vertices is determined at construction time and
//!    growing/shrinking generally requires a new graph to be constructed.
//!
//! In exchange for these assumptions you get these useful properties:
//! * **Correctness**: Cycles (an illegal state) are unrepresentable.
//! * **Compactness**: Edges are just bits in a bit set.  The implementation
//!   stores edges in a roaring bitmap with one bit per *existing* edge.  IOW: A graph with no
//!   edges uses a constant amount of memory irrespective of its max number of vertices.  A full
//!   graph is basically a bitset of all ones with one bit per edge.
//! * **CPU cache locality**: Edges are stored in a [row-major packed
//!   representation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
//!   so that iteration over the neighbours of a vertex is just an iteration
//!   over *consecutive* bits in a bit set.
//! * **Low cognitive overhead**: No need to deal with type-level shenenigans to
//!   get basic tasks done.
//! * **Asymptotic complexity reduction**: Generating a random DAG is a `O(|E|)`
//!   operation.  That was actually the original motivation for writing this
//!   crate.
//!
//! ## Anti-features
//!
//! * No support for storing anything in the vertices.
//! * No support for assigning weights to either edges or vertices.
//! * No serde impls.  Simply serialize/deserialize the list of edges with a
//!   library of your choosing.
//!
//! # Entry points
//!
//! See either [`DirectedAcyclicGraph::new`],
//! [`DirectedAcyclicGraph::from_edges_iter`], or
//! [`DirectedAcyclicGraph::from_adjacency_matrix`] for the "entry point" to
//! this crate.

use std::collections::VecDeque;
use std::io::Write;
use std::ops::Range;

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::Reason;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use roaring::RoaringBitmap;

use crate::delta_debugging_bitmap::DeltaDebuggingBitmapValueTree;
use crate::strictly_upper_triangular_logical_matrix::{
    strictly_upper_triangular_matrix_capacity, RowColumnIterator,
    StrictlyUpperTriangularLogicalMatrix, strictly_upper_triangular_matrix_index,
};
use crate::{TraversableDirectedGraph, Vertex};

/// A mutable, single-threaded directed acyclic graph.
#[derive(Clone, PartialEq, Eq)]
pub struct DirectedAcyclicGraph {
    adjacency_matrix: StrictlyUpperTriangularLogicalMatrix,
}

impl std::fmt::Debug for DirectedAcyclicGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones: Vec<(Vertex, Vertex)> = self.iter_edges().collect();
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
    fn extend_with_children(&self, u: Vertex, children: &mut Vec<Vertex>) {
        self.extend_with_children(u, children)
    }

    fn extend_with_parents(&self, v: Vertex, parents: &mut Vec<Vertex>) {
        self.extend_with_parents(v, parents)
    }
}

impl DirectedAcyclicGraph {
    /// Constructs a new graph without any edges having at most `vertex_count` vertices.
    pub fn new(vertex_count: Vertex) -> Self {
        Self {
            adjacency_matrix: StrictlyUpperTriangularLogicalMatrix::zeroed(vertex_count),
        }
    }

    /// Constructs a DAG from a iterator of edges.
    ///
    /// Requires `u < vertex_count && v < vertex_count && u < v` for every edge
    /// `(u, v)` in `edges`.  Panics otherwise.
    pub fn from_edges_iter<I: Iterator<Item = (Vertex, Vertex)>>(vertex_count: Vertex, edges: I) -> Self {
        let adjacency_matrix = StrictlyUpperTriangularLogicalMatrix::from_iter(vertex_count, edges);
        Self { adjacency_matrix }
    }

    /// Assumes `edges` is a packed representation of the adjacency matrix representing a strictly upper
    /// triangular matrix.  Such representation has a useful property: (1) Every bit sequence in
    /// such a representation corresponds to some valid DAG and (2) Every DAG corresponds to some
    /// valid bit sequence in such a representation.  Thanks to (1) and (2) taken together, we can
    /// be sure proptest will cover the entire search space of random DAGs.
    pub fn from_raw_edges(vertex_count: Vertex, edges: &[bool]) -> Self {
        assert_eq!(
            edges.len(),
            usize::try_from(strictly_upper_triangular_matrix_capacity(vertex_count)).unwrap(),
        );

        let mut iter = RowColumnIterator::new(vertex_count);
        let mut adjacency_matrix = StrictlyUpperTriangularLogicalMatrix::zeroed(vertex_count);
        for value in edges {
            let (row, column) = iter.next().unwrap();
            if *value {
                adjacency_matrix.set(row, column);
            }
        }

        let dag = DirectedAcyclicGraph::from_adjacency_matrix(adjacency_matrix);
        dag
    }

    /// Construct a DAG from a pre-computed adjacency matrix.
    pub fn from_adjacency_matrix(adjacency_matrix: StrictlyUpperTriangularLogicalMatrix) -> Self {
        Self { adjacency_matrix }
    }

    #[inline]
    pub fn get_vertex_count(&self) -> Vertex {
        self.adjacency_matrix.size()
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn get_edge(&self, u: Vertex, v: Vertex) -> bool {
        assert!(u < self.get_vertex_count());
        assert!(v < self.get_vertex_count());
        assert!(u < v);
        self.adjacency_matrix.get(u, v)
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn set_edge(&mut self, u: Vertex, v: Vertex) {
        assert!(u < self.get_vertex_count());
        assert!(v < self.get_vertex_count());
        assert!(u < v);
        self.adjacency_matrix.set(u, v);
    }

    /// Requires `u < v`.  Panics otherwise.
    pub fn clear_edge(&mut self, u: Vertex, v: Vertex) {
        assert!(u < self.get_vertex_count());
        assert!(v < self.get_vertex_count());
        assert!(u < v);
        self.adjacency_matrix.clear(u, v);
    }

    /// Each emitted pair `(u, v)` is guaranteed to satisfy `u < v`.
    pub fn iter_edges(&self) -> impl Iterator<Item = (Vertex, Vertex)> + '_ {
        self.adjacency_matrix.iter_ones()
    }

    /// Iterates over vertices `v` such that there's an edge `(u, v)` in the
    /// DAG.
    pub fn iter_children(&self, u: Vertex) -> impl Iterator<Item = Vertex> + '_ {
        self.adjacency_matrix.iter_ones_at_row(u)
    }

    pub fn extend_with_children(&self, u: Vertex, children: &mut Vec<Vertex>) {
        children.extend(self.adjacency_matrix.iter_ones_at_row(u))
    }

    pub fn extend_with_parents(&self, v: Vertex, parents: &mut Vec<Vertex>) {
        parents.extend(self.adjacency_matrix.iter_ones_at_column(v))
    }

    /// Consume self and return the underlying adjacency matrix.
    pub fn into_adjacency_matrix(self) -> StrictlyUpperTriangularLogicalMatrix {
        self.adjacency_matrix
    }

    /// Visit all vertices reachable from `vertex` in a depth-first-search (DFS)
    /// order.
    pub fn iter_descendants_dfs(&self, start_vertex: Vertex) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = crate::digraph::DfsDescendantsIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: vec![start_vertex],
        };
        let iter = iter.filter(move |vertex| *vertex != start_vertex);
        Box::new(iter)
    }

    pub fn iter_ancestors_dfs(&self, start_vertex: Vertex) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = crate::digraph::DfsAncestorsIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: vec![start_vertex],
        };
        let iter = iter.filter(move |vertex| *vertex != start_vertex);
        Box::new(iter)
    }

    /// Visit all vertices of a DAG in a depth-first-search (DFS) order.
    pub fn iter_vertices_dfs(&self) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = crate::digraph::DfsDescendantsIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: self.get_vertices_without_incoming_edges(),
        };
        Box::new(iter)
    }

    /// Visit all vertices of a DAG in a depth-first-search postorder, i.e. emitting vertices only
    /// after all their descendants were emitted first.
    pub fn iter_vertices_dfs_post_order(&self) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = crate::digraph::DfsPostOrderVerticesIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: self.get_vertices_without_incoming_edges(),
        };
        Box::new(iter)
    }

    /// Visit nodes in a depth-first-search (DFS) emitting edges in postorder, i.e. each node is
    /// visited after all its descendants were already visited.
    pub fn iter_edges_dfs_post_order(&self) -> Box<dyn Iterator<Item = (Vertex, Vertex)> + '_> {
        let iter = crate::digraph::DfsPostOrderEdgesIterator {
            digraph: self,
            inner: self.iter_vertices_dfs_post_order(),
            seen_vertices: RoaringBitmap::new(),
            buffer: Default::default(),
        };
        Box::new(iter)
    }

    /// Visit all vertices reachable from `vertex` in a depth-first-search
    /// postorder, i.e. emitting vertices only after all their descendants have been
    /// emitted first.
    pub fn iter_descendants_dfs_post_order(
        &self,
        vertex: Vertex,
    ) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = crate::digraph::DfsPostOrderVerticesIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: vec![vertex],
        };
        Box::new(iter)
    }

    /// Combines [`Self::iter_vertices_dfs_post_order`] with [`slice::reverse()`] to get a
    /// topologically ordered sequence of vertices of a DAG.
    pub fn get_topologically_ordered_vertices(&self) -> Vec<Vertex> {
        let mut result: Vec<Vertex> = Vec::with_capacity(self.get_vertex_count().into());
        result.extend(self.iter_vertices_dfs_post_order());
        result.reverse();
        result
    }

    /// Computes a mapping: vertex -> set of vertices that are descendants of vertex.
    pub fn get_descendants(&self) -> Vec<RoaringBitmap> {
        let mut descendants: Vec<RoaringBitmap> =
            vec![RoaringBitmap::default(); self.get_vertex_count().into()];

        for u in (0..self.get_vertex_count()).rev() {
            let mut u_descendants = RoaringBitmap::default();
            for v in self.iter_children(u) {
                u_descendants |= &descendants[usize::from(v)];
                u_descendants.insert(v.into());
            }
            descendants[usize::try_from(u).unwrap()] = u_descendants;
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
                for w in descendants[usize::try_from(v).unwrap()].iter() {
                    let w = Vertex::try_from(w).unwrap();
                    if w == v {
                        continue;
                    }
                    result.clear_edge(u, w);
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
            for v in descendants[usize::try_from(u).unwrap()].iter() {
                let v = Vertex::try_from(v).unwrap();
                result.set_edge(u, v);
            }
        }

        result
    }

    /// Returns a set "seed" vertices of a DAG from which a traversal may start so
    /// that the process covers all vertices in the graph.
    pub fn get_vertices_without_incoming_edges(&self) -> Vec<Vertex> {
        let incoming_edges_count = {
            let mut incoming_edges_count: Vec<Vertex> =
                vec![0; self.get_vertex_count().into()];
            for (_, v) in self.iter_edges() {
                incoming_edges_count[usize::try_from(v).unwrap()] += 1;
            }
            incoming_edges_count
        };

        let vertices_without_incoming_edges: Vec<Vertex> = incoming_edges_count
            .into_iter()
            .enumerate()
            .filter(|(_, indegree)| *indegree == 0)
            .map(|(vertex, _)| Vertex::try_from(vertex).unwrap())
            .collect();

        vertices_without_incoming_edges
    }

    /// Visit all vertices reachable from `vertex` in a breadth-first-search (BFS)
    /// order.
    pub fn iter_descendants_bfs(&self, vertex: Vertex) -> BfsVerticesIterator {
        BfsVerticesIterator {
            dag: self,
            visited: RoaringBitmap::new(),
            to_visit: vec![vertex].into(),
        }
    }

    /// Visit all vertices of a DAG in a breadth-first-search (BFS) order.
    pub fn iter_vertices_bfs(&self) -> BfsVerticesIterator {
        BfsVerticesIterator {
            dag: self,
            visited: RoaringBitmap::new(),
            to_visit: self.get_vertices_without_incoming_edges().into(),
        }
    }

    /// Outputs the DAG in the [Graphviz DOT](https://graphviz.org/) format.
    pub fn to_dot<W: Write>(&self, output: &mut W) -> std::result::Result<(), std::io::Error> {
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

pub fn arb_dag(vertex_count: impl Into<Range<Vertex>>) -> DirectedAcyclicGraphStrategy {
    DirectedAcyclicGraphStrategy {
        vertex_count: vertex_count.into(),
    }
}

#[derive(Debug)]
pub struct DirectedAcyclicGraphStrategy {
    vertex_count: Range<Vertex>,
}

#[derive(Debug)]
pub struct DirectedAcyclicGraphValueTree {
    vertex_count: u16,
    vertex_mask_tree: DeltaDebuggingBitmapValueTree,
    edge_bitmap_tree: DeltaDebuggingBitmapValueTree,
    start_simplifying_edges: bool,
}

impl Strategy for DirectedAcyclicGraphStrategy {
    type Tree = DirectedAcyclicGraphValueTree;

    type Value = DirectedAcyclicGraph;

    fn new_tree(
        &self,
        runner: &mut proptest::test_runner::TestRunner,
    ) -> proptest::strategy::NewTree<Self> {
        // Copied out of self.vertex_count.assert_nonempty(), because that's private to proptest
        if self.vertex_count.is_empty() {
            panic!(
                "Invalid use of empty size range. (hint: did you \
                 accidentally write {}..{} where you meant {}..={} \
                 somewhere?)",
                self.vertex_count.start,
                self.vertex_count.end,
                self.vertex_count.start,
                self.vertex_count.end
            );
        }
        let vertex_count =
            Uniform::new(self.vertex_count.start, self.vertex_count.end - 1).sample(runner.rng());
        let bitmap_size = strictly_upper_triangular_matrix_capacity(vertex_count);
        let iter = (0..bitmap_size as u32).filter(|_| runner.rng().gen_bool(0.5));
        let bitmap =
            RoaringBitmap::from_sorted_iter(iter).map_err(|e| Reason::from(e.to_string()))?;
        let vertex_mask = RoaringBitmap::from_sorted_iter(0..vertex_count as u32).unwrap();

        Ok(DirectedAcyclicGraphValueTree {
            vertex_count,
            vertex_mask_tree: DeltaDebuggingBitmapValueTree::new(vertex_mask),
            edge_bitmap_tree: DeltaDebuggingBitmapValueTree::new(bitmap),
            start_simplifying_edges: false,
        })
    }
}

impl ValueTree for DirectedAcyclicGraphValueTree {
    type Value = DirectedAcyclicGraph;

    fn current(&self) -> Self::Value {
        let edge_map = self.edge_bitmap_tree.current();
        let vertex_mask = self.vertex_mask_tree.current();

        let mut from_dst = 0 as Vertex;
        let mut edges = Vec::with_capacity(strictly_upper_triangular_matrix_capacity(
            self.vertex_count,
        ) as usize);
        for from_src in 0..self.vertex_count {
            if vertex_mask.contains(from_src as u32) {
                let mut to_dst = from_dst + 1;
                for to_src in from_src + 1..self.vertex_count {
                    if vertex_mask.contains(to_src as u32) {
                        let edge_idx = strictly_upper_triangular_matrix_index(
                            from_src,
                            to_src,
                            self.vertex_count,
                        );
                        if edge_map.contains(edge_idx) {
                            edges.push((from_dst, to_dst));
                        }
                        to_dst += 1;
                    }
                }
                from_dst += 1;
            }
        }

        let dag =
            DirectedAcyclicGraph::from_edges_iter(vertex_mask.len() as Vertex, edges.into_iter());
        dag
    }

    fn simplify(&mut self) -> bool {
        if self.start_simplifying_edges {
            self.edge_bitmap_tree.simplify()
        } else {
            if !self.vertex_mask_tree.simplify() {
                self.start_simplifying_edges = true;
            }
            true
        }
    }

    fn complicate(&mut self) -> bool {
        if self.start_simplifying_edges {
            self.edge_bitmap_tree.complicate()
        } else {
            if !self.vertex_mask_tree.complicate() {
                self.start_simplifying_edges = true;
            }
            true
        }
    }
}

/// See [`DirectedAcyclicGraph::iter_vertices_bfs`].
pub struct BfsVerticesIterator<'a> {
    dag: &'a DirectedAcyclicGraph,
    visited: RoaringBitmap,
    to_visit: VecDeque<Vertex>,
}

impl<'a> Iterator for BfsVerticesIterator<'a> {
    type Item = Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.to_visit.pop_front() {
            if self.visited.contains(u.into()) {
                continue;
            }
            self.visited.insert(u.into());
            self.to_visit.extend(
                self.dag
                    .iter_children(u)
                    .filter(|v| !self.visited.contains((*v).into())),
            );
            return Some(u);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Borrow,
        collections::{BTreeMap, HashSet},
    };

    use proptest::test_runner::{TestCaseResult, TestError, TestRunner};

    use super::*;

    #[test]
    #[should_panic = "assertion failed: u < v"]
    fn negative_test_smallest_dag() {
        let mut dag = DirectedAcyclicGraph::new(2);
        assert_eq!(dag.get_edge(0, 0), false);
        dag.set_edge(0, 0);
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

        let dag_pairs: HashSet<(Vertex, Vertex)> = HashSet::from_iter(dag.iter_edges_dfs_post_order());
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

    proptest! {
        // This mostly ensures `iter_edges()` really returns *all* the edges.
        #[test]
        fn unblocking_preserves_transitivity(mut dag in arb_dag(0..25)) {
            println!("{:?}", dag);
            let mut edges: Vec<(Vertex, Vertex)> = dag.iter_edges().collect();
            while let Some((left, right)) = edges.pop() {
                dag.clear_edge(left, right);
            }
            let edges: Vec<(Vertex, Vertex)> = dag.iter_edges().collect();
            prop_assert!(edges.is_empty());
        }
    }

    /// Does not include the trivial divisors: k | k for every integer k.
    #[derive(Clone, Debug)]
    struct IntegerDivisibilityPoset {
        number: Vertex,
        divisors_of: BTreeMap<Vertex, Vec<Vertex>>,
    }

    impl IntegerDivisibilityPoset {
        fn get_divisors(number: Vertex) -> BTreeMap<Vertex, Vec<Vertex>> {
            let mut result: BTreeMap<Vertex, Vec<Vertex>> = Default::default();
            let mut numbers: Vec<Vertex> = vec![number];
            while let Some(n) = numbers.pop() {
                let divisors_of_n: Vec<Vertex> = (1..n / 2 + 1).filter(|d| n % d == 0).rev().collect();
                for divisor in &divisors_of_n {
                    if !result.contains_key(&divisor) {
                        numbers.push(*divisor);
                    }
                }
                result.insert(n, divisors_of_n);
            }
            result
        }

        fn of_number(number: Vertex) -> Self {
            IntegerDivisibilityPoset {
                number,
                divisors_of: Self::get_divisors(number),
            }
        }

        fn get_pairs(&self) -> Vec<(Vertex, Vertex)> {
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

    proptest! {
        #[test]
        fn prop_integer_divisibility_poset_isomorphism(size in 3u32..1000u32) {
            let integer_divisibility_poset = IntegerDivisibilityPoset::of_number(size.try_into().unwrap());

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
                prop_assert!(dag.get_edge(left, right));
            }

            for (left, right) in dag.iter_edges() {
                prop_assert_eq!(right % left, 0);
            }
        }
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
        assert_eq!(dag.iter_children(12).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_children(11).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_children(9).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_children(8).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_children(7).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_children(6).collect::<Vec<Vertex>>(), vec![12]);
        assert_eq!(dag.iter_children(5).collect::<Vec<Vertex>>(), vec![10]);
        assert_eq!(dag.iter_children(4).collect::<Vec<Vertex>>(), vec![8, 12]);
        assert_eq!(dag.iter_children(3).collect::<Vec<Vertex>>(), vec![6, 9, 12]);
        assert_eq!(
            dag.iter_children(2).collect::<Vec<Vertex>>(),
            vec![4, 6, 8, 10, 12]
        );
        assert_eq!(
            dag.iter_children(1).collect::<Vec<Vertex>>(),
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
        assert_eq!(descendants[12], RoaringBitmap::new());
        assert_eq!(descendants[11], RoaringBitmap::new());
        assert_eq!(descendants[10], RoaringBitmap::new());
        assert_eq!(descendants[9], RoaringBitmap::new());
        assert_eq!(descendants[8], RoaringBitmap::new());
        assert_eq!(descendants[7], RoaringBitmap::new());
        assert_eq!(descendants[6], RoaringBitmap::from_iter(vec![12]));
        assert_eq!(descendants[5], RoaringBitmap::from_iter(vec![10]));
        assert_eq!(descendants[4], RoaringBitmap::from_iter(vec![8, 12]));
        assert_eq!(descendants[3], RoaringBitmap::from_iter(vec![6, 9, 12]),);
        assert_eq!(
            descendants[2],
            RoaringBitmap::from_iter(vec![4, 6, 8, 10, 12]),
        );
        assert_eq!(
            descendants[1],
            RoaringBitmap::from_iter(vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        );
    }

    proptest! {
        #[test]
        fn prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
            dag in arb_dag(0..25),
        ) {
            println!("{:?}", dag);
            let transitive_closure: HashSet<(Vertex, Vertex)> =
                dag.transitive_closure().iter_edges().collect();
            let transitive_reduction: HashSet<(Vertex, Vertex)> =
                dag.transitive_reduction().iter_edges().collect();
            let intersection: HashSet<(Vertex, Vertex)> = transitive_closure
                .intersection(&transitive_reduction)
                .cloned()
                .collect();
            prop_assert_eq!(intersection, transitive_reduction);
        }
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

        assert_eq!(dag.iter_descendants_dfs(12).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_descendants_dfs(11).collect::<Vec<Vertex>>(), vec![]);
        assert_eq!(dag.iter_descendants_dfs(6).collect::<Vec<Vertex>>(), vec![12]);
    }

    proptest! {
        #[test]
        fn traversals_equal_modulo_order(dag in arb_dag(0..25)) {
            let bfs: HashSet<Vertex> = dag.iter_vertices_bfs().collect();
            let dfs: HashSet<Vertex> = dag.iter_vertices_dfs().collect();
            let dfs_post_order: HashSet<Vertex> = dag.iter_vertices_dfs_post_order().collect();
            prop_assert_eq!(&bfs, &dfs);
            prop_assert_eq!(&dfs_post_order, &dfs);
            prop_assert_eq!(&dfs_post_order, &bfs);
        }
    }

    /// A pseudo-test-case that fails on DAGs that contain a node with two ancestors
    fn fail_on_two_incoming(dag: impl Borrow<DirectedAcyclicGraph>) -> TestCaseResult {
        for to in 0..dag.borrow().get_vertex_count() {
            let mut count = 0;
            for from in 0..to {
                count += dag.borrow().get_edge(from, to) as u32;
                if count >= 2 {
                    return Err(TestCaseError::Fail(
                        "contains an edge with two incoming".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn minify_dag_to_3_nodes() {
        // This is the minimal DAG that has a node with two ancestors
        let minimal_dag = DirectedAcyclicGraph::from_edges_iter(3, vec![(0, 2), (1, 2)].into_iter());
        assert!(fail_on_two_incoming(&minimal_dag).is_err());

        // We construct a fully-connected DAG of 10 vertices
        let vertex_count = 10;
        let bitmap_size = strictly_upper_triangular_matrix_capacity(vertex_count);
        let mut edge_bitmap = RoaringBitmap::new();
        edge_bitmap.insert_range(0..bitmap_size);
        let mut vertex_mask = RoaringBitmap::new();
        vertex_mask.insert_range(0..vertex_count as u32);

        let full_graph_tree = DirectedAcyclicGraphValueTree {
            vertex_count,
            edge_bitmap_tree: DeltaDebuggingBitmapValueTree::new(edge_bitmap),
            vertex_mask_tree: DeltaDebuggingBitmapValueTree::new(vertex_mask),
            start_simplifying_edges: false,
        };

        let mut runner = TestRunner::new(Default::default());
        let result = runner.run_one(full_graph_tree, fail_on_two_incoming);

        // After running the shrinker, the DAG should be exactly the minimal DAG:
        assert_eq!(
            result,
            Err(TestError::Fail(
                "contains an edge with two incoming".into(),
                minimal_dag
            ))
        );
    }
}
