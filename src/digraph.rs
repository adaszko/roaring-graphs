/// Unlike with [`DirectedAcyclicGraph`] data type, it is *not* the case that edges go from smaller
/// integers to bigger!
use std::{collections::VecDeque, io::Write};

use proptest::prelude::*;
use roaring::RoaringBitmap;

use crate::{dag::DirectedAcyclicGraph, TraversableDirectedGraph, Vertex};

pub type BitmapIndex = u32;

/// A mutable, single-threaded directed graph.
#[derive(Clone)]
pub struct DirectedGraph {
    vertex_count: Vertex,
    adjacency_matrix: RoaringBitmap,
}

impl std::fmt::Debug for DirectedGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones: Vec<(Vertex, Vertex)> = self.iter_edges().collect();
        write!(
            f,
            "DirectedGraph::from_edges_iter({}, vec!{:?}.iter().cloned())",
            self.get_vertex_count(),
            ones
        )?;
        Ok(())
    }
}

impl TraversableDirectedGraph for DirectedGraph {
    fn extend_with_children(&self, u: Vertex, children: &mut Vec<Vertex>) {
        self.extend_with_children(u, children)
    }

    fn extend_with_parents(&self, v: Vertex, parents: &mut Vec<Vertex>) {
        self.extend_with_parents(v, parents)
    }
}

#[inline]
fn index_from_row_column(i: Vertex, j: Vertex, size: Vertex) -> BitmapIndex {
    (i * size + j).into()
}

#[inline]
fn row_column_from_index(index: BitmapIndex, size: Vertex) -> (Vertex, Vertex) {
    let row = Vertex::try_from(index / BitmapIndex::from(size)).unwrap();
    let column = Vertex::try_from(index % BitmapIndex::from(size)).unwrap();
    (row, column)
}

impl DirectedGraph {
    /// Constructs a new graph without any edges having at most `vertex_count` vertices.
    pub fn new(vertex_count: Vertex) -> Self {
        Self {
            vertex_count,
            adjacency_matrix: RoaringBitmap::new(),
        }
    }

    pub fn has_edges(&self) -> bool {
        !self.adjacency_matrix.is_empty()
    }

    pub fn from_edges_iter<I>(vertex_count: Vertex, edges: I) -> Self
    where
        I: Iterator<Item = (Vertex, Vertex)>,
    {
        let mut adjacency_matrix = RoaringBitmap::new();
        for (from, to) in edges {
            let index = index_from_row_column(from, to, vertex_count);
            adjacency_matrix.insert(index);
        }
        Self {
            vertex_count,
            adjacency_matrix,
        }
    }

    pub fn from_dag(dag: &DirectedAcyclicGraph) -> Self {
        Self::from_edges_iter(
            dag.get_vertex_count(),
            dag.iter_edges()
                .map(|(u, v)| (u, v)),
        )
    }

    pub fn get_vertex_count(&self) -> Vertex {
        self.vertex_count
    }

    fn index_from_row_column(&self, i: Vertex, j: Vertex) -> BitmapIndex {
        assert!(i < self.vertex_count);
        assert!(j < self.vertex_count);
        index_from_row_column(i, j, self.vertex_count)
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = (Vertex, Vertex)> + '_ {
        self.adjacency_matrix.iter().map(|index| row_column_from_index(index, self.vertex_count))
    }

    pub fn get_edge(&self, parent: Vertex, child: Vertex) -> bool {
        assert_ne!(parent, child);
        assert!(parent < self.get_vertex_count());
        assert!(child < self.get_vertex_count());
        let index = self.index_from_row_column(parent, child);
        self.adjacency_matrix.contains(index)
    }

    pub fn set_edge(&mut self, parent: Vertex, child: Vertex) {
        assert_ne!(parent, child);
        assert!(parent < self.get_vertex_count());
        assert!(child < self.get_vertex_count());
        let index = self.index_from_row_column(parent, child);
        self.adjacency_matrix.insert(index);
    }

    pub fn clear_edge(&mut self, parent: Vertex, child: Vertex) {
        assert_ne!(parent, child);
        assert!(parent < self.get_vertex_count());
        assert!(child < self.get_vertex_count());
        let index = self.index_from_row_column(parent, child);
        self.adjacency_matrix.remove(index);
    }

    /// Returns `None` if the graph has more than connected component or there's no root.
    pub fn find_tree_root(&self) -> Option<Vertex> {
        let mut candidates = RoaringBitmap::new();
        candidates.insert_range(0..u32::from(self.vertex_count));
        for (_, to) in self.iter_edges() {
            candidates.remove(to.into());
        }
        if candidates.len() != 1 {
            return None;
        }
        let root = Vertex::try_from(candidates.select(0).unwrap()).unwrap();
        Some(root)
    }

    /// Pushes vertices `v` at the end of `children` such that there's an edge `(u, v)` in the graph.
    pub fn extend_with_children(&self, u: Vertex, children: &mut Vec<Vertex>) {
        assert!(u < self.vertex_count);
        let mut index = u * self.vertex_count;
        for v in 0..self.vertex_count {
            if self.adjacency_matrix.contains(index.into()) {
                children.push(v);
            }
            index += 1;
        }
    }

    /// Pushes vertices `u` at the end of `parents` such that there's an edge `(u, v)` in the graph.
    pub fn extend_with_parents(&self, v: Vertex, parents: &mut Vec<Vertex>) {
        assert!(v < self.vertex_count);
        let mut index = v;
        for u in 0..self.vertex_count {
            if self.adjacency_matrix.contains(index.into()) {
                parents.push(u);
            }
            index += self.vertex_count;
        }
    }

    /// Visit all vertices reachable from `vertex` in a depth-first-search (DFS)
    /// order.
    pub fn iter_descendants_dfs(&self, start_vertex: Vertex) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = DfsDescendantsIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: vec![start_vertex],
        };
        let iter = iter.filter(move |vertex| *vertex != start_vertex);
        Box::new(iter)
    }

    pub fn iter_ancestors_dfs(&self, start_vertex: Vertex) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = DfsAncestorsIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: vec![start_vertex],
        };
        let iter = iter.filter(move |vertex| *vertex != start_vertex);
        Box::new(iter)
    }

    /// Visit all vertices of a DAG in a depth-first-search postorder, i.e. emitting
    /// vertices only after all their descendants have been emitted first.
    pub fn iter_vertices_dfs_post_order(&self, start_vertices: &[Vertex]) -> Box<dyn Iterator<Item = Vertex> + '_> {
        let iter = DfsPostOrderVerticesIterator {
            digraph: self,
            visited: RoaringBitmap::new(),
            to_visit: start_vertices.to_vec(),
        };
        Box::new(iter)
    }

    /// Visit nodes in a depth-first-search (DFS) emitting edges in postorder, i.e.
    /// each node is emitted only after all its descendants have been emitted.
    pub fn iter_edges_dfs_post_order(&self, start_vertices: &[Vertex]) -> Box<dyn Iterator<Item = (Vertex, Vertex)> + '_> {
        let iter = DfsPostOrderEdgesIterator {
            digraph: self,
            inner: self.iter_vertices_dfs_post_order(start_vertices),
            seen_vertices: RoaringBitmap::new(),
            buffer: Default::default(),
        };
        Box::new(iter)
    }

    /// Returns "seed" vertices of a DAG from which a traversal may start so that the process
    /// covers all vertices in the graph.
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
            .map(|(vertex, _)| vertex.try_into().unwrap())
            .collect();

        vertices_without_incoming_edges
    }

    /// Return either an `Ok()` containing a sequence of topologically ordered vertices of a
    /// digraph, or an `Err()`, containing a cycle in the digraph, in which case the topological
    /// order does not exist.
    pub fn get_topologically_ordered_vertices(&self) -> Result<Vec<Vertex>, Vec<Vertex>> {
        let mut starting_vertices = self.get_vertices_without_incoming_edges();
        if starting_vertices.is_empty() && self.has_edges() {
            // If there are no vertices without incoming edges and yet there are some edges in the
            // graph, we have a highly cyclic graph.
            cov_mark::hit!(nonempty_graph_without_starting_vertices_graph_is_cyclic);

            // Pick any vertex as a starting point so that the traversal below returns *any* cycle
            // as an error.
            let (any_vertex, _) = self.iter_edges().next().unwrap();
            starting_vertices.push(any_vertex);
        }

        #[derive(Debug, Clone, Copy)]
        enum VisitStep {
            VertexChild(Vertex),
            OutOfVertexChildren(Vertex), // this marker is used as an indicator when to pop from the visitation stack
        }

        let mut result: Vec<Vertex> = Default::default();
        let mut visited = RoaringBitmap::new();
        while let Some(starting_vertex) = starting_vertices.pop() {
            let mut to_visit: Vec<VisitStep> = vec![VisitStep::VertexChild(starting_vertex)];
            let mut path: Vec<Vertex> = Default::default();
            while let Some(vertex) = to_visit.pop() {
                match vertex {
                    VisitStep::VertexChild(vertex) => {
                        if path.contains(&vertex) {
                            // We have a cycle
                            return Err(path);
                        }
                        if visited.contains(vertex.into()) {
                            // We have something homeomorphic to a diamond
                            continue;
                        }
                        path.push(vertex);
                        to_visit.push(VisitStep::OutOfVertexChildren(vertex));
                        let mut children: Vec<Vertex> = Default::default();
                        self.extend_with_children(vertex, &mut children);
                        for child in children {
                            to_visit.push(VisitStep::VertexChild(child));
                        }
                        visited.insert(vertex.into());
                    }
                    VisitStep::OutOfVertexChildren(vertex) => {
                        path.pop().unwrap();
                        result.push(vertex);
                    }
                };
            }
        }
        Ok(result)
    }

    /// Computes a mapping: vertex -> set of vertices that are descendants of vertex.
    fn get_descendants(&self) -> Vec<RoaringBitmap> {
        let mut descendants: Vec<RoaringBitmap> =
            vec![RoaringBitmap::default(); self.get_vertex_count().into()];

        let mut children = Vec::with_capacity(self.get_vertex_count().into());
        for u in (0..self.get_vertex_count()).rev() {
            children.clear();
            self.extend_with_children(u, &mut children);
            let mut u_descendants = RoaringBitmap::default();
            for &v in &children {
                u_descendants |= descendants[usize::try_from(v).unwrap()].clone();
                u_descendants.insert(v.into());
            }
            descendants[usize::try_from(u).unwrap()] = u_descendants;
        }

        descendants
    }

    /// Returns a new DAG that is a [transitive
    /// reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of a DAG.
    pub fn transitive_reduction(&self) -> DirectedGraph {
        let mut result = self.clone();

        let mut children = Vec::with_capacity(self.get_vertex_count().into());
        let descendants = self.get_descendants();
        for u in 0..self.get_vertex_count() {
            children.clear();
            self.extend_with_children(u, &mut children);
            for &v in &children {
                for w in descendants[usize::from(v)].iter() {
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

    /// Outputs the DAG in the [Graphviz DOT](https://graphviz.org/) format.
    pub fn to_dot<W: Write>(&self, output: &mut W) -> std::result::Result<(), std::io::Error> {
        writeln!(output, "digraph tree_{} {{", self.get_vertex_count())?;

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

pub fn arb_prufer_sequence(vertex_count: Vertex) -> BoxedStrategy<Vec<Vertex>> {
    assert!(vertex_count >= 2); // trees smaller than this have to be enumerated by hand
    proptest::collection::vec(0..vertex_count, usize::try_from(vertex_count - 2).unwrap()).boxed()
}

// https://www.geeksforgeeks.org/random-tree-generator-using-prufer-sequence-with-examples/
// https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence#Algorithm_to_convert_a_Pr%C3%BCfer_sequence_into_a_tree
pub fn random_tree_from_prufer_sequence(prufer_sequence: &[Vertex]) -> DirectedGraph {
    let nvertices = prufer_sequence.len() + 2;

    let mut degree: Vec<Vertex> = Vec::with_capacity(nvertices);
    degree.resize(nvertices, 1);

    let mut tree = DirectedGraph::new(nvertices.try_into().unwrap());

    // Number of occurrences of vertex in code
    for i in prufer_sequence {
        degree[usize::try_from(*i).unwrap()] += 1;
    }

    // Find the smallest label not present in prufer_sequence[]
    for i in prufer_sequence {
        for j in 0..nvertices {
            if degree[j] == 1 {
                tree.set_edge(*i, Vertex::try_from(j).unwrap());
                degree[usize::try_from(*i).unwrap()] -= 1;
                degree[j] -= 1;
                break;
            }
        }
    }

    let (u, v) = {
        let mut u: Option<Vertex> = None;
        let mut v: Option<Vertex> = None;
        for i in 0..nvertices {
            if degree[i] == 1 {
                if u == None {
                    u = Some(i.try_into().unwrap());
                } else {
                    v = Some(i.try_into().unwrap());
                    break;
                }
            }
        }
        (u.unwrap(), v.unwrap())
    };
    tree.set_edge(u, v);

    tree
}

pub fn arb_nonempty_tree(max_vertex_count: Vertex) -> BoxedStrategy<DirectedGraph> {
    (2..max_vertex_count)
        .prop_flat_map(|vertex_count| {
            arb_prufer_sequence(vertex_count).prop_flat_map(move |prufer_sequence| {
                let tree = random_tree_from_prufer_sequence(&prufer_sequence);
                Just(tree).boxed()
            })
        })
        .boxed()
}

pub fn arb_tree(max_vertex_count: Vertex) -> BoxedStrategy<DirectedGraph> {
    prop_oneof![
        1 => Just(DirectedGraph::new(max_vertex_count)).boxed(),
        99 => arb_nonempty_tree(max_vertex_count),
    ]
    .boxed()
}

/// See [`iter_vertices_dfs`].
pub(crate) struct DfsDescendantsIterator<'a, G: TraversableDirectedGraph> {
    pub(crate) digraph: &'a G,
    pub(crate) visited: RoaringBitmap,
    pub(crate) to_visit: Vec<Vertex>,
}

impl<'a, G: TraversableDirectedGraph> Iterator for DfsDescendantsIterator<'a, G> {
    type Item = Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.to_visit.pop() {
            if self.visited.contains(u.into()) {
                continue;
            }
            self.digraph.extend_with_children(u, &mut self.to_visit);
            self.visited.insert(u.into());
            return Some(u);
        }
        None
    }
}

pub(crate) struct DfsAncestorsIterator<'a, G: TraversableDirectedGraph> {
    pub(crate) digraph: &'a G,
    pub(crate) visited: RoaringBitmap,
    pub(crate) to_visit: Vec<Vertex>,
}

impl<'a, G: TraversableDirectedGraph> Iterator for DfsAncestorsIterator<'a, G> {
    type Item = Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.to_visit.pop() {
            if self.visited.contains(u.into()) {
                continue;
            }
            self.digraph.extend_with_parents(u, &mut self.to_visit);
            self.visited.insert(u.into());
            return Some(u);
        }
        None
    }
}

/// See [`iter_vertices_dfs_post_order`].
pub(crate) struct DfsPostOrderVerticesIterator<'a, G: TraversableDirectedGraph> {
    pub(crate) digraph: &'a G,
    pub(crate) visited: RoaringBitmap,
    pub(crate) to_visit: Vec<Vertex>,
}

impl<'a, G: TraversableDirectedGraph> Iterator for DfsPostOrderVerticesIterator<'a, G> {
    type Item = Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let u = match self.to_visit.last().copied() {
                Some(u) => u,
                None => return None,
            };
            if self.visited.contains(u.into()) {
                self.to_visit.pop();
                continue;
            }
            let unvisited_neighbours: Vec<Vertex> = {
                let mut neighbours: Vec<Vertex> = Default::default();
                self.digraph.extend_with_children(u, &mut neighbours);
                neighbours.retain(|v| !self.visited.contains((*v).into()));
                neighbours
            };
            if unvisited_neighbours.is_empty() {
                // We have visited all the descendants of u.  We can now emit u
                // from the iterator.
                self.to_visit.pop();
                self.visited.insert(u.into());
                return Some(u);
            }
            self.to_visit.extend(unvisited_neighbours);
        }
    }
}

/// See [`iter_edges_dfs_post_order`].
pub(crate) struct DfsPostOrderEdgesIterator<'a, G: TraversableDirectedGraph> {
    pub(crate) digraph: &'a G,
    pub(crate) inner: Box<dyn Iterator<Item = Vertex> + 'a>,
    pub(crate) seen_vertices: RoaringBitmap,
    pub(crate) buffer: VecDeque<(Vertex, Vertex)>,
}

impl<'a, G: TraversableDirectedGraph> Iterator for DfsPostOrderEdgesIterator<'a, G> {
    type Item = (Vertex, Vertex);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((u, v)) = self.buffer.pop_front() {
                return Some((u, v));
            }

            let u = self.inner.next()?;

            let mut children: Vec<Vertex> = Default::default();
            self.digraph.extend_with_children(u, &mut children);
            for v in children {
                if self.seen_vertices.contains(v.into()) {
                    self.buffer.push_back((u, v));
                }
            }
            self.seen_vertices.insert(u.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dag::arb_dag;

    use super::*;

    #[test]
    fn empty_graph_has_no_cycle() {
        let digraph = DirectedGraph::from_edges_iter(1, vec![].into_iter());
        assert!(digraph.get_topologically_ordered_vertices().is_ok());
    }

    #[test]
    fn diamond_has_no_cycle() {
        let diamond =
            DirectedGraph::from_edges_iter(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)].into_iter());
        assert!(diamond.get_topologically_ordered_vertices().is_ok());
    }

    #[test]
    fn simple_cyclic_digraph_has_cycle() {
        let digraph = DirectedGraph::from_edges_iter(2, vec![(0, 1), (1, 0)].into_iter());
        cov_mark::check!(nonempty_graph_without_starting_vertices_graph_is_cyclic);
        assert!(digraph.get_topologically_ordered_vertices().is_err());
    }

    #[test]
    fn triangle_has_cycle() {
        let digraph = DirectedGraph::from_edges_iter(3, vec![(0, 1), (1, 2), (2, 0)].into_iter());
        assert!(digraph.get_topologically_ordered_vertices().is_err());
    }

    proptest! {
        #[test]
        fn arb_tree_has_exactly_one_root(tree in arb_nonempty_tree(100)) {
            prop_assert!(tree.find_tree_root().is_some());
        }

        #[test]
        fn arb_tree_has_no_cycle(tree in arb_tree(100)) {
            prop_assert!(tree.get_topologically_ordered_vertices().is_ok());
        }

        #[test]
        fn arb_dag_has_no_cycle(dag in arb_dag(100)) {
            let digraph = DirectedGraph::from_dag(&dag);
            prop_assert!(digraph.get_topologically_ordered_vertices().is_ok());
        }
    }

    #[test]
    fn simple_topological_order() {
        // 0: 1, 2, 3, 4
        // 1: 3
        // 2: 3, 4
        // 3: 4
        let digraph = DirectedGraph::from_edges_iter(5, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (2, 3), (2, 4), (3, 4)].into_iter());
        assert_eq!(digraph.get_topologically_ordered_vertices().unwrap(), [4, 3, 2, 1, 0]);
    }
}
