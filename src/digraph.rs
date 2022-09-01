use std::io::Write;

use proptest::prelude::*;
use fixedbitset::FixedBitSet;

#[derive(Clone, Debug)]
pub struct DirectedGraph {
    vertex_count: usize,
    adjacency_matrix: FixedBitSet,
}

impl DirectedGraph {
    pub fn empty(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            adjacency_matrix: FixedBitSet::with_capacity(vertex_count * vertex_count),
        }
    }

    pub fn get_vertex_count(&self) -> usize {
        self.vertex_count
    }

    fn index_from_row_column(&self, i: usize, j: usize) -> usize {
        assert!(i < self.vertex_count);
        assert!(j < self.vertex_count);
        i * self.vertex_count + j
    }

    /// Iterates over the edges in an order that favors CPU cache locality.
    pub fn iter_edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.adjacency_matrix.ones().map(|index| {
            let row = index / self.vertex_count;
            let column = index % self.vertex_count;
            (row, column)
        })
    }

    pub fn set_edge(&mut self, parent: usize, child: usize, exists: bool) {
        assert_ne!(parent, child);
        assert!(parent < self.get_vertex_count());
        assert!(child < self.get_vertex_count());
        let index = self.index_from_row_column(parent, child);
        self.adjacency_matrix.set(index, exists);
    }

    // Returns None if the graph has more than connected component or there's no root.
    pub fn find_tree_root(&self) -> Option<usize> {
        let mut candidates = FixedBitSet::with_capacity(self.vertex_count);
        candidates.set_range(.., true);
        for (_, to) in self.iter_edges() {
            candidates.set(to, false);
        }
        let roots: Vec<usize> = candidates.ones().collect();
        if roots.len() != 1 {
            return None;
        }
        Some(roots[0])
    }

    /// Iterates over vertices `v` such that there's an edge `(u, v)` in the graph.
    pub fn extend_with_children(&self, children: &mut Vec<usize>, u: usize) {
        assert!(u < self.vertex_count);
        for v in 0..self.vertex_count {
            if self.adjacency_matrix[u * self.vertex_count + v] {
                children.push(v);
            }
        }
    }

    /// Iterates over vertices `u` such that there's an edge `(u, v)` in the graph.
    pub fn extend_with_parents(&self, parents: &mut Vec<usize>, v: usize) {
        assert!(v < self.vertex_count);
        for u in 0..self.vertex_count {
            if self.adjacency_matrix[u * self.vertex_count + v] {
                parents.push(u);
            }
        }
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

pub fn arb_prufer_sequence(vertex_count: usize) -> BoxedStrategy<Vec<usize>> {
    assert!(vertex_count >= 2); // trees smaller than this have to be enumerated by hand
    proptest::collection::vec(0..vertex_count, vertex_count - 2).boxed()
}

// https://www.geeksforgeeks.org/random-tree-generator-using-prufer-sequence-with-examples/
// https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence#Algorithm_to_convert_a_Pr%C3%BCfer_sequence_into_a_tree
pub fn random_tree_from_prufer_sequence(prufer_sequence: &[usize]) -> DirectedGraph {
    let nvertices = prufer_sequence.len() + 2;

    let mut degree: Vec<usize> = Vec::with_capacity(nvertices);
    degree.resize(nvertices, 1);

    let mut tree = DirectedGraph::empty(nvertices);

    // Number of occurrences of vertex in code
    for i in prufer_sequence {
        degree[*i] += 1;
    }

    // Find the smallest label not present in prufer_sequence[]
    for i in prufer_sequence {
        for j in 0..nvertices {
            if degree[j] == 1 {
                tree.set_edge(*i, j, true);
                degree[*i] -= 1;
                degree[j] -= 1;
                break;
            }
        }
    }

    let (u, v) = {
        let mut u: Option<usize> = None;
        let mut v: Option<usize> = None;
        for i in 0..nvertices {
            if degree[i] == 1 {
                if u == None {
                    u = Some(i);
                } else {
                    v = Some(i);
                    break;
                }
            }
        }
        (u.unwrap(), v.unwrap())
    };
    tree.set_edge(u, v, true);

    tree
}

pub fn arb_tree(max_vertex_count: usize) -> BoxedStrategy<DirectedGraph> {
    (2..max_vertex_count)
        .prop_flat_map(|vertex_count| {
            arb_prufer_sequence(vertex_count).prop_flat_map(move |prufer_sequence| {
                let tree = random_tree_from_prufer_sequence(&prufer_sequence);
                Just(tree).boxed()
            })
        })
        .boxed()
}


/// See [`iter_vertices_dfs`].
pub struct DfsVerticesIterator<'a> {
    digraph: &'a DirectedGraph,
    visited: FixedBitSet,
    to_visit: Vec<usize>,
}

impl<'a> Iterator for DfsVerticesIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.to_visit.pop() {
            if self.visited[u] {
                continue;
            }
            self.digraph.extend_with_children(&mut self.to_visit, u);
            self.visited.insert(u);
            return Some(u);
        }
        None
    }
}

/// Visit all vertices reachable from `vertex` in a depth-first-search (DFS)
/// order.
pub fn iter_descendants_dfs(digraph: &DirectedGraph, vertex: usize) -> DfsVerticesIterator {
    DfsVerticesIterator {
        digraph,
        visited: FixedBitSet::with_capacity(digraph.get_vertex_count()),
        to_visit: vec![vertex],
    }
}

pub struct DfsAncestorsIterator<'a> {
    digraph: &'a DirectedGraph,
    visited: FixedBitSet,
    to_visit: Vec<usize>,
}

impl<'a> Iterator for DfsAncestorsIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u) = self.to_visit.pop() {
            if self.visited[u] {
                continue;
            }
            self.digraph.extend_with_parents(&mut self.to_visit, u);
            self.visited.insert(u);
            return Some(u);
        }
        None
    }
}


pub fn iter_ancestors_dfs(digraph: &DirectedGraph, vertex: usize) -> DfsAncestorsIterator {
    DfsAncestorsIterator {
        digraph,
        visited: FixedBitSet::with_capacity(digraph.get_vertex_count()),
        to_visit: vec![vertex],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn arb_tree_has_exactly_one_root(tree in arb_tree(100)) {
            prop_assert!(tree.find_tree_root().is_some());
        }
    }
}
