use std::collections::VecDeque;

use fixedbitset::FixedBitSet;

use crate::DirectedAcyclicGraph;

/// Returns a set "seed" vertices of a DAG from which a traversal may start so
/// that the process covers all vertices in the graph.
pub fn get_vertices_without_incoming_edges(dag: &DirectedAcyclicGraph) -> Vec<usize> {
    let incoming_edges_count = {
        let mut incoming_edges_count: Vec<usize> = vec![0; dag.get_vertex_count()];
        for (_, v) in dag.iter_edges() {
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

pub fn iter_descendants_bfs(dag: &DirectedAcyclicGraph, vertex: usize) -> BfsVerticesIterator {
    BfsVerticesIterator {
        dag,
        visited: FixedBitSet::with_capacity(dag.get_vertex_count()),
        to_visit: vec![vertex].into(),
    }
}

pub fn iter_vertices_bfs(dag: &DirectedAcyclicGraph) -> BfsVerticesIterator {
    BfsVerticesIterator {
        dag,
        visited: FixedBitSet::with_capacity(dag.get_vertex_count()),
        to_visit: get_vertices_without_incoming_edges(dag).into(),
    }
}

/// See [`iter_vertices_dfs`].
pub struct DfsVerticesIterator<'a> {
    dag: &'a DirectedAcyclicGraph,
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
            self.to_visit.extend(self.dag.iter_children(u));
            self.visited.insert(u);
            return Some(u);
        }
        None
    }
}

pub fn iter_descendants_dfs(dag: &DirectedAcyclicGraph, vertex: usize) -> DfsVerticesIterator {
    DfsVerticesIterator {
        dag,
        visited: FixedBitSet::with_capacity(dag.get_vertex_count()),
        to_visit: vec![vertex],
    }
}

pub fn iter_vertices_dfs(dag: &DirectedAcyclicGraph) -> DfsVerticesIterator {
    DfsVerticesIterator {
        dag,
        visited: FixedBitSet::with_capacity(dag.get_vertex_count()),
        to_visit: get_vertices_without_incoming_edges(dag),
    }
}

/// See [`iter_vertices_dfs_post_order`].

pub struct DfsPostOrderVerticesIterator<'a> {
    dag: &'a DirectedAcyclicGraph,
    visited: FixedBitSet,
    to_visit: Vec<usize>,
}

impl<'a> Iterator for DfsPostOrderVerticesIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
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
                .dag
                .iter_children(u)
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

pub fn iter_descendants_dfs_post_order(
    dag: &DirectedAcyclicGraph,
    vertex: usize,
) -> DfsPostOrderVerticesIterator {
    DfsPostOrderVerticesIterator {
        dag,
        visited: FixedBitSet::with_capacity(dag.get_vertex_count()),
        to_visit: vec![vertex],
    }
}

pub fn iter_vertices_dfs_post_order(dag: &DirectedAcyclicGraph) -> DfsPostOrderVerticesIterator {
    DfsPostOrderVerticesIterator {
        dag,
        visited: FixedBitSet::with_capacity(dag.get_vertex_count()),
        to_visit: get_vertices_without_incoming_edges(dag),
    }
}

/// See [`iter_edges_dfs_post_order`].
pub struct DfsPostOrderEdgesIterator<'a> {
    dag: &'a DirectedAcyclicGraph,
    inner: DfsPostOrderVerticesIterator<'a>,
    seen_vertices: FixedBitSet,
    buffer: VecDeque<(usize, usize)>,
}

impl<'a> Iterator for DfsPostOrderEdgesIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((u, v)) = self.buffer.pop_front() {
                return Some((u, v));
            }

            let u = self.inner.next()?;

            for v in self.dag.iter_children(u) {
                if self.seen_vertices[v] {
                    self.buffer.push_back((u, v));
                }
            }
            self.seen_vertices.set(u, true);
        }
    }
}

/// Visit nodes in a depth-first-search (DFS) emitting edges in postorder, i.e.
/// each node after all its descendants have been emitted.
///
/// Note that when a DAG represents a [partially ordered
/// set](https://en.wikipedia.org/wiki/Partially_ordered_set), this function
/// iterates over pairs of that poset.  It may be necessary to first compute
/// either a [`crate::algorithm::transitive_reduction`] of a DAG, to only get
/// the minimal set of pairs spanning the entire poset, or a
/// [`crate::algorithm::transitive_closure`] to get all the pairs of that poset.
pub fn iter_edges_dfs_post_order(dag: &DirectedAcyclicGraph) -> DfsPostOrderEdgesIterator {
    DfsPostOrderEdgesIterator {
        dag,
        inner: iter_vertices_dfs_post_order(dag),
        seen_vertices: FixedBitSet::with_capacity(dag.get_vertex_count()),
        buffer: Default::default(),
    }
}

/// Combines [`iter_vertices_dfs_post_order`], [`Iterator::collect()`] and
/// [`slice::reverse()`] to get a topologically ordered sequence of vertices of a
/// DAG.
pub fn get_topologically_ordered_vertices(dag: &DirectedAcyclicGraph) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(dag.get_vertex_count());
    result.extend(iter_vertices_dfs_post_order(dag));
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

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
            iter_descendants_dfs(&dag, 12).collect::<Vec<usize>>(),
            vec![12]
        );
        assert_eq!(
            iter_descendants_dfs(&dag, 11).collect::<Vec<usize>>(),
            vec![11]
        );
        assert_eq!(
            iter_descendants_dfs(&dag, 6).collect::<Vec<usize>>(),
            vec![6, 12]
        );
    }

    fn prop_traversals_equal_modulo_order(dag: DirectedAcyclicGraph) {
        let bfs: HashSet<usize> = iter_vertices_bfs(&dag).collect();
        let dfs: HashSet<usize> = iter_vertices_dfs(&dag).collect();
        let dfs_post_order: HashSet<usize> = iter_vertices_dfs_post_order(&dag).collect();
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
