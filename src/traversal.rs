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
            self.to_visit
                .extend(self.dag.iter_neighbours(u).filter(|v| !self.visited[*v]));
            self.visited.insert(u);
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
            self.to_visit.extend(self.dag.iter_neighbours(u));
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
                .dag
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

            for v in self.dag.iter_neighbours(u) {
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
