use std::collections::VecDeque;

use fixedbitset::FixedBitSet;

use crate::DirectedAcyclicGraph;

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

pub fn iter_reachable_vertices_starting_at(
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

/// When a DAG represents a [partially ordered
/// set](https://en.wikipedia.org/wiki/Partially_ordered_set), this method
/// iterates over all the pairs of that poset.
pub fn iter_edges_dfs_post_order(dag: &DirectedAcyclicGraph) -> DfsPostOrderEdgesIterator {
    DfsPostOrderEdgesIterator {
        dag,
        inner: iter_vertices_dfs_post_order(dag),
        seen_vertices: FixedBitSet::with_capacity(dag.get_vertex_count()),
        buffer: Default::default(),
    }
}

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

/// Simply calls `collect()` plus `reverse()` on the result of
/// [`iter_vertices_dfs_post_order()`].
pub fn get_topologically_ordered_vertices(dag: &DirectedAcyclicGraph) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(dag.get_vertex_count());
    result.extend(iter_vertices_dfs_post_order(dag));
    result.reverse();
    result
}
