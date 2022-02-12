use crate::{traversal::iter_reachable_vertices_starting_at, DirectedAcyclicGraph};

/// Returns a new DAG that is a [transitive
/// reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of a DAG.
pub fn transitive_reduction(dag: &DirectedAcyclicGraph) -> DirectedAcyclicGraph {
    let mut result = dag.clone();

    for u in 0..dag.get_vertex_count() {
        for v in dag.iter_neighbours(u) {
            for w in iter_reachable_vertices_starting_at(dag, v) {
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
            for w in iter_reachable_vertices_starting_at(dag, v) {
                if w == v {
                    continue;
                }
                result.set_edge(u, w, true);
            }
        }
    }
    result
}
