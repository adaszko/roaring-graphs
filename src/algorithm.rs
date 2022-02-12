use crate::{traversal::iter_descendants_dfs, DirectedAcyclicGraph};

/// Returns a new DAG that is a [transitive
/// reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of a DAG.
pub fn transitive_reduction(dag: &DirectedAcyclicGraph) -> DirectedAcyclicGraph {
    let mut result = dag.clone();
    for u in 0..dag.get_vertex_count() {
        for v in dag.iter_children(u) {
            for w in iter_descendants_dfs(dag, v) {
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
/// closure](https://en.wikipedia.org/wiki/Transitive_closure) of a DAG.  Note
/// that this is equivalent to computing a reachability matrix.
pub fn transitive_closure(dag: &DirectedAcyclicGraph) -> DirectedAcyclicGraph {
    let mut result = dag.clone();

    for u in 0..dag.get_vertex_count() {
        for v in dag.iter_children(u) {
            for w in iter_descendants_dfs(dag, v) {
                if w == v {
                    continue;
                }
                result.set_edge(u, w, true);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
        dag: DirectedAcyclicGraph,
    ) -> bool {
        let transitive_closure: HashSet<(usize, usize)> =
            transitive_closure(&dag).iter_edges().collect();
        let transitive_reduction: HashSet<(usize, usize)> =
            transitive_reduction(&dag).iter_edges().collect();
        transitive_closure == transitive_reduction
    }

    #[test]
    fn transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
    ) {
        quickcheck::QuickCheck::new().quickcheck(prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order as fn(DirectedAcyclicGraph) -> bool);
    }
}
