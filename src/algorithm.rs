use fixedbitset::FixedBitSet;

use crate::DirectedAcyclicGraph;

/// Computes a mapping: vertex -> set of vertices that are descendants of vertex.
pub fn get_descendants(dag: &DirectedAcyclicGraph) -> Vec<FixedBitSet> {
    let mut descendants: Vec<FixedBitSet> = vec![FixedBitSet::default(); dag.get_vertex_count()];

    for u in (0..dag.get_vertex_count()).rev() {
        let mut u_descendants = FixedBitSet::default();
        for v in dag.iter_children(u) {
            u_descendants.union_with(&descendants[v]);
            u_descendants.grow(v + 1);
            u_descendants.set(v, true);
        }
        descendants[u] = u_descendants;
    }

    descendants
}

/// Returns a new DAG that is a [transitive
/// reduction](https://en.wikipedia.org/wiki/Transitive_reduction) of a DAG.
pub fn transitive_reduction(dag: &DirectedAcyclicGraph) -> DirectedAcyclicGraph {
    let mut result = dag.clone();

    let descendants = get_descendants(dag);
    for u in 0..dag.get_vertex_count() {
        for v in dag.iter_children(u) {
            for w in descendants[v].ones() {
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

    // http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci335/lecture_notes/chapter08.pdf

    let descendants = get_descendants(dag);
    for u in 0..dag.get_vertex_count() {
        for v in descendants[u].ones() {
            result.set_edge(u, v, true);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
        let descendants = get_descendants(&dag);
        assert_eq!(descendants[12], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[11], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[10], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[9], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[8], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[7], FixedBitSet::from_iter(vec![]));
        assert_eq!(descendants[6].ones().collect::<Vec<usize>>(), vec![12]);
        assert_eq!(descendants[5].ones().collect::<Vec<usize>>(), vec![10]);
        assert_eq!(descendants[4].ones().collect::<Vec<usize>>(), vec![8, 12]);
        assert_eq!(
            descendants[3].ones().collect::<Vec<usize>>(),
            vec![6, 9, 12]
        );
        assert_eq!(
            descendants[2].ones().collect::<Vec<usize>>(),
            vec![4, 6, 8, 10, 12]
        );
        assert_eq!(
            descendants[1].ones().collect::<Vec<usize>>(),
            vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    fn prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
        dag: DirectedAcyclicGraph,
    ) -> bool {
        println!("{:?}", dag);
        let transitive_closure: HashSet<(usize, usize)> =
            transitive_closure(&dag).iter_edges().collect();
        let transitive_reduction: HashSet<(usize, usize)> =
            transitive_reduction(&dag).iter_edges().collect();
        let intersection: HashSet<(usize, usize)> = transitive_closure
            .intersection(&transitive_reduction)
            .cloned()
            .collect();
        intersection == transitive_reduction
    }

    #[test]
    fn transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order(
    ) {
        quickcheck::QuickCheck::new().quickcheck(prop_transitive_closure_and_transitive_reduction_intersection_equals_transitive_reduction_modulo_order as fn(DirectedAcyclicGraph) -> bool);
    }
}
