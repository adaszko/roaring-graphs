pub mod dag;
pub mod digraph;
pub mod strictly_upper_triangular_logical_matrix;

trait TraversableDirectedGraph {
    fn extend_with_children(&self, children: &mut Vec<u32>, u: u32);
    fn extend_with_parents(&self, parents: &mut Vec<u32>, v: u32);
}

pub use dag::{arb_dag, DirectedAcyclicGraph};
pub use digraph::{arb_tree, DirectedGraph};
