pub mod dag;
pub mod digraph;
pub mod strictly_upper_triangular_logical_matrix;

pub type Vertex = u16;

trait TraversableDirectedGraph {
    fn extend_with_children(&self, children: &mut Vec<Vertex>, u: Vertex);
    fn extend_with_parents(&self, parents: &mut Vec<Vertex>, v: Vertex);
}

pub use dag::{arb_dag, DirectedAcyclicGraph};
pub use digraph::{arb_tree, DirectedGraph};
