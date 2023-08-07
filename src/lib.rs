pub mod dag;
pub mod digraph;
pub mod strictly_upper_triangular_logical_matrix;

pub type Vertex = u16;

trait TraversableDirectedGraph {
    fn extend_with_children(&self, u: Vertex, children: &mut Vec<Vertex>);
    fn extend_with_parents(&self, v: Vertex, parents: &mut Vec<Vertex>);
}

pub use dag::{arb_dag, DirectedAcyclicGraph};
pub use digraph::{arb_tree, arb_nonempty_tree, DirectedGraph};
