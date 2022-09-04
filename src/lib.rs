pub mod dag;
pub mod digraph;
pub mod strictly_upper_triangular_logical_matrix;

trait TraversableDirectedGraph {
    fn extend_with_children(&self, children: &mut Vec<usize>, u: usize);
    fn extend_with_parents(&self, parents: &mut Vec<usize>, v: usize);
}
