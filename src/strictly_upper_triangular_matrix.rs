use fixedbitset::{FixedBitSet, Ones};

#[derive(Clone)]
pub struct StrictlyUpperTriangularMatrix {
    size: usize,
    matrix: FixedBitSet,
}


pub struct EdgesIterator<'a> {
    size: usize,
    inner: Ones<'a>,
}

impl<'a> EdgesIterator<'a> {
    pub fn new(size: usize, bitset: &'a FixedBitSet) -> Self {
        let inner = bitset.ones();
        Self { size, inner }
    }
}

impl<'a> Iterator for EdgesIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(index) = self.inner.next() {
            return Some((index / self.size, index % self.size));
        }
        None
    }
}


pub struct NeighboursIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularMatrix,
    left_vertex: usize,
    right_vertex: usize,
}


impl<'a> Iterator for NeighboursIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.right_vertex < self.adjacency_matrix.size() {
            if self.adjacency_matrix.get(self.left_vertex, self.right_vertex) {
                let result = self.right_vertex;
                self.right_vertex += 1;
                return Some(result);
            }
            self.right_vertex += 1;
        }
        None
    }
}


impl StrictlyUpperTriangularMatrix {
    pub fn zeroed(size: usize) -> Self {
        // XXX: The optimal capacity is (size * size - size) / 2
        let capacity = size * size;
        Self {
            size,
            matrix: FixedBitSet::with_capacity(capacity),
        }
    }

    pub fn from_ones(size: usize, ones: &[(usize, usize)]) -> Self {
        let mut result = Self::zeroed(size);
        for (i, j) in ones {
            result.set(*i, *j, true);
        }
        result
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn index_from_row_column(&self, i: usize, j: usize) -> usize {
        let m = self.size();
        assert!(i < m, "assertion failed: i < m; i={}, m={}", i, m);
        assert!(j < m, "assertion failed: j < m; j={}, m={}", j, m);
        assert!(i < j, "assertion failed: i < j; i={}, j={}", i, j);
        m * i + j
    }

    pub fn get(&self, i: usize, j: usize) -> bool {
        let index = self.index_from_row_column(i, j);
        self.matrix[index]
    }

    pub fn set(&mut self, i: usize, j: usize, value: bool) -> bool {
        let index = self.index_from_row_column(i, j);
        let current = self.matrix[index];
        self.matrix.set(index, value);
        current
    }

    pub fn iter_ones(&self) -> EdgesIterator {
        EdgesIterator::new(self.size, &self.matrix)
    }

    pub fn iter_neighbours(&self, u: usize) -> NeighboursIterator {
        assert!(u < self.size());
        NeighboursIterator {
            adjacency_matrix: self,
            left_vertex: u,
            right_vertex: u + 1,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_test_3x3_matrix() {
        let mut matrix = StrictlyUpperTriangularMatrix::zeroed(3);
        assert_eq!(matrix.get(0, 1), false);
        let ones: Vec<(usize, usize)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![]);

        matrix.set(0, 1, true);
        let ones: Vec<(usize, usize)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![(0, 1)]);
    }
}
