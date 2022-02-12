use fixedbitset::FixedBitSet;

const fn strictly_upper_triangular_matrix_capacity(n: usize) -> usize {
    (n * n - n) / 2
}

/// A zero-indexed [row-major
/// packed](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
/// matrix of booleans.
#[derive(Clone)]
pub struct StrictlyUpperTriangularLogicalMatrix {
    size: usize,
    matrix: FixedBitSet,
}

pub struct RowColumnPairsIterator {
    size: usize,
    i: usize,
    j: usize,
}

impl<'a> Iterator for RowColumnPairsIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.j < self.size {
            let result = (self.i, self.j);
            self.j += 1;
            return Some(result);
        }
        if self.i < self.size {
            let result = (self.i, self.j);
            self.i += 1;
            self.j = self.i + 1;
            return Some(result);
        }
        None
    }
}

// Reference: https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html
// Formulas adjusted for indexing from zero.
#[inline]
fn get_index_from_row_column(i: usize, j: usize, size: usize) -> usize {
    debug_assert!(i < size, "assertion failed: i < m; i={}, m={}", i, size);
    debug_assert!(j < size, "assertion failed: j < m; j={}, m={}", j, size);
    debug_assert!(i < j, "assertion failed: i < j; i={}, j={}", i, j);
    ((2 * size - i - 1) * i) / 2 + j - i - 1
}

pub struct EdgesIterator<'a> {
    size: usize,
    bitset: &'a FixedBitSet,
    i: usize,
    j: usize,
}

impl<'a> EdgesIterator<'a> {
    pub fn new(size: usize, bitset: &'a FixedBitSet) -> Self {
        Self {
            size,
            bitset,
            i: 0,
            j: 1,
        }
    }
}

impl<'a> Iterator for EdgesIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.size {
            while self.j < self.size {
                let index = get_index_from_row_column(self.i, self.j, self.size);
                let current_j = self.j;
                self.j += 1;
                if self.bitset[index] {
                    return Some((self.i, current_j));
                }
            }
            self.i += 1;
        }
        None
    }
}

pub struct NeighboursIterator<'a> {
    adjacency_matrix: &'a StrictlyUpperTriangularLogicalMatrix,
    left_vertex: usize,
    right_vertex: usize,
}

impl<'a> Iterator for NeighboursIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.right_vertex < self.adjacency_matrix.size() {
            if self
                .adjacency_matrix
                .get(self.left_vertex, self.right_vertex)
            {
                let result = self.right_vertex;
                self.right_vertex += 1;
                return Some(result);
            }
            self.right_vertex += 1;
        }
        None
    }
}

impl StrictlyUpperTriangularLogicalMatrix {
    pub fn zeroed(size: usize) -> Self {
        let capacity = strictly_upper_triangular_matrix_capacity(size);
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

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn index_from_row_column(&self, i: usize, j: usize) -> usize {
        get_index_from_row_column(i, j, self.size())
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

    /// Iterates over `(i, j)` pairs in an order that favors CPU cache locality.
    /// If your graph algorithm can process edges in an arbitrary order, it is
    /// recommended you use this iterator.
    pub fn iter_row_column_pairs(&self) -> RowColumnPairsIterator {
        RowColumnPairsIterator {
            size: self.size,
            i: 0,
            j: 1,
        }
    }

    pub fn iter_ones(&self) -> EdgesIterator {
        EdgesIterator::new(self.size, &self.matrix)
    }

    pub fn iter_neighbours(&self, u: usize) -> NeighboursIterator {
        debug_assert!(u < self.size());
        NeighboursIterator {
            adjacency_matrix: self,
            left_vertex: u,
            right_vertex: u + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::strictly_upper_triangular_logical_matrix::*;

    #[test]
    fn positive_test_3x3_matrix() {
        let mut matrix = StrictlyUpperTriangularLogicalMatrix::zeroed(3);
        assert_eq!(matrix.get(0, 1), false);
        let ones: Vec<(usize, usize)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![]);

        matrix.set(0, 1, true);
        let ones: Vec<(usize, usize)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![(0, 1)]);
    }

    #[test]
    fn index_computation_is_sane() {
        // 2x2
        assert_eq!(get_index_from_row_column(0, 1, 2), 0);

        // 3x3
        assert_eq!(get_index_from_row_column(0, 1, 3), 0);
        assert_eq!(get_index_from_row_column(0, 2, 3), 1);
        assert_eq!(get_index_from_row_column(1, 2, 3), 2);

        // 4x4
        assert_eq!(get_index_from_row_column(0, 1, 4), 0);
        assert_eq!(get_index_from_row_column(0, 2, 4), 1);
        assert_eq!(get_index_from_row_column(0, 3, 4), 2);
        assert_eq!(get_index_from_row_column(1, 2, 4), 3);
        assert_eq!(get_index_from_row_column(1, 3, 4), 4);
        assert_eq!(get_index_from_row_column(2, 3, 4), 5);
    }
}
