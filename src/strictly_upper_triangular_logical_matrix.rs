use fixedbitset::FixedBitSet;

const fn strictly_upper_triangular_matrix_capacity(n: usize) -> usize {
    (n * n - n) / 2
}

pub struct CacheFriendlyMatrixIterator {
    size: usize,
    i: usize,
    j: usize,
    index: usize,
}

impl<'a> Iterator for CacheFriendlyMatrixIterator {
    type Item = (usize, usize, usize); // (i, j, bitset index)

    fn next(&mut self) -> Option<Self::Item> {
        let result = (self.i, self.j, self.index);
        if self.j < self.size - 1 {
            self.j += 1;
            self.index += 1;
            return Some(result);
        }
        if self.i < self.size - 1 {
            self.i += 1;
            self.j = self.i + 1;
            self.index += 1;
            return Some(result);
        }
        None
    }
}

pub fn iter_matrix_starting_at(i: usize, size: usize) -> CacheFriendlyMatrixIterator {
    let j = i + 1;
    let index = unchecked_get_index_from_row_column(i, j, size);
    CacheFriendlyMatrixIterator { size, i, j, index }
}

/// Iterates over `(i, j)` pairs in an order that favors CPU cache locality.
/// If your graph algorithm can process edges in an arbitrary order, it is
/// recommended you use this iterator.
pub fn iter_matrix(size: usize) -> CacheFriendlyMatrixIterator {
    iter_matrix_starting_at(0, size)
}

/// A zero-indexed [row-major
/// packed](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
/// matrix of booleans.
#[derive(Clone, Debug)]
pub struct StrictlyUpperTriangularLogicalMatrix {
    size: usize,
    matrix: FixedBitSet,
}

// Reference: https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html
// Formulas adjusted for indexing from zero.
#[inline]
fn unchecked_get_index_from_row_column(i: usize, j: usize, size: usize) -> usize {
    ((2 * size - i - 1) * i) / 2 + j - i - 1
}

impl StrictlyUpperTriangularLogicalMatrix {
    pub fn zeroed(size: usize) -> Self {
        let capacity = strictly_upper_triangular_matrix_capacity(size);
        Self {
            size,
            matrix: FixedBitSet::with_capacity(capacity),
        }
    }

    pub fn from_iter<I: Iterator<Item = (usize, usize)>>(size: usize, iter: I) -> Self {
        let mut matrix = Self::zeroed(size);
        for (i, j) in iter {
            matrix.set(i, j, true);
        }
        matrix
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn index_from_row_column(&self, i: usize, j: usize) -> usize {
        assert!(i < self.size);
        assert!(j < self.size);
        assert!(i < j);
        unchecked_get_index_from_row_column(i, j, self.size)
    }

    pub fn get(&self, i: usize, j: usize) -> bool {
        let index = self.index_from_row_column(i, j);
        self.matrix[index]
    }

    /// Returns the previous value.
    pub fn set(&mut self, i: usize, j: usize, value: bool) -> bool {
        let index = self.index_from_row_column(i, j);
        let current = self.matrix[index];
        self.matrix.set(index, value);
        current
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        iter_matrix(self.size()).filter_map(move |(i, j, index)| {
            if self.matrix[index] {
                Some((i, j))
            } else {
                None
            }
        })
    }

    pub fn iter_ones_at_row(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        assert!(i < self.size());
        iter_matrix_starting_at(i, self.size())
            .take_while(move |(ii, _, _)| *ii == i)
            .filter(move |(_, _, index)| self.matrix[*index])
            .map(move |(_, jj, _)| jj)
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
        assert_eq!(unchecked_get_index_from_row_column(0, 1, 2), 0);

        // 3x3
        assert_eq!(unchecked_get_index_from_row_column(0, 1, 3), 0);
        assert_eq!(unchecked_get_index_from_row_column(0, 2, 3), 1);
        assert_eq!(unchecked_get_index_from_row_column(1, 2, 3), 2);

        // 4x4
        assert_eq!(unchecked_get_index_from_row_column(0, 1, 4), 0);
        assert_eq!(unchecked_get_index_from_row_column(0, 2, 4), 1);
        assert_eq!(unchecked_get_index_from_row_column(0, 3, 4), 2);
        assert_eq!(unchecked_get_index_from_row_column(1, 2, 4), 3);
        assert_eq!(unchecked_get_index_from_row_column(1, 3, 4), 4);
        assert_eq!(unchecked_get_index_from_row_column(2, 3, 4), 5);
    }

    #[test]
    fn test_matrix_iterator() {
        let row_column_index: Vec<(usize, usize, usize)> = iter_matrix(3).collect();
        assert_eq!(row_column_index, vec![(0, 1, 0), (0, 2, 1), (1, 2, 2),]);

        let row_column_index: Vec<(usize, usize, usize)> = iter_matrix(4).collect();
        assert_eq!(
            row_column_index,
            vec![
                (0, 1, 0),
                (0, 2, 1),
                (0, 3, 2),
                (1, 2, 3),
                (1, 3, 4),
                (2, 3, 5),
            ]
        );
    }
}
