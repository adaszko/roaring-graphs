use roaring::RoaringBitmap;

pub const fn strictly_upper_triangular_matrix_capacity(n: u32) -> u32 {
    (n * n - n) / 2
}

pub struct CacheFriendlyMatrixIterator {
    size: u32,
    i: u32,
    j: u32,
    index: u32,
}

impl<'a> Iterator for CacheFriendlyMatrixIterator {
    type Item = (u32, u32, u32); // (i, j, bitset index)

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

pub fn iter_matrix_starting_at(i: u32, size: u32) -> CacheFriendlyMatrixIterator {
    let j = i + 1;
    let index = unchecked_get_index_from_row_column(i, j, size);
    CacheFriendlyMatrixIterator { size, i, j, index }
}

/// Iterates over `(i, j)` pairs in an order that favors CPU cache locality.
/// If your graph algorithm can process edges in an arbitrary order, it is
/// recommended you use this iterator.
pub fn iter_matrix(size: u32) -> CacheFriendlyMatrixIterator {
    iter_matrix_starting_at(0, size)
}

/// A zero-indexed [row-major
/// packed](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html)
/// matrix of booleans.
#[derive(Clone, Debug)]
pub struct StrictlyUpperTriangularLogicalMatrix {
    size: u32,
    matrix: RoaringBitmap,
}

// Reference: https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html
// Formulas adjusted for indexing from zero.
#[inline]
fn unchecked_get_index_from_row_column(i: u32, j: u32, size: u32) -> u32 {
    ((2 * size - i - 1) * i) / 2 + j - i - 1
}

impl StrictlyUpperTriangularLogicalMatrix {
    pub fn zeroed(size: u32) -> Self {
        Self {
            size,
            matrix: RoaringBitmap::new(),
        }
    }

    pub fn from_bitset(size: u32, bitset: RoaringBitmap) -> Self {
        Self {
            size,
            matrix: bitset,
        }
    }

    pub fn from_iter<I: Iterator<Item = (u32, u32)>>(size: u32, iter: I) -> Self {
        let mut matrix = Self::zeroed(size);
        for (i, j) in iter {
            matrix.set(i, j, true);
        }
        matrix
    }

    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    #[inline]
    fn index_from_row_column(&self, i: u32, j: u32) -> u32 {
        assert!(i < self.size);
        assert!(j < self.size);
        assert!(i < j);
        unchecked_get_index_from_row_column(i, j, self.size)
    }

    pub fn get(&self, i: u32, j: u32) -> bool {
        let index = self.index_from_row_column(i, j);
        self.matrix.contains(index)
    }

    /// Returns the previous value.
    pub fn set(&mut self, i: u32, j: u32, value: bool) -> bool {
        let index = self.index_from_row_column(i, j);
        let current = self.matrix.contains(index);
        if value {
            self.matrix.insert(index);
        } else {
            self.matrix.remove(index);
        }
        current
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        iter_matrix(self.size()).filter_map(move |(i, j, index)| {
            if self.matrix.contains(index) {
                Some((i, j))
            } else {
                None
            }
        })
    }

    pub fn iter_ones_at_row(&self, i: u32) -> impl Iterator<Item = u32> + '_ {
        assert!(i < self.size());
        iter_matrix_starting_at(i, self.size())
            .take_while(move |(ii, _, _)| *ii == i)
            .filter(move |(_, _, index)| self.matrix.contains(*index))
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
        let ones: Vec<(u32, u32)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![]);

        matrix.set(0, 1, true);
        let ones: Vec<(u32, u32)> = matrix.iter_ones().collect();
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
        let row_column_index: Vec<(u32, u32, u32)> = iter_matrix(3).collect();
        assert_eq!(row_column_index, vec![(0, 1, 0), (0, 2, 1), (1, 2, 2),]);

        let row_column_index: Vec<(u32, u32, u32)> = iter_matrix(4).collect();
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
