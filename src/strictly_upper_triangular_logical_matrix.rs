use roaring::{RoaringBitmap, MultiOps};

pub const fn strictly_upper_triangular_matrix_capacity(n: u32) -> u32 {
    (n * n - n) / 2
}

pub struct StrictlyUpperTriangularMatrixRowColumnIterator {
    size: u32,
    i: u32,
    j: u32,
}

impl StrictlyUpperTriangularMatrixRowColumnIterator {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            i: 0,
            j: 1,
        }
    }
}

impl<'a> Iterator for StrictlyUpperTriangularMatrixRowColumnIterator {
    type Item = (u32, u32); // (i, j, bitset index)

    fn next(&mut self) -> Option<Self::Item> {
        let result = (self.i, self.j);
        if self.j < self.size - 1 {
            self.j += 1;
            return Some(result);
        }
        if self.i < self.size - 1 {
            self.i += 1;
            self.j = self.i + 1;
            return Some(result);
        }
        None
    }
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
fn unchecked_index_from_row_column(row: u32, column: u32, size: u32) -> u32 {
    row * size + column
}

fn row_column_from_index(index: u32, size: u32) -> (u32, u32) {
    let row = index / size;
    let column = index % size;
    (row, column)
}

#[inline]
fn index_from_row_column(i: u32, j: u32, size: u32) -> u32 {
    assert!(i < size);
    assert!(j < size);
    assert!(i < j);
    unchecked_index_from_row_column(i, j, size)
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
        let mut bitmap = RoaringBitmap::new();
        for (i, j) in iter {
            let index = index_from_row_column(i, j, size);
            bitmap.insert(index);
        }
        Self::from_bitset(size, bitmap)
    }

    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn get(&self, i: u32, j: u32) -> bool {
        let index = index_from_row_column(i, j, self.size);
        self.matrix.contains(index)
    }

    /// Returns the previous value.
    pub fn set_to(&mut self, i: u32, j: u32, value: bool) -> bool {
        let index = index_from_row_column(i, j, self.size);
        let current = self.matrix.contains(index);
        if value {
            self.matrix.insert(index);
        } else {
            self.matrix.remove(index);
        }
        current
    }

    /// Returns the previous value.
    pub fn set(&mut self, i: u32, j: u32) {
        let index = index_from_row_column(i, j, self.size);
        self.matrix.insert(index);
    }

    pub fn clear(&mut self, i: u32, j: u32) {
        let index = index_from_row_column(i, j, self.size);
        self.matrix.remove(index);
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.matrix.iter().map(|index| row_column_from_index(index, self.size))
    }

    pub fn iter_ones_at_row(&self, i: u32) -> impl Iterator<Item = u32> + '_ {
        assert!(i < self.size());
        let mask = RoaringBitmap::from_iter((i * self.size)..((i + 1) * self.size));
        let result = [&self.matrix, &mask].intersection();
        result.into_iter().map(|index| row_column_from_index(index, self.size).1)
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

        matrix.set_to(0, 1, true);
        let ones: Vec<(u32, u32)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![(0, 1)]);
    }
}
