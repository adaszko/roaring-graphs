use roaring::{MultiOps, RoaringBitmap};

#[inline]
pub fn strictly_upper_triangular_matrix_capacity(n: u16) -> u32 {
    let n = u32::from(n);
    (n * n - n) / 2
}

pub struct RowColumnIterator {
    size: u16,
    i: u16,
    j: u16,
}

impl RowColumnIterator {
    pub fn new(size: u16) -> Self {
        Self {
            size,
            i: 0,
            j: 1,
        }
    }
}

impl<'a> Iterator for RowColumnIterator {
    type Item = (u16, u16);

    fn next(&mut self) -> Option<Self::Item> {
        if self.size == 0 {
            return None;
        }
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
    size: u16,
    matrix: RoaringBitmap,
}

impl Eq for StrictlyUpperTriangularLogicalMatrix {}

impl PartialEq for StrictlyUpperTriangularLogicalMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        self.iter_ones().eq(other.iter_ones())
    }
}

// Reference: https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/matrix-storage-schemes-for-lapack-routines.html
// Formulas adjusted for indexing from zero.
#[inline]
pub(crate) fn index_from_row_column(row: u16, column: u16, size: u16) -> u32 {
    u32::from(row) * u32::from(size) + u32::from(column)
}

#[inline]
pub(crate) fn row_from_index(index: u32, size: u16) -> u16 {
    u16::try_from(index / u32::from(size)).unwrap()
}

#[inline]
pub(crate) fn column_from_index(index: u32, size: u16) -> u16 {
    u16::try_from(index % u32::from(size)).unwrap()
}

#[inline]
pub(crate) fn row_column_from_index(index: u32, size: u16) -> (u16, u16) {
    let row = u16::try_from(index / u32::from(size)).unwrap();
    let column = u16::try_from(index % u32::from(size)).unwrap();
    (row, column)
}

impl StrictlyUpperTriangularLogicalMatrix {
    pub fn zeroed(size: u16) -> Self {
        Self {
            size,
            matrix: RoaringBitmap::new(),
        }
    }

    pub fn from_bitset(size: u16, bitset: RoaringBitmap) -> Self {
        Self {
            size,
            matrix: bitset,
        }
    }

    pub fn from_iter<I: Iterator<Item = (u16, u16)>>(size: u16, iter: I) -> Self {
        let mut bitmap = RoaringBitmap::new();
        for (i, j) in iter {
            let index = index_from_row_column(i, j, size);
            bitmap.insert(index);
        }
        Self::from_bitset(size, bitmap)
    }

    #[inline]
    pub fn size(&self) -> u16 {
        self.size
    }

    pub fn get(&self, i: u16, j: u16) -> bool {
        let index = index_from_row_column(i, j, self.size);
        self.matrix.contains(index)
    }

    /// Returns the previous value.
    pub fn set_to(&mut self, i: u16, j: u16, value: bool) -> bool {
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
    pub fn set(&mut self, i: u16, j: u16) {
        let index = index_from_row_column(i, j, self.size);
        self.matrix.insert(index);
    }

    pub fn clear(&mut self, i: u16, j: u16) {
        let index = index_from_row_column(i, j, self.size);
        self.matrix.remove(index);
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = (u16, u16)> + '_ {
        self.matrix.iter().map(|index| row_column_from_index(index, self.size))
    }

    pub fn iter_ones_at_row(&self, i: u16) -> impl Iterator<Item = u16> + '_ {
        assert!(i < self.size());
        ((i*self.size+i+1)..((i+1)*self.size)).into_iter().filter(|index| self.matrix.contains((*index).into())).map(|index| column_from_index(index.into(), self.size))
    }

    pub fn iter_ones_at_column(&self, j: u16) -> impl Iterator<Item = u16> + '_ {
        assert!(j < self.size());
        let mask = RoaringBitmap::from_iter((0..j).map(|k| u32::from(k * self.size + j)));
        let ones_indexes = [&self.matrix, &mask].intersection();
        ones_indexes.into_iter().map(|index| row_from_index(index, self.size))
    }
}

#[cfg(test)]
mod tests {
    use crate::strictly_upper_triangular_logical_matrix::*;

    #[test]
    fn positive_test_3x3_matrix() {
        let mut matrix = StrictlyUpperTriangularLogicalMatrix::zeroed(3);
        assert_eq!(matrix.get(0, 1), false);
        let ones: Vec<(u16, u16)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![]);

        matrix.set_to(0, 1, true);
        let ones: Vec<(u16, u16)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![(0, 1)]);
    }

    #[test]
    fn ones_at_row() {
        let mut matrix = StrictlyUpperTriangularLogicalMatrix::zeroed(3);
        matrix.set(0, 1);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_row(0)), vec![1]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_row(1)), vec![]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_row(2)), vec![]);
    }

    #[test]
    fn ones_at_column_bug1() {
        let mut matrix = StrictlyUpperTriangularLogicalMatrix::zeroed(3);
        matrix.set(0, 1);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(0)), vec![]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(1)), vec![0]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(2)), vec![]);
    }

    #[test]
    fn ones_at_column_bug2() {
        let matrix = StrictlyUpperTriangularLogicalMatrix::from_iter(5, vec![(1, 2)].into_iter());
        dbg!(&matrix);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(0)), vec![]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(1)), vec![]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(2)), vec![1]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(3)), vec![]);
        assert_eq!(Vec::from_iter(matrix.iter_ones_at_column(4)), vec![]);
    }
}
