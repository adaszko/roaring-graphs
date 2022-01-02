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
        assert!(i < m);
        assert!(j < m);
        assert!(i < j);
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
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic = "assertion failed: i < j"]
    fn positive_test_2x2_matrix() {
        let mut matrix = StrictlyUpperTriangularMatrix::zeroed(2);
        assert_eq!(matrix.get(0, 0), false);
        let ones: Vec<(usize, usize)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![]);
        matrix.set(0, 0, true);
        let ones: Vec<(usize, usize)> = matrix.iter_ones().collect();
        assert_eq!(ones, vec![(0, 0)]);
    }

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
