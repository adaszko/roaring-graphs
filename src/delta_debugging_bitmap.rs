use proptest::{
    strategy::{Strategy, ValueTree},
    test_runner::Reason,
};
use rand::{distributions::Uniform, prelude::Distribution, Rng};
use roaring::RoaringBitmap;
use std::{
    cmp::{max, min},
    ops::Range,
};

#[derive(Debug, Clone)]
pub struct UniformBitProbabilityStrategy {
    size: Range<u32>,
    ones_probability: f64,
}

pub fn arb_bitmap_uniform(
    size: impl Into<Range<u32>>,
    ones_probability: f64,
) -> UniformBitProbabilityStrategy {
    UniformBitProbabilityStrategy {
        size: size.into(),
        ones_probability,
    }
}

impl Strategy for UniformBitProbabilityStrategy {
    type Tree = DeltaDebuggingBitmapValueTree;

    type Value = RoaringBitmap;

    fn new_tree(
        &self,
        runner: &mut proptest::test_runner::TestRunner,
    ) -> proptest::strategy::NewTree<Self> {
        // Copied out of self.size.assert_nonempty(), because that's private to proptest
        if self.size.is_empty() {
            panic!(
                "Invalid use of empty size range. (hint: did you \
                 accidentally write {}..{} where you meant {}..={} \
                 somewhere?)",
                self.size.start, self.size.end, self.size.start, self.size.end
            );
        }
        if self.ones_probability < 0.0 || self.ones_probability > 1.0 {
            panic!(
                "Invalid propability set for generating ones. \
                 Needs to be a number between 0 and 1, but got {}",
                self.ones_probability
            );
        }
        let size = Uniform::new(self.size.start, self.size.end - 1).sample(runner.rng());
        let iter = (0..size as u32).filter(|_| runner.rng().gen_bool(self.ones_probability));
        let bitmap =
            RoaringBitmap::from_sorted_iter(iter).map_err(|e| Reason::from(e.to_string()))?;
        Ok(DeltaDebuggingBitmapValueTree::new(bitmap))
    }
}

/// This struct holds the state needed for an implementation of the
/// [delta debugging] algorithm on bitmaps.
///
/// Together with proptest's shrinking code and the `ValueTree` implementation,
/// this shrinks bitmaps in O(log(n)) time to a single bit causing a failure,
/// or shrinks bitmaps in O(n²) time to exactly the bits causing the failure
/// (more precisely: a set of bits where removing a single one of them will make
/// the failure go away).
///
/// [delta debugging]: https://www.st.cs.uni-saarland.de/papers/tse2002/tse2002.pdf
#[derive(Debug)]
pub struct DeltaDebuggingBitmapValueTree {
    upper: RoaringBitmap,
    split_into: u64,
    split_index: u64,
    complement: bool,
}

impl DeltaDebuggingBitmapValueTree {
    pub fn new(bitmap: RoaringBitmap) -> Self {
        // We initialize `split_index` to 1, so .current() returns
        // `bitmap` initially.
        Self {
            upper: bitmap,
            split_into: 1,
            split_index: 0,
            complement: false,
        }
    }
}

impl ValueTree for DeltaDebuggingBitmapValueTree {
    type Value = RoaringBitmap;

    fn current(&self) -> Self::Value {
        let mut current = RoaringBitmap::new();
        let num_items = self.upper.len();
        let split_width = num_items / self.split_into;

        let mut idx = 0;
        for n in self.upper.iter() {
            let split_start = split_width * self.split_index;
            let split_end = split_start + split_width;
            let in_current_split = split_start <= idx && idx < split_end;

            if in_current_split ^ self.complement {
                current.insert(n);
            }

            idx += 1;
        }

        current
    }

    fn simplify(&mut self) -> bool {
        // This is called when the test case failed (as it should during shrinking!)

        let num_elems = self.upper.len();
        if num_elems == 0 || num_elems == 1 {
            return false;
        }

        self.upper = self.current();

        if self.complement {
            // Successfully reduced to subset complement
            self.split_into = max(self.split_into - 1, 2);
            self.complement = false;
        } else if self.split_into == 1 {
            // We add a special case to test the empty set as early as possible
            self.complement = true;
        } else {
            // Successfully reduced to subset
            self.split_into = 2;
            self.complement = false;
        }
        self.split_into = max(1, min(self.split_into, self.upper.len()));
        self.split_index = 0;
        true
    }

    fn complicate(&mut self) -> bool {
        // This is called when the test case didn't fail anymore.
        // So we try shrinking somewhere else.
        let num_elems = self.upper.len();
        if num_elems == 0 {
            return false;
        }

        self.split_index += 1;
        if self.split_index >= self.split_into {
            // we went through all split indices
            if self.complement {
                // We went through all split indices twice:
                // Once for all subsets, then for all of their complements.
                // We need to try to increase granularity:
                if self.split_into < num_elems {
                    // Try it "twice as granular".
                    // This is the binary search element that makes this algorithm efficient:
                    self.split_into = min(num_elems, 2 * self.split_into);
                    self.complement = false;
                    self.split_index = 0;
                    true
                } else {
                    // We tried the smallest granularity (individual elements).
                    // Nothing else we can do.
                    false
                }
            } else {
                // We went through all split indices once for all subsets.
                // Checking all of the subsets' complements remains.
                self.complement = true;
                self.split_index = 0;
                true
            }
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeltaDebuggingBitmapValueTree;
    use proptest::{
        collection::vec,
        prelude::*,
        strategy::ValueTree,
        test_runner::{TestError, TestRunner},
    };
    use roaring::RoaringBitmap;

    fn best_case_complexity(elems: u32) -> u32 {
        2 * ((elems as f64).log2() as u32)
    }

    fn worst_case_complexity(elems: u32) -> u32 {
        elems * elems + 3 * elems
    }

    fn runner_with_shrink_iters(max_shrink_iters: u32) -> TestRunner {
        TestRunner::new(ProptestConfig {
            max_shrink_iters,
            ..Default::default()
        })
    }

    #[test]
    fn shrinks_to_nothing_immediately() {
        // should shrink pretty much immedately
        let mut runner = runner_with_shrink_iters(2);

        let bitmap = RoaringBitmap::from_iter(0..100);
        let tree = DeltaDebuggingBitmapValueTree::new(bitmap);

        let result = runner.run_one(tree, |_| {
            // Always fail
            Err(TestCaseError::Fail("just because".into()))
        });

        assert_eq!(
            result,
            Err(TestError::Fail("just because".into(), RoaringBitmap::new()))
        )
    }

    #[test]
    fn initial_current_is_the_full_set() {
        let bitmap = RoaringBitmap::from_iter(0..100);
        let tree = DeltaDebuggingBitmapValueTree::new(bitmap.clone());
        assert_eq!(tree.current(), bitmap);
    }

    const SMALL_SIZE: u32 = 1000;
    const BIG_SIZE: u32 = 100_000;

    proptest! {
        #[test]
        fn should_shrink_to_minimal_elements(error_bits in vec(0..SMALL_SIZE, 1..10)) {
            let bitmap = RoaringBitmap::from_iter(0..SMALL_SIZE);
            prop_assert!(error_bits.iter().all(|e| bitmap.contains(*e)));
            let tree = DeltaDebuggingBitmapValueTree::new(bitmap);

            // multiplying with a factor of 2 due to some weird proptest behaviour.
            let mut runner = runner_with_shrink_iters(2 * worst_case_complexity(SMALL_SIZE));
            let result = runner.run_one(tree, |bitmap| {
                if error_bits.iter().all(|err_bit| bitmap.contains(*err_bit)) {
                    Err(TestCaseError::Fail("contains all error bits".into()))
                } else {
                    Ok(())
                }
            });

            let minimal_bitmap = RoaringBitmap::from_iter(error_bits.iter());
            prop_assert_eq!(
                result,
                Err(TestError::Fail(
                    "contains all error bits".into(),
                    minimal_bitmap
                ))
            );
        }

        #[test]
        fn test_best_case_complexity(error_bit in 0..BIG_SIZE) {
            let bitmap = RoaringBitmap::from_iter(0..BIG_SIZE);
            prop_assert!(bitmap.contains(error_bit));
            let tree = DeltaDebuggingBitmapValueTree::new(bitmap);

            // multiplying with a factor of 2 due to some weird proptest behaviour.
            let mut runner = runner_with_shrink_iters(2 * best_case_complexity(BIG_SIZE));
            let result = runner.run_one(tree, move |bitmap| {
                if bitmap.contains(error_bit) {
                    Err(TestCaseError::Fail("contains error bit".into()))
                } else {
                    Ok(())
                }
            });

            prop_assert_eq!(
                result,
                Err(TestError::Fail("contains error bit".into(), RoaringBitmap::from([error_bit])))
            );
        }
    }
}
