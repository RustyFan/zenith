use std::hash::Hasher;
use foldhash::fast::FoldHasher;
use foldhash::SharedSeed;

pub type SmallVec<A> = smallvec::SmallVec<A>;

pub struct DefaultHasher(FoldHasher);

pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
pub type HashSet<T> = hashbrown::HashSet<T>;

pub mod hashmap {
    pub use hashbrown::hash_map::*;
}

pub mod hashset {
    pub use hashbrown::hash_set::*;
}

impl DefaultHasher {
    pub fn new() -> Self {
        Self(FoldHasher::with_seed(0, SharedSeed::global_random()))
    }
}

impl Hasher for DefaultHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}
