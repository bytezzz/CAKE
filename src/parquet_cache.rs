use crate::{configuration::get_parquet_cache_budget_bytes, ArrowQuiver};
use arrow_array::Array;
use schnellru::{Limiter, LruMap};
use std::{cell::RefCell, collections::hash_map::RandomState, path::Path};

thread_local! {
    static PARQUET_CACHE: RefCell<Option<ParquetCache>> = RefCell::new(None);
}

#[derive(Clone)]
struct CachedQuiver {
    quiver: ArrowQuiver,
    cost_bytes: usize,
}

#[derive(Clone, Copy)]
struct QuiverMemoryLimiter {
    max_bytes: usize,
    current_bytes: usize,
}

impl QuiverMemoryLimiter {
    fn new(max_bytes: usize) -> Self {
        QuiverMemoryLimiter {
            max_bytes,
            current_bytes: 0,
        }
    }
}

impl Limiter<String, CachedQuiver> for QuiverMemoryLimiter {
    type KeyToInsert<'a> = String;
    type LinkType = u32;

    fn is_over_the_limit(&self, _length: usize) -> bool {
        self.current_bytes > self.max_bytes
    }

    fn on_insert(
        &mut self,
        _length: usize,
        key: Self::KeyToInsert<'_>,
        value: CachedQuiver,
    ) -> Option<(String, CachedQuiver)> {
        if value.cost_bytes > self.max_bytes {
            return None;
        }
        self.current_bytes = self.current_bytes.saturating_add(value.cost_bytes);
        Some((key, value))
    }

    fn on_replace(
        &mut self,
        _length: usize,
        _old_key: &mut String,
        _new_key: String,
        old_value: &mut CachedQuiver,
        new_value: &mut CachedQuiver,
    ) -> bool {
        if new_value.cost_bytes > self.max_bytes {
            return false;
        }

        self.current_bytes = self
            .current_bytes
            .saturating_sub(old_value.cost_bytes)
            .saturating_add(new_value.cost_bytes);
        true
    }

    fn on_removed(&mut self, _key: &mut String, value: &mut CachedQuiver) {
        self.current_bytes = self.current_bytes.saturating_sub(value.cost_bytes);
    }

    fn on_cleared(&mut self) {
        self.current_bytes = 0;
    }

    fn on_grow(&mut self, _new_memory_usage: usize) -> bool {
        true
    }
}

struct ParquetCache {
    lru: LruMap<String, CachedQuiver, QuiverMemoryLimiter, RandomState>,
}

impl ParquetCache {
    fn new(budget_bytes: usize) -> Self {
        let limiter = QuiverMemoryLimiter::new(budget_bytes);
        let lru = LruMap::with_hasher(limiter, RandomState::new());
        ParquetCache { lru }
    }

    fn get(&mut self, key: &str) -> Option<ArrowQuiver> {
        self.lru.get(key).map(|entry| entry.quiver.clone())
    }

    fn insert(&mut self, key: String, quiver: ArrowQuiver, cost_bytes: usize) {
        let entry = CachedQuiver { quiver, cost_bytes };
        self.lru.insert(key, entry);
    }
}

fn with_parquet_cache_mut<R>(budget_bytes: usize, f: impl FnOnce(&mut ParquetCache) -> R) -> R {
    debug_assert!(budget_bytes > 0);
    PARQUET_CACHE.with(|cell| {
        let mut cache_opt = cell.borrow_mut();
        let cache = cache_opt.get_or_insert_with(|| ParquetCache::new(budget_bytes));
        f(cache)
    })
}

fn estimate_quiver_bytes(quiver: &ArrowQuiver) -> usize {
    quiver
        .columns()
        .iter()
        .map(|col| col.get_array_memory_size())
        .fold(0usize, |acc, bytes| acc.saturating_add(bytes))
}

pub fn load_parquet_cached(full_path: &Path) -> ArrowQuiver {
    let budget_bytes = get_parquet_cache_budget_bytes()
        .ok()
        .flatten()
        .filter(|b| *b > 0);
    let Some(budget_bytes) = budget_bytes else {
        return ArrowQuiver::from_parquet_file(full_path).unwrap();
    };

    let key = full_path.to_string_lossy().into_owned();

    if let Some(hit) = with_parquet_cache_mut(budget_bytes, |cache| cache.get(&key)) {
        return hit;
    }

    let loaded = ArrowQuiver::from_parquet_file(full_path).unwrap();
    let cost_bytes = estimate_quiver_bytes(&loaded);

    if cost_bytes <= budget_bytes {
        let stored = loaded.clone();
        with_parquet_cache_mut(budget_bytes, |cache| cache.insert(key, stored, cost_bytes));
    }

    loaded
}
