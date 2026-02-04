use crate::configuration::get_chunk_size;
use crate::ArrowQuiver;
use arrow_array::{
    ArrowPrimitiveType, BooleanArray, GenericStringArray, OffsetSizeTrait, PrimitiveArray,
};
use std::sync::Arc;

// Helper functions to get chunk sizes with proper error handling
pub fn get_chunk_size_query_plan() -> usize {
    get_chunk_size("query_plan").unwrap_or(2048)
}

pub fn get_chunk_size_filter() -> usize {
    get_chunk_size("filter").unwrap_or(4096)
}

pub fn get_chunk_size_sort() -> usize {
    get_chunk_size("sort").unwrap_or(1024)
}

#[macro_export]
macro_rules! execute_with_adaptive_chunking {
    (
        data: $data:expr,
        chunk_size: $chunk_size:expr,
        process_chunk: |$offset:ident, $chunk:ident| $process_body:expr,
        merge_results: |$results:ident| $merge_body:expr
    ) => {{
        use $crate::kernels::chunking::Chunkable;

        let data = $data;
        let chunk_size = $chunk_size;

        if data.len() < chunk_size * 2 {
            // Process as single chunk
            let $offset = 0;
            let $chunk = Chunkable::slice(&data, 0, data.len());
            $process_body
        } else {
            // Process with chunking
            let chunks = data.iter_chunks(chunk_size).collect::<Vec<_>>();

            let chunk_results: Vec<_> = chunks
                .into_iter()
                .map(|($offset, $chunk)| {
                    $process_body
                })
                .collect();

            // Merge results
            let $results = chunk_results;
            $merge_body
        }
    }};
    // Alternative form that allows different handling for single vs multiple chunks
    (
        data: $data:expr,
        chunk_size: $chunk_size:expr,
        process_chunk: |$offset:ident, $chunk:ident| $process_body:expr,
        single_chunk: |$offset_s:ident, $chunk_s:ident| $single_body:expr,
        merge_results: |$results:ident| $merge_body:expr
    ) => {{
        use $crate::kernels::chunking::Chunkable;

        let data = $data;
        let chunk_size = $chunk_size;

        if data.len() < chunk_size * 2 {
            // Process as single chunk
            let $offset_s = 0;
            let $chunk_s = Chunkable::slice(&data, 0, data.len());
            $single_body
        } else {
            // Process with chunking
            let chunks = data.iter_chunks(chunk_size).collect::<Vec<_>>();

            let chunk_results: Vec<_> = chunks
                .into_iter()
                .map(|($offset, $chunk)| {
                    $process_body
                })
                .collect();

            // Merge results
            let $results = chunk_results;
            $merge_body
        }
    }};
}

pub trait Chunkable {
    type Chunk;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn slice(&self, offset: usize, length: usize) -> Self::Chunk;

    fn iter_chunks(&'_ self, chunk_size: usize) -> ChunkIterator<'_, Self>
    where
        Self: Sized,
    {
        ChunkIterator {
            target: self,
            chunk_size,
            current_pos: 0,
        }
    }
}

pub struct ChunkIterator<'a, T: Chunkable + ?Sized> {
    target: &'a T,
    chunk_size: usize,
    current_pos: usize,
}

impl<T: Chunkable + ?Sized> Iterator for ChunkIterator<'_, T> {
    type Item = (usize, T::Chunk);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.target.len() {
            return None;
        }

        let offset = self.current_pos;
        let length = std::cmp::min(self.chunk_size, self.target.len() - offset);
        if length == 0 {
            return None;
        }
        let chunk = self.target.slice(offset, length);
        self.current_pos += length;

        Some((offset, chunk))
    }
}

impl<'b> Chunkable for &'b (dyn arrow_array::Array + 'b) {
    type Chunk = Arc<dyn arrow_array::Array>;

    fn len(&self) -> usize {
        arrow_array::Array::len(*self)
    }
    fn slice(&self, offset: usize, length: usize) -> Self::Chunk {
        arrow_array::Array::slice(*self, offset, length)
    }
}

impl<T: ArrowPrimitiveType> Chunkable for &PrimitiveArray<T> {
    type Chunk = Arc<PrimitiveArray<T>>;

    fn len(&self) -> usize {
        arrow_array::Array::len(*self)
    }
    fn slice(&self, offset: usize, length: usize) -> Self::Chunk {
        Arc::new((*self).slice(offset, length))
    }
}

impl Chunkable for &BooleanArray {
    type Chunk = Arc<BooleanArray>;

    fn len(&self) -> usize {
        arrow_array::Array::len(*self)
    }
    fn slice(&self, offset: usize, length: usize) -> Self::Chunk {
        Arc::new((*self).slice(offset, length))
    }
}

impl<T: OffsetSizeTrait> Chunkable for &GenericStringArray<T> {
    type Chunk = Arc<GenericStringArray<T>>;

    fn len(&self) -> usize {
        arrow_array::Array::len(*self)
    }
    fn slice(&self, offset: usize, length: usize) -> Self::Chunk {
        Arc::new((*self).slice(offset, length))
    }
}

impl Chunkable for &ArrowQuiver {
    type Chunk = ArrowQuiver;

    fn len(&self) -> usize {
        ArrowQuiver::num_rows(self)
    }

    fn slice(&self, offset: usize, length: usize) -> Self::Chunk {
        ArrowQuiver::slice(self, offset..offset + length)
    }
}

impl<C1: Chunkable, C2: Chunkable> Chunkable for (C1, C2) {
    type Chunk = (C1::Chunk, C2::Chunk);

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, length: usize) -> Self::Chunk {
        (self.0.slice(offset, length), self.1.slice(offset, length))
    }
}
