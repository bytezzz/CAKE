use std::{
    error::Error,
    fs::File,
    io,
    path::Path,
    sync::{
        mpsc::{self, Receiver},
        Arc,
    },
    thread,
};

use arrow_array::{Array, UInt64Array};
use arrow_schema::{ArrowError, Schema};
use parquet::{
    arrow::arrow_reader::{
        statistics::StatisticsConverter, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder,
    },
    file::{metadata::ParquetMetaData, reader::ChunkReader},
};

use crate::{AQExecutorContext, ArrowQuiver, ColumnIdentifier, Operator, Source};

pub struct AQParquetMetaData<T: ChunkReader> {
    pub schema: Arc<Schema>,
    pub metadata: Arc<ParquetMetaData>,
    builder: ParquetRecordBatchReaderBuilder<T>,
}

pub fn read_parquet_meta(fp: &Path) -> Result<AQParquetMetaData<File>, io::Error> {
    let f = File::open(fp)?;
    let b = ParquetRecordBatchReaderBuilder::try_new(f)?;
    let schema = b.schema().clone();
    let metadata = b.metadata().clone();

    Ok(AQParquetMetaData {
        schema,
        metadata,
        builder: b,
    })
}

impl<T: ChunkReader> AQParquetMetaData<T> {
    fn ci_to_col_name<'a>(&'a self, col: &'a ColumnIdentifier) -> &'a str {
        match col {
            ColumnIdentifier::Name(s) => &s,
            ColumnIdentifier::Index(i) => self.schema.field(*i).name(),
        }
    }

    pub fn row_counts(&self, col: &ColumnIdentifier) -> UInt64Array {
        let scon = StatisticsConverter::try_new(
            &self.ci_to_col_name(col),
            &self.schema,
            self.metadata.file_metadata().schema_descr(),
        )
        .unwrap();
        scon.row_group_row_counts(self.metadata.row_groups().iter())
            .unwrap()
            .unwrap()
    }

    pub fn null_counts(&self, col: &ColumnIdentifier) -> UInt64Array {
        let scon = StatisticsConverter::try_new(
            &self.ci_to_col_name(col),
            &self.schema,
            self.metadata.file_metadata().schema_descr(),
        )
        .unwrap();
        scon.row_group_null_counts(self.metadata.row_groups().iter())
            .unwrap()
    }

    pub fn mins(&self, col: &ColumnIdentifier) -> Arc<dyn Array> {
        let scon = StatisticsConverter::try_new(
            &self.ci_to_col_name(col),
            &self.schema,
            self.metadata.file_metadata().schema_descr(),
        )
        .unwrap();
        scon.row_group_mins(self.metadata.row_groups().iter())
            .unwrap()
    }

    pub fn maxs(&self, col: &ColumnIdentifier) -> Arc<dyn Array> {
        let scon = StatisticsConverter::try_new(
            &self.ci_to_col_name(col),
            &self.schema,
            self.metadata.file_metadata().schema_descr(),
        )
        .unwrap();
        scon.row_group_maxes(self.metadata.row_groups().iter())
            .unwrap()
    }
}

pub fn stream_parquet(
    mut rdr: ParquetRecordBatchReader,
) -> Receiver<Result<ArrowQuiver, ArrowError>> {
    let (tx, rx) = mpsc::sync_channel(8);
    thread::spawn(move || {
        while let Some(batch) = rdr.next() {
            let batch = batch.map(|rb| ArrowQuiver::from(rb));
            tx.send(batch).unwrap();
        }
    });
    rx
}

pub struct ParquetSource {
    rdr: ParquetRecordBatchReader,
}

impl ParquetSource {
    pub fn new(fp: &Path) -> Result<ParquetSource, io::Error> {
        let f = File::open(fp)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)?;
        let rdr = builder.build()?;

        Ok(ParquetSource { rdr })
    }
}

impl Operator for ParquetSource {
    fn name(&self) -> String {
        format!("Parquet Source")
    }
}

impl Source for ParquetSource {
    fn produce(&mut self, _ctx: &AQExecutorContext) -> Option<Result<ArrowQuiver, Box<dyn Error>>> {
        self.rdr.next().map(|r| {
            r.map(|rb| ArrowQuiver::from(rb))
                .map_err(|e| Box::new(e) as Box<dyn Error>)
        })
    }
}

#[cfg(test)]
pub mod tests {
    use std::{
        fs::{self, File},
        path::Path,
        sync::Arc,
    };

    use arrow_array::{
        cast::AsArray, types::Int32Type, ArrayRef, Int32Array, RecordBatch, StringArray,
    };
    use itertools::Itertools;
    use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};

    use crate::{AQExecutorContext, Source};

    use super::{read_parquet_meta, ParquetSource};

    pub fn create_test_parquet() -> &'static Path {
        let _ = fs::create_dir_all("test_data");
        let p = Path::new("test_data/test.parquet");
        let f = match File::create_new(&p) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return p,
            Err(e) => {
                panic!("{}", e);
            }
        };

        let c1 = Int32Array::from((0..100_000).collect_vec());
        let c2 = StringArray::from(
            (0..100_000)
                .map(|i| {
                    if i % 2 == 0 {
                        Some(format!("s{}", i))
                    } else {
                        None
                    }
                })
                .collect_vec(),
        );

        let b = RecordBatch::try_from_iter(vec![
            ("c1", Arc::new(c1) as ArrayRef),
            ("c2", Arc::new(c2) as ArrayRef),
        ])
        .unwrap();

        let props = WriterProperties::builder()
            .set_max_row_group_size(1000)
            .set_compression(Compression::UNCOMPRESSED)
            .build();

        let mut w = ArrowWriter::try_new(f, b.schema(), Some(props)).unwrap();
        w.write(&b).unwrap();
        w.close().unwrap();
        return p;
    }

    #[test]
    fn test_read_md() {
        let p = create_test_parquet();

        let md = read_parquet_meta(p).unwrap();
        assert_eq!(md.metadata.num_row_groups(), 100);

        md.null_counts(&0.into()).iter().all(|v| v.unwrap() == 0);
        md.null_counts(&1.into()).iter().all(|v| v.unwrap() > 0);

        md.mins(&0.into())
            .as_primitive::<Int32Type>()
            .iter()
            .enumerate()
            .for_each(|(idx, min)| {
                assert_eq!(idx as i32 * 1000, min.unwrap());
            });

        md.maxs(&0.into())
            .as_primitive::<Int32Type>()
            .iter()
            .enumerate()
            .for_each(|(idx, min)| {
                assert_eq!((idx + 1) as i32 * 1000 - 1, min.unwrap());
            });
    }

    #[test]
    fn test_read_parquet() {
        let p = create_test_parquet();
        let mut source = ParquetSource::new(p).unwrap();

        let mut all_rows = Vec::new();
        let ctx = AQExecutorContext::default();

        while let Some(r) = source.produce(&ctx) {
            let r = r.unwrap();
            all_rows.push(r);
        }

        assert_eq!(
            all_rows.iter().map(|q| q.num_rows()).sum::<usize>(),
            100_000
        );
    }
}
