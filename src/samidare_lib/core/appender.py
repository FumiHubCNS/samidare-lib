import pathlib
import sys
import pyarrow as pa
import pyarrow.parquet as pq

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

class PulseParquetAppender:                    

    def __init__(self, path: str, batch_rows: int = 8192):
        self.schema = pa.schema([            
            ("chip",          pa.int64()),              
            ("timestamp", pa.list_(pa.int64())),
            ("sample_index", pa.list_(pa.int64())),            
            ("samples_value", pa.list_(pa.list_(pa.int64()))) 
        ])
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.batch_rows = batch_rows
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def _flush(self):
        if self._n == 0:
            return
        arrays = []
        for name, typ in zip(self.schema.names, self.schema.types):
            vals = self._buf[name]
            arrays.append(pa.array(vals, type=typ))
        table = pa.Table.from_arrays(arrays, names=self.schema.names)

        self.writer.write_table(table, row_group_size=self._n)
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def append(self, row: dict):
        for name in self.schema.names:
            self._buf[name].append(row.get(name))
        self._n += 1
        if self._n >= self.batch_rows:
            self._flush()

    def close(self):
        self._flush()
        self.writer.close()
        
class SAMPADataParquetAppender:
    
    def __init__(self, path: str, batch_rows: int = 8192):
        self.schema = pa.schema([
            ("data_block",    pa.int64()),
            ("error_level",   pa.int64()),
            ("timestamp",     pa.int64()),
            ("chip",          pa.int64()),
            ("sample_index",  pa.int64()),
            ("samples_value", pa.list_(pa.int64())),
        ])
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.batch_rows = batch_rows
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def _flush(self):
        if self._n == 0:
            return
        arrays = []
        for name, typ in zip(self.schema.names, self.schema.types):
            vals = self._buf[name]
            arrays.append(pa.array(vals, type=typ))
        table = pa.Table.from_arrays(arrays, names=self.schema.names)
    
        self.writer.write_table(table, row_group_size=self._n)
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def append(self, row: dict):
        for name in self.schema.names:
            self._buf[name].append(row.get(name))
        self._n += 1
        if self._n >= self.batch_rows:
            self._flush()

    def close(self):
        self._flush()
        self.writer.close()
