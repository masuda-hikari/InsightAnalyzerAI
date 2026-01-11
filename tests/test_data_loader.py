"""DataLoaderのテスト"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# srcをパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


class TestDataLoader:
    """DataLoaderクラスのテスト"""

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """サンプルCSVのパスを返す"""
        return Path(__file__).parent.parent / "data" / "sample_sales.csv"

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンスを返す"""
        return DataLoader()

    def test_load_csv_success(self, loader: DataLoader, sample_csv_path: Path):
        """CSVファイルの正常読み込み"""
        df = loader.load(sample_csv_path)

        assert df is not None
        assert len(df) == 25  # サンプルデータの行数
        assert "region" in df.columns
        assert "total_sales" in df.columns

    def test_load_file_not_found(self, loader: DataLoader):
        """存在しないファイルの読み込み"""
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.csv")

    def test_load_unsupported_extension(self, loader: DataLoader):
        """サポートされていない拡張子"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test data")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="サポートされていない拡張子"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_metadata_after_load(self, loader: DataLoader, sample_csv_path: Path):
        """読み込み後のメタデータ"""
        loader.load(sample_csv_path)

        metadata = loader.metadata
        assert metadata["rows"] == 25
        assert metadata["columns"] == 7
        assert "date" in metadata["column_names"]
        assert metadata["memory_usage_mb"] > 0

    def test_get_schema(self, loader: DataLoader, sample_csv_path: Path):
        """スキーマ取得"""
        loader.load(sample_csv_path)
        schema = loader.get_schema()

        assert "データスキーマ" in schema
        assert "行数: 25" in schema
        assert "region" in schema

    def test_get_schema_without_data(self, loader: DataLoader):
        """データなしでのスキーマ取得"""
        schema = loader.get_schema()
        assert "データが読み込まれていません" in schema

    def test_get_summary_stats(self, loader: DataLoader, sample_csv_path: Path):
        """基本統計量の取得"""
        loader.load(sample_csv_path)
        stats = loader.get_summary_stats()

        assert isinstance(stats, pd.DataFrame)
        assert "quantity" in stats.columns
        assert "total_sales" in stats.columns

    def test_get_summary_stats_without_data(self, loader: DataLoader):
        """データなしでの統計量取得"""
        with pytest.raises(ValueError, match="データが読み込まれていません"):
            loader.get_summary_stats()


class TestDataLoaderSampling:
    """サンプリング機能のテスト"""

    def test_sampling_large_data(self):
        """大規模データのサンプリング"""
        # 小さいmax_rowsでテスト
        loader = DataLoader(max_rows=10)

        # 25行のサンプルデータを読み込む
        sample_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
        df = loader.load(sample_path)

        # サンプリングされている
        assert len(df) == 10
        assert loader.metadata["sampled"] is True


class TestDataLoaderExcel:
    """Excel読み込みテスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_xlsx_file(self, loader: DataLoader):
        """xlsxファイルの読み込み"""
        # テスト用のxlsxファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            # pandasでExcelファイルを作成
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            df.to_excel(temp_path, index=False)

            # 読み込みテスト
            result = loader.load(temp_path)
            assert result is not None
            assert len(result) == 3
            assert "a" in result.columns
        except ImportError:
            # openpyxlがない場合はスキップ
            pytest.skip("openpyxl not installed")
        finally:
            os.unlink(temp_path)

    def test_load_empty_xlsx_file(self, loader: DataLoader):
        """空のxlsxファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            # 空のExcelファイルを作成
            df = pd.DataFrame()
            df.to_excel(temp_path, index=False)

            with pytest.raises(ValueError, match="空です|エラー"):
                loader.load(temp_path)
        except ImportError:
            pytest.skip("openpyxl not installed")
        finally:
            os.unlink(temp_path)


class TestDataLoaderJSON:
    """JSON読み込みテスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_json_file(self, loader: DataLoader):
        """JSONファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            import json
            json.dump([{"a": 1, "b": 2}, {"a": 3, "b": 4}], f)
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result is not None
            assert len(result) == 2
            assert "a" in result.columns
        finally:
            os.unlink(temp_path)

    def test_load_empty_json_file(self, loader: DataLoader):
        """空のJSONファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            import json
            json.dump([], f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="空です"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json_file(self, loader: DataLoader):
        """不正なJSONファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("invalid json {{{")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestDataLoaderParquet:
    """Parquet読み込みテスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_parquet_file(self, loader: DataLoader):
        """Parquetファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # pandasでParquetファイルを作成
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            df.to_parquet(temp_path, index=False)

            # 読み込みテスト
            result = loader.load(temp_path)
            assert result is not None
            assert len(result) == 3
            assert "a" in result.columns
        except ImportError:
            # pyarrowがない場合はスキップ
            pytest.skip("pyarrow not installed")
        finally:
            os.unlink(temp_path)

    def test_load_empty_parquet_file(self, loader: DataLoader):
        """空のParquetファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # 空のParquetファイルを作成
            df = pd.DataFrame()
            df.to_parquet(temp_path, index=False)

            with pytest.raises(ValueError, match="空です"):
                loader.load(temp_path)
        except ImportError:
            pytest.skip("pyarrow not installed")
        finally:
            os.unlink(temp_path)


class TestDataLoaderCSVEdgeCases:
    """CSV読み込みのエッジケーステスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_csv_utf8_bom(self, loader: DataLoader):
        """UTF-8 BOM付きCSVの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
            # BOM付きUTF-8
            f.write(b'\xef\xbb\xbfa,b\n1,2\n3,4')
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result is not None
            assert len(result) == 2
        finally:
            os.unlink(temp_path)

    def test_load_csv_shift_jis(self, loader: DataLoader):
        """Shift-JIS CSVの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
            # Shift-JISでエンコード
            content = "名前,金額\n山田,1000\n鈴木,2000"
            f.write(content.encode('shift_jis'))
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result is not None
            assert len(result) == 2
            assert "名前" in result.columns
        finally:
            os.unlink(temp_path)

    def test_load_csv_empty_data(self, loader: DataLoader):
        """空のCSVファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="データがありません"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_csv_header_only(self, loader: DataLoader):
        """ヘッダーのみのCSVファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("a,b,c\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="空です"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestDataLoaderProperties:
    """DataLoaderプロパティのテスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """サンプルCSVのパス"""
        return Path(__file__).parent.parent / "data" / "sample_sales.csv"

    def test_dataframe_property_before_load(self, loader: DataLoader):
        """読み込み前のdataframeプロパティ"""
        assert loader.dataframe is None

    def test_dataframe_property_after_load(self, loader: DataLoader, sample_csv_path: Path):
        """読み込み後のdataframeプロパティ"""
        loader.load(sample_csv_path)
        assert loader.dataframe is not None
        assert len(loader.dataframe) == 25

    def test_metadata_property_after_load(self, loader: DataLoader, sample_csv_path: Path):
        """読み込み後のmetadataプロパティ"""
        loader.load(sample_csv_path)
        metadata = loader.metadata

        assert "rows" in metadata
        assert "columns" in metadata
        assert "column_names" in metadata
        assert "dtypes" in metadata
        assert "memory_usage_mb" in metadata
        assert "sampled" in metadata

    def test_supported_extensions_constant(self):
        """サポート拡張子定数の確認"""
        assert '.csv' in DataLoader.SUPPORTED_EXTENSIONS
        assert '.xlsx' in DataLoader.SUPPORTED_EXTENSIONS
        assert '.xls' in DataLoader.SUPPORTED_EXTENSIONS
        assert '.json' in DataLoader.SUPPORTED_EXTENSIONS
        assert '.parquet' in DataLoader.SUPPORTED_EXTENSIONS

    def test_default_max_rows_constant(self):
        """デフォルト最大行数定数の確認"""
        assert DataLoader.DEFAULT_MAX_ROWS == 100_000

    def test_custom_max_rows(self):
        """カスタム最大行数の設定"""
        loader = DataLoader(max_rows=50)
        assert loader.max_rows == 50

    def test_default_max_rows_initialization(self):
        """デフォルト最大行数の初期化"""
        loader = DataLoader()
        assert loader.max_rows == DataLoader.DEFAULT_MAX_ROWS


class TestDataLoaderEncodingEdgeCases:
    """エンコーディング関連のエッジケーステスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_csv_with_invalid_encoding(self, loader: DataLoader):
        """すべてのエンコーディングで読み込めないファイル"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
            # 完全に無効なバイトシーケンス
            f.write(b'\xff\xfe\x00\x01\x00\x02\x00\x03\xff\xff\xff')
            temp_path = f.name

        try:
            # ValueErrorが発生することを確認（メッセージは環境依存のためmatchは使わない）
            with pytest.raises(ValueError):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_csv_euc_jp(self, loader: DataLoader):
        """EUC-JP CSVの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
            content = "名前,金額\n山田,1000\n鈴木,2000"
            f.write(content.encode('euc-jp'))
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result is not None
            assert len(result) == 2
        finally:
            os.unlink(temp_path)


class TestDataLoaderExcelEdgeCases:
    """Excel読み込みのエッジケーステスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_corrupted_xlsx_file(self, loader: DataLoader):
        """破損したxlsxファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, mode="wb") as f:
            f.write(b'PK\x03\x04corrupted_data_here')
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestDataLoaderParquetEdgeCases:
    """Parquet読み込みのエッジケーステスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_corrupted_parquet_file(self, loader: DataLoader):
        """破損したParquetファイルの読み込み"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, mode="wb") as f:
            f.write(b'PAR1corrupted_data_here')
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestDataLoaderJSONEdgeCases:
    """JSON読み込みのエッジケーステスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    def test_load_json_trailing_data(self, loader: DataLoader):
        """余分なデータがあるJSONファイル"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write('[{"a": 1}]extra_trailing_data')
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="フォーマットが不正|JSON"):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_json_nested_object(self, loader: DataLoader):
        """ネストしたオブジェクトのJSONファイル"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            import json
            json.dump({"data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}, f)
            temp_path = f.name

        try:
            result = loader.load(temp_path)
            assert result is not None
        finally:
            os.unlink(temp_path)


class TestDataLoaderSamplingEdgeCases:
    """サンプリング機能のエッジケーステスト"""

    def test_sampling_exact_max_rows(self):
        """最大行数と同じ行数のデータ"""
        loader = DataLoader(max_rows=25)

        sample_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
        df = loader.load(sample_path)

        # 25行なのでサンプリングされない
        assert len(df) == 25
        assert loader.metadata["sampled"] is False

    def test_sampling_one_less_than_max(self):
        """最大行数より1行少ないデータ"""
        loader = DataLoader(max_rows=26)

        sample_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
        df = loader.load(sample_path)

        # 25行で最大26行なのでサンプリングされない
        assert len(df) == 25
        assert loader.metadata["sampled"] is False

    def test_sampling_very_small_max_rows(self):
        """非常に小さい最大行数"""
        loader = DataLoader(max_rows=1)

        sample_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
        df = loader.load(sample_path)

        assert len(df) == 1
        assert loader.metadata["sampled"] is True


class TestDataLoaderMetadata:
    """メタデータ関連のテスト"""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """DataLoaderインスタンス"""
        return DataLoader()

    @pytest.fixture
    def sample_csv_path(self) -> Path:
        """サンプルCSVのパス"""
        return Path(__file__).parent.parent / "data" / "sample_sales.csv"

    def test_metadata_dtypes(self, loader: DataLoader, sample_csv_path: Path):
        """メタデータのdtypes確認"""
        loader.load(sample_csv_path)
        metadata = loader.metadata

        assert "dtypes" in metadata
        assert isinstance(metadata["dtypes"], dict)
        # 各カラムの型が文字列として格納されている
        for col, dtype in metadata["dtypes"].items():
            assert isinstance(dtype, str)

    def test_metadata_memory_usage(self, loader: DataLoader, sample_csv_path: Path):
        """メタデータのメモリ使用量確認"""
        loader.load(sample_csv_path)
        metadata = loader.metadata

        assert "memory_usage_mb" in metadata
        assert metadata["memory_usage_mb"] > 0
        assert isinstance(metadata["memory_usage_mb"], float)

    def test_get_schema_with_numeric_columns(self, loader: DataLoader, sample_csv_path: Path):
        """数値カラムを含むスキーマ取得"""
        loader.load(sample_csv_path)
        schema = loader.get_schema()

        # 数値カラムが含まれている
        assert "quantity" in schema or "total_sales" in schema

    def test_get_schema_sample_values(self, loader: DataLoader, sample_csv_path: Path):
        """スキーマにサンプル値が含まれる"""
        loader.load(sample_csv_path)
        schema = loader.get_schema()

        # "例"というキーワードが含まれている
        assert "例" in schema
