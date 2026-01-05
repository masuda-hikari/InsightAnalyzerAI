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
