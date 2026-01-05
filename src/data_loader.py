"""
データ読み込みモジュール

CSV、Excel等のデータファイルを読み込み、Pandas DataFrameに変換する
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class DataLoader:
    """データファイルの読み込みを担当するクラス"""

    # サポートする拡張子
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}

    # デフォルトの最大行数（サンプリング用）
    DEFAULT_MAX_ROWS = 100_000

    def __init__(self, max_rows: Optional[int] = None):
        """
        Args:
            max_rows: 読み込む最大行数。Noneの場合は全行読み込み
        """
        self.max_rows = max_rows or self.DEFAULT_MAX_ROWS
        self._dataframe: Optional[pd.DataFrame] = None
        self._file_path: Optional[Path] = None
        self._metadata: dict = {}

    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        ファイルを読み込んでDataFrameを返す

        Args:
            file_path: 読み込むファイルのパス

        Returns:
            読み込んだデータのDataFrame

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: サポートされていない拡張子の場合
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"サポートされていない拡張子: {ext}. "
                f"対応形式: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        self._file_path = path

        # 拡張子に応じた読み込み
        if ext == '.csv':
            df = self._load_csv(path)
        elif ext in {'.xlsx', '.xls'}:
            df = self._load_excel(path)
        elif ext == '.json':
            df = self._load_json(path)
        elif ext == '.parquet':
            df = self._load_parquet(path)
        else:
            raise ValueError(f"未実装の拡張子: {ext}")

        # 大規模データのサンプリング
        if len(df) > self.max_rows:
            df = df.sample(n=self.max_rows, random_state=42)
            self._metadata['sampled'] = True
            self._metadata['original_rows'] = len(df)
        else:
            self._metadata['sampled'] = False

        self._dataframe = df
        self._update_metadata()

        return df

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """CSV読み込み"""
        return pd.read_csv(path, encoding='utf-8')

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Excel読み込み"""
        return pd.read_excel(path)

    def _load_json(self, path: Path) -> pd.DataFrame:
        """JSON読み込み"""
        return pd.read_json(path)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Parquet読み込み"""
        return pd.read_parquet(path)

    def _update_metadata(self) -> None:
        """メタデータを更新"""
        if self._dataframe is not None:
            self._metadata.update({
                'rows': len(self._dataframe),
                'columns': len(self._dataframe.columns),
                'column_names': list(self._dataframe.columns),
                'dtypes': {col: str(dtype) for col, dtype in self._dataframe.dtypes.items()},
                'memory_usage_mb': self._dataframe.memory_usage(deep=True).sum() / (1024 * 1024),
            })

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """読み込んだDataFrameを取得"""
        return self._dataframe

    @property
    def metadata(self) -> dict:
        """メタデータを取得"""
        return self._metadata

    def get_schema(self) -> str:
        """
        データスキーマの文字列表現を取得（LLMプロンプト用）

        Returns:
            カラム名と型の情報を含む文字列
        """
        if self._dataframe is None:
            return "データが読み込まれていません"

        lines = ["データスキーマ:", f"行数: {len(self._dataframe)}", "カラム:"]

        for col in self._dataframe.columns:
            dtype = self._dataframe[col].dtype
            sample = self._dataframe[col].dropna().head(3).tolist()
            lines.append(f"  - {col} ({dtype}): 例 {sample}")

        return "\n".join(lines)

    def get_summary_stats(self) -> pd.DataFrame:
        """
        基本統計量を取得

        Returns:
            数値カラムの統計量DataFrame
        """
        if self._dataframe is None:
            raise ValueError("データが読み込まれていません")

        return self._dataframe.describe()
