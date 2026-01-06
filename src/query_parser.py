"""
クエリ解析モジュール

自然言語クエリを解析し、実行可能な分析指示に変換する
Phase 1: キーワードベース / Phase 2: LLM統合予定
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any

import pandas as pd


class QueryType(Enum):
    """クエリタイプの列挙"""
    SUM = auto()        # 合計
    MEAN = auto()       # 平均
    COUNT = auto()      # 件数
    GROUPBY = auto()    # グループ別集計
    FILTER = auto()     # フィルタリング
    SORT = auto()       # ソート
    DESCRIBE = auto()   # 基本統計
    UNKNOWN = auto()    # 不明


@dataclass
class ParsedQuery:
    """解析済みクエリを格納するクラス"""
    query_type: QueryType
    target_column: Optional[str] = None
    group_column: Optional[str] = None
    filter_conditions: List[Dict[str, Any]] = field(default_factory=list)
    sort_ascending: bool = True
    aggregation: str = "sum"  # sum, mean, count, min, max
    original_question: str = ""
    confidence: float = 1.0  # 解析の確信度（0.0-1.0）


class QueryParser:
    """
    自然言語クエリを解析するクラス

    Phase 1では単純なキーワードマッチング、
    Phase 2でLLM統合を予定
    """

    # クエリタイプのキーワードマッピング
    QUERY_KEYWORDS: Dict[QueryType, List[str]] = {
        QueryType.SUM: ["合計", "総計", "トータル", "total", "sum", "合わせて"],
        QueryType.MEAN: ["平均", "アベレージ", "average", "mean", "avg"],
        QueryType.COUNT: ["件数", "数", "何件", "いくつ", "count", "how many"],
        QueryType.GROUPBY: ["別", "ごと", "毎", "by", "per", "each", "グループ"],
        QueryType.FILTER: ["だけ", "のみ", "以上", "以下", "より", "未満", "where", "filter"],
        QueryType.SORT: ["順", "ソート", "並べ", "sort", "order", "ランキング", "top"],
        QueryType.DESCRIBE: ["概要", "統計", "サマリ", "describe", "summary", "overview"],
    }

    # 日本語カラム名キーワード
    COLUMN_KEYWORDS: Dict[str, List[str]] = {
        "region": ["地域", "エリア", "都道府県", "region", "area"],
        "product": ["製品", "商品", "プロダクト", "product", "item"],
        "salesperson": ["担当", "営業", "販売員", "salesperson", "sales_rep"],
        "date": ["日付", "月", "年", "期間", "date", "month", "year", "period"],
        "quantity": ["数量", "個数", "quantity", "qty", "amount"],
        "total_sales": ["売上", "金額", "収益", "sales", "revenue", "total"],
        "unit_price": ["単価", "価格", "price", "unit_price"],
    }

    def __init__(self, schema: Optional[Dict[str, str]] = None):
        """
        Args:
            schema: カラム名と型のマッピング（オプション）
        """
        self._schema = schema or {}
        self._column_names: List[str] = list(self._schema.keys())

    def set_schema(self, df: pd.DataFrame) -> None:
        """DataFrameからスキーマを設定"""
        self._schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self._column_names = list(df.columns)

    def parse(self, question: str) -> ParsedQuery:
        """
        自然言語クエリを解析

        Args:
            question: 自然言語の質問

        Returns:
            解析結果
        """
        question_lower = question.lower()

        # クエリタイプを判定
        query_type = self._detect_query_type(question)

        # 対象カラムを検出
        target_column = self._find_target_column(question)

        # グループカラムを検出
        group_column = self._find_group_column(question)

        # フィルタ条件を検出
        filter_conditions = self._parse_filters(question)

        # 集計方法を決定
        aggregation = self._determine_aggregation(query_type)

        # 確信度を計算
        confidence = self._calculate_confidence(
            query_type, target_column, group_column, question
        )

        return ParsedQuery(
            query_type=query_type,
            target_column=target_column,
            group_column=group_column,
            filter_conditions=filter_conditions,
            aggregation=aggregation,
            original_question=question,
            confidence=confidence,
        )

    def _detect_query_type(self, question: str) -> QueryType:
        """クエリタイプを検出"""
        question_lower = question.lower()

        # 各クエリタイプのキーワードをチェック
        type_scores: Dict[QueryType, int] = {}

        for qtype, keywords in self.QUERY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in question_lower or kw in question)
            if score > 0:
                type_scores[qtype] = score

        if not type_scores:
            return QueryType.DESCRIBE

        # グループ別が検出された場合、他のタイプと組み合わせる
        if QueryType.GROUPBY in type_scores:
            # 合計や平均も検出されていればGROUPBYを優先
            return QueryType.GROUPBY

        # 最高スコアのタイプを返す
        return max(type_scores, key=type_scores.get)

    def _find_target_column(self, question: str) -> Optional[str]:
        """数値対象カラムを検出"""
        question_lower = question.lower()

        # 直接カラム名が含まれているかチェック
        for col in self._column_names:
            if col.lower() in question_lower or col in question:
                # 数値型カラムを優先
                if self._schema.get(col, "").startswith(("int", "float")):
                    return col

        # キーワードマッチング
        for col, keywords in self.COLUMN_KEYWORDS.items():
            for kw in keywords:
                if kw in question_lower or kw in question:
                    # 実際のカラム名と照合
                    for actual_col in self._column_names:
                        if col in actual_col.lower():
                            return actual_col

        # 数値型の最初のカラムを返す
        for col, dtype in self._schema.items():
            if dtype.startswith(("int", "float")):
                return col

        return None

    def _find_group_column(self, question: str) -> Optional[str]:
        """グループ化カラムを検出"""
        question_lower = question.lower()

        # 「〜別」「〜ごと」パターンを検出
        patterns = [
            r"(.+?)別",
            r"(.+?)ごと",
            r"(.+?)毎",
            r"by\s+(\w+)",
            r"per\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                keyword = match.group(1).strip()
                # キーワードからカラムを特定
                for col, keywords in self.COLUMN_KEYWORDS.items():
                    if keyword in keywords or any(k in keyword for k in keywords):
                        for actual_col in self._column_names:
                            if col in actual_col.lower():
                                return actual_col

        # カテゴリ型の最初のカラムを返す
        for col, dtype in self._schema.items():
            if dtype == "object":
                return col

        return None

    def _parse_filters(self, question: str) -> List[Dict[str, Any]]:
        """フィルタ条件を解析"""
        filters = []

        # 数値比較パターン
        patterns = [
            (r"(\d+)以上", ">="),
            (r"(\d+)以下", "<="),
            (r"(\d+)より大きい", ">"),
            (r"(\d+)より小さい", "<"),
            (r"(\d+)未満", "<"),
        ]

        for pattern, operator in patterns:
            match = re.search(pattern, question)
            if match:
                value = int(match.group(1))
                filters.append({
                    "operator": operator,
                    "value": value,
                    "column": None,  # 後で特定
                })

        # カテゴリフィルタ（「東京の」「製品Aの」等）
        # TODO: より高度な解析が必要

        return filters

    def _determine_aggregation(self, query_type: QueryType) -> str:
        """集計方法を決定"""
        mapping = {
            QueryType.SUM: "sum",
            QueryType.MEAN: "mean",
            QueryType.COUNT: "count",
            QueryType.GROUPBY: "sum",  # デフォルト
            QueryType.DESCRIBE: "describe",
        }
        return mapping.get(query_type, "sum")

    def _calculate_confidence(
        self,
        query_type: QueryType,
        target_column: Optional[str],
        group_column: Optional[str],
        question: str,
    ) -> float:
        """解析の確信度を計算"""
        confidence = 0.5  # 基本値

        # クエリタイプが明確
        if query_type != QueryType.UNKNOWN:
            confidence += 0.2

        # 対象カラムが特定できた
        if target_column:
            confidence += 0.15

        # グループカラムが必要で特定できた
        if query_type == QueryType.GROUPBY and group_column:
            confidence += 0.15

        return min(confidence, 1.0)


def create_parser_from_dataframe(df: pd.DataFrame) -> QueryParser:
    """DataFrameからQueryParserを作成するヘルパー関数"""
    parser = QueryParser()
    parser.set_schema(df)
    return parser
