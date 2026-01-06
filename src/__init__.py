"""
InsightAnalyzerAI - AI駆動のデータ分析アシスタント

主要クラス:
    InsightAnalyzer: メイン分析クラス
    DataLoader: データ読み込み
    QueryParser: クエリ解析
    QueryExecutor: クエリ実行
    Visualizer: 可視化
    LLMHandler: LLM統合（Phase 2）
"""

__version__ = "0.2.0"
__author__ = "InsightAnalyzerAI Team"

from .insight_analyzer import InsightAnalyzer, AnalysisResult
from .data_loader import DataLoader
from .query_parser import QueryParser, ParsedQuery, QueryType
from .executor import QueryExecutor, SafeExecutor, ExecutionResult
from .visualizer import Visualizer, ChartConfig, ChartType, ChartResult
from .llm_handler import LLMHandler, LLMConfig, LLMResponse, LLMQueryParser

__all__ = [
    # メイン
    "InsightAnalyzer",
    "AnalysisResult",
    # データ
    "DataLoader",
    # クエリ
    "QueryParser",
    "ParsedQuery",
    "QueryType",
    # 実行
    "QueryExecutor",
    "SafeExecutor",
    "ExecutionResult",
    # 可視化
    "Visualizer",
    "ChartConfig",
    "ChartType",
    "ChartResult",
    # LLM（Phase 2）
    "LLMHandler",
    "LLMConfig",
    "LLMResponse",
    "LLMQueryParser",
]
