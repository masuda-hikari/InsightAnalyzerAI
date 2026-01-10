"""
InsightAnalyzerAI - AI駆動のデータ分析アシスタント

主要クラス:
    InsightAnalyzer: メイン分析クラス
    DataLoader: データ読み込み
    QueryParser: クエリ解析
    QueryExecutor: クエリ実行
    Visualizer: 可視化
    LLMHandler: LLM統合（Phase 2）
    InsightEngine: 自動インサイト発見（Phase 5）
"""

__version__ = "0.3.0"  # Phase 5: 自動インサイト機能追加
__author__ = "InsightAnalyzerAI Team"

from .insight_analyzer import InsightAnalyzer, AnalysisResult
from .data_loader import DataLoader
from .query_parser import QueryParser, ParsedQuery, QueryType
from .executor import QueryExecutor, SafeExecutor, ExecutionResult
from .visualizer import Visualizer, ChartConfig, ChartType, ChartResult
from .llm_handler import LLMHandler, LLMConfig, LLMResponse, LLMQueryParser
from .insight_engine import (
    InsightEngine,
    InsightReport,
    Insight,
    InsightType,
    InsightSeverity,
    generate_insights,
)

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
    # 自動インサイト（Phase 5）
    "InsightEngine",
    "InsightReport",
    "Insight",
    "InsightType",
    "InsightSeverity",
    "generate_insights",
]
