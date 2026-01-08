"""
InsightAnalyzerAI - Streamlit Web UI

データ分析を自然言語で行えるWebインターフェース
Phase 4: Web UI実装
Phase 5: 認証・課金統合
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io
import sys

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insight_analyzer import InsightAnalyzer, AnalysisResult
from src.auth import AuthManager, PlanType, render_auth_ui
from src.billing import render_pricing_ui, render_billing_status


# ページ設定
st.set_page_config(
    page_title="InsightAnalyzerAI - データ分析アシスタント",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """セッション状態を初期化"""
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "auth_manager" not in st.session_state:
        st.session_state.auth_manager = AuthManager()


def load_data_from_file(uploaded_file) -> bool:
    """アップロードされたファイルからデータを読み込む"""
    try:
        # 認証マネージャーを取得
        auth_manager = st.session_state.auth_manager

        # ファイルサイズチェック
        file_size = uploaded_file.size
        can_upload, message = auth_manager.can_upload_file(file_size)
        if not can_upload:
            st.error(message)
            return False

        # ファイル拡張子を取得
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()

        # ファイル内容を読み込み
        if file_ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_ext == ".parquet":
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"サポートされていないファイル形式です: {file_ext}")
            return False

        # LLM使用可否をプランから判定
        use_llm = auth_manager.can_use_llm()

        # Analyzerを初期化
        st.session_state.analyzer = InsightAnalyzer(df, use_llm=use_llm)
        st.session_state.data_loaded = True
        st.session_state.file_name = file_name

        # ファイルアップロードを記録
        auth_manager.usage_tracker.add_file_upload(file_size)

        return True

    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {str(e)}")
        return False


def display_data_info():
    """データ情報を表示"""
    if st.session_state.analyzer is None:
        return

    analyzer = st.session_state.analyzer
    df = analyzer.dataframe

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("行数", f"{len(df):,}")
    with col2:
        st.metric("列数", f"{len(df.columns):,}")
    with col3:
        llm_status = "✅ 有効" if analyzer.llm_available else "⚠️ 無効"
        st.metric("LLM統合", llm_status)


def display_schema():
    """スキーマ情報を表示"""
    if st.session_state.analyzer is None:
        return

    with st.expander("📋 データスキーマ", expanded=False):
        st.code(st.session_state.analyzer.schema)


def display_insights():
    """自動インサイトを表示"""
    if st.session_state.analyzer is None:
        return

    with st.expander("💡 自動インサイト", expanded=False):
        insights = st.session_state.analyzer.get_insights()
        for insight in insights:
            st.write(f"• {insight}")


def display_data_preview():
    """データプレビューを表示"""
    if st.session_state.analyzer is None:
        return

    with st.expander("👁️ データプレビュー", expanded=False):
        df = st.session_state.analyzer.dataframe
        st.dataframe(df.head(100), use_container_width=True)


def process_query(question: str, generate_chart: bool, explain_result: bool):
    """クエリを処理して結果を返す"""
    if st.session_state.analyzer is None:
        return None

    # クエリ実行可否をチェック
    auth_manager = st.session_state.auth_manager
    can_execute, message = auth_manager.can_execute_query()
    if not can_execute:
        st.error(message)
        return None

    # チャート機能の制限チェック
    if generate_chart and not auth_manager.can_use_charts():
        st.warning("チャート機能は有料プランで利用できます")
        generate_chart = False

    result = st.session_state.analyzer.ask(
        question,
        generate_chart=generate_chart,
        explain_result=explain_result,
    )

    # クエリカウントを増加
    auth_manager.usage_tracker.increment_query_count()

    # 履歴に追加
    st.session_state.history.append({
        "question": question,
        "result": result,
    })

    return result


def create_plotly_chart(data: pd.DataFrame, question: str = "") -> go.Figure:
    """
    データからPlotlyチャートを自動生成

    Args:
        data: チャート用データ
        question: 元の質問（タイトル用）

    Returns:
        Plotlyフィギュア
    """
    if data is None or len(data) == 0:
        return None

    # データの特性を分析
    num_rows = len(data)
    num_cols = len(data.columns)

    # 数値列を取得
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

    # インデックスがカテゴリ的かどうか
    index_is_categorical = data.index.dtype == 'object' or num_rows <= 20

    # チャートタイプを自動判定
    if num_rows <= 6 and len(numeric_cols) == 1:
        # 少数カテゴリ: 円グラフ
        fig = px.pie(
            data,
            values=data.columns[0],
            names=data.index,
            title=question or "分析結果",
        )
    elif num_rows <= 15:
        # 中程度カテゴリ: 棒グラフ
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.bar(
                data.reset_index(),
                x='index',
                y=col,
                title=question or "分析結果",
                labels={'index': '', col: col},
            )
            fig.update_layout(xaxis_tickangle=-45)
    elif num_cols >= 2 and len(numeric_cols) >= 2:
        # 2列以上の数値: 散布図
        fig = px.scatter(
            data,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=question or "分析結果",
        )
    else:
        # デフォルト: 横棒グラフ（上位10件）
        plot_data = data.head(10)
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.bar(
                plot_data.reset_index(),
                y='index',
                x=col,
                orientation='h',
                title=question or f"上位{len(plot_data)}件",
                labels={'index': '', col: col},
            )
        else:
            return None

    # 共通スタイル設定
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Meiryo, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def display_result(result: AnalysisResult, show_chart: bool = False):
    """分析結果を表示"""
    if not result.success:
        st.error(f"エラー: {result.error}")
        return

    # 回答を表示
    st.markdown("### 📊 分析結果")
    st.write(result.answer)

    # LLM説明がある場合
    if result.llm_explanation:
        with st.expander("🤖 AIによる解説", expanded=True):
            st.write(result.llm_explanation)

    # チャート表示（Plotly）
    if show_chart and result.data is not None and len(result.data) > 0:
        fig = create_plotly_chart(result.data, result.query_used or "")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    # データがある場合はテーブル表示
    if result.data is not None and len(result.data) > 0:
        with st.expander("📋 詳細データ", expanded=not show_chart):
            st.dataframe(result.data, use_container_width=True)

    # メタ情報
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"実行時間: {result.execution_time_ms:.2f}ms")
    with col2:
        st.caption(f"信頼度: {result.confidence:.0%}")
    with col3:
        if result.llm_used:
            st.caption("🤖 LLM使用")
        else:
            st.caption("📝 キーワード解析")

    # 使用したクエリ
    if result.query_used:
        with st.expander("🔍 実行されたコード", expanded=False):
            st.code(result.query_used, language="python")


def display_history():
    """履歴を表示"""
    if not st.session_state.history:
        return

    with st.expander("📜 クエリ履歴", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            st.write(f"**Q{len(st.session_state.history) - i}:** {item['question']}")
            if item['result'].success:
                st.write(f"A: {item['result'].answer[:200]}...")
            else:
                st.write(f"⚠️ エラー: {item['result'].error}")
            st.divider()


def main():
    """メインアプリケーション"""
    init_session_state()

    # ヘッダー
    st.title("📊 InsightAnalyzerAI")
    st.markdown("*自然言語でデータを分析するAIアシスタント*")

    # 認証UI（サイドバー内）
    render_auth_ui()

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")

        # ファイルアップロード
        st.subheader("📂 データファイル")
        uploaded_file = st.file_uploader(
            "CSV, Excel, Parquet ファイルをアップロード",
            type=["csv", "xlsx", "xls", "parquet"],
            help="最大50MBまでのファイルに対応"
        )

        if uploaded_file is not None:
            if not st.session_state.data_loaded or \
               st.session_state.get("file_name") != uploaded_file.name:
                with st.spinner("データを読み込み中..."):
                    if load_data_from_file(uploaded_file):
                        st.success(f"✅ {uploaded_file.name} を読み込みました")

        # サンプルデータ使用オプション
        st.divider()
        if st.button("📊 サンプルデータを使用"):
            sample_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
            if sample_path.exists():
                st.session_state.analyzer = InsightAnalyzer(str(sample_path), use_llm=True)
                st.session_state.data_loaded = True
                st.session_state.file_name = "sample_sales.csv"
                st.success("✅ サンプルデータを読み込みました")
            else:
                st.warning("サンプルデータファイルが見つかりません")

        st.divider()

        # オプション
        st.subheader("🎛️ オプション")
        generate_chart = st.checkbox("チャートを生成", value=False)
        explain_result = st.checkbox("AIで結果を説明", value=True)

        # 履歴クリア
        st.divider()
        if st.button("🗑️ 履歴をクリア"):
            st.session_state.history = []
            st.rerun()

    # メインコンテンツ
    if not st.session_state.data_loaded:
        # ウェルカムメッセージ
        st.info("👈 左のサイドバーからデータファイルをアップロードしてください")

        st.markdown("""
        ### 🚀 使い方
        1. **データをアップロード**: CSV, Excel, Parquet ファイルに対応
        2. **質問を入力**: 自然言語で分析したい内容を入力
        3. **結果を確認**: AIが自動的にデータを分析し、回答を生成

        ### 💡 質問例
        - 「売上の合計はいくら？」
        - 「カテゴリ別の売上を教えて」
        - 「データの概要を教えて」
        - 「最も売れている商品は？」
        """)

        return

    # データ情報
    display_data_info()

    # タブで情報を整理
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 クエリ", "📊 データ情報", "📜 履歴", "💰 プラン"])

    with tab1:
        # クエリ入力
        st.subheader("💬 質問を入力")

        # 質問例ボタン
        example_queries = [
            "データの概要を教えて",
            "売上の合計は？",
            "カテゴリ別の売上",
            "上位5件を表示",
        ]

        cols = st.columns(len(example_queries))
        selected_example = None
        for i, (col, query) in enumerate(zip(cols, example_queries)):
            if col.button(query, key=f"example_{i}", use_container_width=True):
                selected_example = query

        # テキスト入力
        question = st.text_input(
            "質問",
            value=selected_example or "",
            placeholder="例: 売上の合計を教えて",
            label_visibility="collapsed",
        )

        # 実行ボタン
        if st.button("🔍 分析実行", type="primary", disabled=not question):
            with st.spinner("分析中..."):
                result = process_query(
                    question,
                    generate_chart,
                    explain_result,
                )
                if result:
                    display_result(result, show_chart=generate_chart)

        # 最新の結果を表示
        if st.session_state.history and not question:
            st.divider()
            st.subheader("最新の分析結果")
            display_result(st.session_state.history[-1]["result"], show_chart=generate_chart)

    with tab2:
        display_schema()
        display_insights()
        display_data_preview()

    with tab3:
        display_history()

    with tab4:
        render_pricing_ui()
        render_billing_status()


if __name__ == "__main__":
    main()
