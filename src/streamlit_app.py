"""
InsightAnalyzerAI - Streamlit Web UI

データ分析を自然言語で行えるWebインターフェース
Phase 4: Web UI実装
Phase 5: 認証・課金統合
Phase 6: UI/UX改善・オンボーディング
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


# カスタムCSS
CUSTOM_CSS = """
<style>
    /* メインコンテナ */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ヘッダースタイル */
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* カード風コンテナ */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #f0f0f0;
    }

    /* クエリ入力エリア */
    .query-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        margin-bottom: 1.5rem;
    }

    /* 例示ボタン */
    .stButton > button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* プライマリボタン */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        border: none;
    }

    /* メトリクスカード */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    [data-testid="stMetricValue"] {
        color: #FF6B6B;
        font-weight: 700;
    }

    /* 結果表示エリア */
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #FF6B6B;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
    }

    /* サイドバー */
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* オンボーディングカード */
    .onboarding-step {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #4ECDC4;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }

    .onboarding-step.completed {
        border-left-color: #28a745;
        background: #f8fff8;
    }

    /* ウェルカムセクション */
    .welcome-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #fff5f5 0%, #f0ffff 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }

    .welcome-section h2 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }

    /* デモカード */
    .demo-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .demo-card:hover {
        transform: translateY(-5px);
        border-color: #FF6B6B;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.2);
    }

    /* フィーチャーアイコン */
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* 進捗インジケータ */
    .progress-indicator {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1.5rem 0;
    }

    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #ddd;
    }

    .progress-dot.active {
        background: #FF6B6B;
    }

    /* レスポンシブ調整 */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }

        .main-header {
            padding: 1rem;
        }

        .main-header h1 {
            font-size: 1.4rem;
        }

        .welcome-section {
            padding: 1.5rem 1rem;
        }
    }

    /* アニメーション */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* ツールチップ改善 */
    .tooltip-text {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
    }
</style>
"""


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
    # オンボーディング状態
    if "onboarding_completed" not in st.session_state:
        st.session_state.onboarding_completed = False
    if "show_tutorial" not in st.session_state:
        st.session_state.show_tutorial = True
    # デモモード
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False


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


def render_welcome_page():
    """ウェルカムページを表示（データ未読み込み時）"""
    # カスタムヘッダー
    st.markdown("""
    <div class="welcome-section animate-fade-in">
        <h2>📊 InsightAnalyzerAI へようこそ</h2>
        <p style="font-size: 1.1rem; color: #6c757d;">
            CSVをアップロードして、自然言語で質問するだけ。<br>
            AIがプロ級のデータ分析を実行します。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # クイックスタートガイド
    st.markdown("### 🚀 3ステップで始める")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div class="feature-icon">📂</div>
            <h4>1. データをアップロード</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">
                CSV, Excel, Parquetに対応<br>
                左のサイドバーから
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div class="feature-icon">💬</div>
            <h4>2. 質問を入力</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">
                日本語で自然に<br>
                「売上の合計は？」
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div class="feature-icon">📈</div>
            <h4>3. 結果を確認</h4>
            <p style="color: #6c757d; font-size: 0.9rem;">
                AIが分析を実行<br>
                チャートも自動生成
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # デモを試すセクション
    st.markdown("### 🎯 今すぐ試す")
    st.info("👉 サンプルデータを使ってすぐに体験できます")

    col_demo1, col_demo2 = st.columns(2)

    with col_demo1:
        if st.button("📊 サンプルデータで始める", type="primary", use_container_width=True):
            load_sample_data()
            st.rerun()

    with col_demo2:
        st.markdown("""
        <p style="padding: 0.5rem; color: #6c757d; font-size: 0.9rem;">
            売上データ（25件）を使って<br>
            分析機能をお試しください
        </p>
        """, unsafe_allow_html=True)

    # 質問例
    st.markdown("### 💡 こんな質問ができます")

    example_questions = [
        {"icon": "🔢", "q": "売上の合計はいくら？", "desc": "数値の集計"},
        {"icon": "📊", "q": "カテゴリ別の売上を教えて", "desc": "グループ集計"},
        {"icon": "📈", "q": "月別の売上推移は？", "desc": "時系列分析"},
        {"icon": "🏆", "q": "最も売れている商品は？", "desc": "ランキング"},
        {"icon": "📋", "q": "データの概要を教えて", "desc": "統計サマリー"},
        {"icon": "🔍", "q": "東京の売上を見せて", "desc": "フィルタリング"},
    ]

    cols = st.columns(3)
    for i, example in enumerate(example_questions):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="info-card" style="padding: 1rem;">
                <span style="font-size: 1.5rem;">{example['icon']}</span>
                <p style="margin: 0.5rem 0 0.3rem 0; font-weight: 500;">"{example['q']}"</p>
                <span class="tooltip-text">{example['desc']}</span>
            </div>
            """, unsafe_allow_html=True)


def load_sample_data():
    """サンプルデータを読み込む"""
    sample_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
    if sample_path.exists():
        # LLM使用可否をプランから判定
        auth_manager = st.session_state.auth_manager
        use_llm = auth_manager.can_use_llm()

        st.session_state.analyzer = InsightAnalyzer(str(sample_path), use_llm=use_llm)
        st.session_state.data_loaded = True
        st.session_state.file_name = "sample_sales.csv"
        st.session_state.demo_mode = True
        return True
    return False


def render_demo_analyses():
    """デモ分析ボタンを表示"""
    st.markdown("### 🎮 ワンクリック分析")
    st.caption("ボタンを押すだけで分析を実行")

    demo_queries = [
        {"label": "📊 データ概要", "query": "データの概要を教えて"},
        {"label": "💰 売上合計", "query": "売上の合計を教えて"},
        {"label": "🏢 地域別売上", "query": "地域別の売上合計を教えて"},
        {"label": "📦 商品別売上", "query": "商品別の売上を教えて"},
        {"label": "🏆 売上トップ5", "query": "売上上位5件を表示して"},
        {"label": "👤 担当者別", "query": "担当者別の売上を教えて"},
    ]

    cols = st.columns(3)
    for i, demo in enumerate(demo_queries):
        with cols[i % 3]:
            if st.button(demo["label"], key=f"demo_{i}", use_container_width=True):
                return demo["query"]

    return None


def render_onboarding_sidebar():
    """サイドバーにオンボーディング進捗を表示"""
    with st.sidebar:
        if not st.session_state.onboarding_completed:
            st.markdown("### 📝 はじめてのガイド")

            steps = [
                {"name": "データを読み込む", "done": st.session_state.data_loaded},
                {"name": "質問を入力する", "done": len(st.session_state.history) > 0},
                {"name": "チャートを生成", "done": any(
                    h.get("chart_generated", False) for h in st.session_state.history
                ) if st.session_state.history else False},
            ]

            completed_count = sum(1 for s in steps if s["done"])

            # 進捗バー
            st.progress(completed_count / len(steps))
            st.caption(f"{completed_count}/{len(steps)} 完了")

            for step in steps:
                icon = "✅" if step["done"] else "⬜"
                st.markdown(f"{icon} {step['name']}")

            if completed_count == len(steps):
                st.success("🎉 すべて完了！")
                st.session_state.onboarding_completed = True
                st.balloons()

            st.divider()


def main():
    """メインアプリケーション"""
    init_session_state()

    # カスタムCSSを適用
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # カスタムヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>📊 InsightAnalyzerAI</h1>
        <p>自然言語でデータを分析するAIアシスタント</p>
    </div>
    """, unsafe_allow_html=True)

    # 認証UI（サイドバー内）
    render_auth_ui()

    # オンボーディング進捗（サイドバー）
    render_onboarding_sidebar()

    # サイドバー設定
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
                        st.session_state.demo_mode = False

        # サンプルデータ使用オプション
        st.divider()
        if st.button("📊 サンプルデータを使用", use_container_width=True):
            if load_sample_data():
                st.success("✅ サンプルデータを読み込みました")
                st.rerun()
            else:
                st.warning("サンプルデータファイルが見つかりません")

        st.divider()

        # オプション
        st.subheader("🎛️ オプション")
        generate_chart = st.checkbox("チャートを生成", value=True)
        explain_result = st.checkbox("AIで結果を説明", value=True)

        # 履歴クリア
        st.divider()
        if st.button("🗑️ 履歴をクリア", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        # データをリセット
        if st.session_state.data_loaded:
            if st.button("🔄 データをリセット", use_container_width=True):
                st.session_state.analyzer = None
                st.session_state.data_loaded = False
                st.session_state.demo_mode = False
                st.session_state.history = []
                st.rerun()

    # メインコンテンツ
    if not st.session_state.data_loaded:
        # ウェルカムページを表示
        render_welcome_page()
        return

    # デモモード表示
    if st.session_state.demo_mode:
        st.info("🎮 **デモモード**: サンプルの売上データを使用しています。自分のデータをアップロードして試すこともできます。")

    # データ情報
    display_data_info()

    # タブで情報を整理
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 クエリ", "📊 データ情報", "📜 履歴", "💰 プラン"])

    with tab1:
        # デモモードならワンクリック分析を表示
        demo_query = None
        if st.session_state.demo_mode:
            demo_query = render_demo_analyses()
            st.divider()

        # クエリ入力セクション
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.subheader("💬 質問を入力")

        # 質問例ボタン
        example_queries = [
            "データの概要を教えて",
            "売上の合計は？",
            "地域別の売上",
            "上位5件を表示",
        ]

        cols = st.columns(len(example_queries))
        selected_example = demo_query  # デモクエリがあれば使用
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
        st.markdown('</div>', unsafe_allow_html=True)

        # 実行ボタン
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            execute_button = st.button(
                "🔍 分析実行",
                type="primary",
                disabled=not question,
                use_container_width=True
            )

        # 分析実行
        if execute_button or (selected_example and question):
            with st.spinner("分析中..."):
                result = process_query(
                    question,
                    generate_chart,
                    explain_result,
                )
                if result:
                    # チャート生成フラグを記録
                    if st.session_state.history:
                        st.session_state.history[-1]["chart_generated"] = generate_chart

                    st.markdown('<div class="result-container animate-fade-in">', unsafe_allow_html=True)
                    display_result(result, show_chart=generate_chart)
                    st.markdown('</div>', unsafe_allow_html=True)

        # 最新の結果を表示（質問がない場合）
        elif st.session_state.history and not question:
            st.divider()
            st.subheader("📋 最新の分析結果")
            display_result(st.session_state.history[-1]["result"], show_chart=generate_chart)

    with tab2:
        display_schema()
        display_insights()
        display_data_preview()

    with tab3:
        if st.session_state.history:
            display_history()
        else:
            st.info("まだクエリ履歴がありません。質問を入力して分析を実行してください。")

    with tab4:
        render_pricing_ui()
        render_billing_status()


if __name__ == "__main__":
    main()
