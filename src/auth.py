"""
InsightAnalyzerAI - 認証システム

Streamlit認証とユーザー管理を提供
Phase 5: 収益化機能（認証）
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import streamlit as st


class PlanType(Enum):
    """料金プラン種別"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class PlanLimits:
    """プラン別の制限"""
    max_file_size_mb: int
    daily_queries: int
    charts_enabled: bool
    api_access: bool
    llm_enabled: bool
    priority_support: bool


# プラン別の制限定義
PLAN_LIMITS: dict[PlanType, PlanLimits] = {
    PlanType.FREE: PlanLimits(
        max_file_size_mb=1,
        daily_queries=10,
        charts_enabled=False,
        api_access=False,
        llm_enabled=False,
        priority_support=False,
    ),
    PlanType.BASIC: PlanLimits(
        max_file_size_mb=50,
        daily_queries=100,
        charts_enabled=True,
        api_access=False,
        llm_enabled=True,
        priority_support=False,
    ),
    PlanType.PRO: PlanLimits(
        max_file_size_mb=500,
        daily_queries=10000,  # 実質無制限
        charts_enabled=True,
        api_access=True,
        llm_enabled=True,
        priority_support=True,
    ),
    PlanType.ENTERPRISE: PlanLimits(
        max_file_size_mb=5000,
        daily_queries=100000,  # 無制限
        charts_enabled=True,
        api_access=True,
        llm_enabled=True,
        priority_support=True,
    ),
}


@dataclass
class User:
    """ユーザー情報"""
    user_id: str
    email: str
    password_hash: str
    plan: PlanType = PlanType.FREE
    created_at: datetime = field(default_factory=datetime.now)
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    daily_query_count: int = 0
    last_query_date: Optional[str] = None


class UsageTracker:
    """使用量追跡"""

    def __init__(self):
        """使用量追跡を初期化"""
        self._init_session_state()

    def _init_session_state(self):
        """セッション状態を初期化"""
        if "usage_query_count" not in st.session_state:
            st.session_state.usage_query_count = 0
        if "usage_last_date" not in st.session_state:
            st.session_state.usage_last_date = datetime.now().strftime("%Y-%m-%d")
        if "usage_file_sizes" not in st.session_state:
            st.session_state.usage_file_sizes = []

    def reset_if_new_day(self):
        """日が変わったらカウントをリセット"""
        today = datetime.now().strftime("%Y-%m-%d")
        if st.session_state.usage_last_date != today:
            st.session_state.usage_query_count = 0
            st.session_state.usage_last_date = today
            st.session_state.usage_file_sizes = []

    def increment_query_count(self):
        """クエリカウントを増加"""
        self.reset_if_new_day()
        st.session_state.usage_query_count += 1

    def get_query_count(self) -> int:
        """現在のクエリカウントを取得"""
        self.reset_if_new_day()
        return st.session_state.usage_query_count

    def add_file_upload(self, size_bytes: int):
        """ファイルアップロードを記録"""
        st.session_state.usage_file_sizes.append(size_bytes)

    def get_total_upload_size(self) -> int:
        """合計アップロードサイズを取得（バイト）"""
        return sum(st.session_state.usage_file_sizes)


class AuthManager:
    """認証マネージャー

    注意: この実装はデモ/開発用です。
    本番環境では、適切なデータベースとセキュリティ対策が必要です。
    """

    def __init__(self):
        """認証マネージャーを初期化"""
        self._init_session_state()
        self.usage_tracker = UsageTracker()

    def _init_session_state(self):
        """セッション状態を初期化"""
        if "auth_user" not in st.session_state:
            st.session_state.auth_user = None
        if "auth_users_db" not in st.session_state:
            # デモ用のメモリ内ユーザーDB
            st.session_state.auth_users_db = {}

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """パスワードをハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hash_obj.hex(), salt

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        """パスワードを検証"""
        # stored_hashは「salt:hash」形式で保存
        try:
            salt, hash_value = stored_hash.split(":")
            computed_hash, _ = AuthManager._hash_password(password, salt)
            return secrets.compare_digest(computed_hash, hash_value)
        except ValueError:
            return False

    def register(self, email: str, password: str) -> tuple[bool, str]:
        """新規ユーザー登録

        Args:
            email: メールアドレス
            password: パスワード

        Returns:
            (成功フラグ, メッセージ)
        """
        # バリデーション
        if not email or "@" not in email:
            return False, "有効なメールアドレスを入力してください"

        if len(password) < 8:
            return False, "パスワードは8文字以上必要です"

        # 既存ユーザーチェック
        if email.lower() in st.session_state.auth_users_db:
            return False, "このメールアドレスは既に登録されています"

        # ユーザー作成
        user_id = secrets.token_urlsafe(16)
        hash_value, salt = self._hash_password(password)
        password_hash = f"{salt}:{hash_value}"

        user = User(
            user_id=user_id,
            email=email.lower(),
            password_hash=password_hash,
        )

        st.session_state.auth_users_db[email.lower()] = user

        return True, "登録完了しました"

    def login(self, email: str, password: str) -> tuple[bool, str]:
        """ログイン

        Args:
            email: メールアドレス
            password: パスワード

        Returns:
            (成功フラグ, メッセージ)
        """
        email_lower = email.lower()

        if email_lower not in st.session_state.auth_users_db:
            return False, "メールアドレスまたはパスワードが正しくありません"

        user = st.session_state.auth_users_db[email_lower]

        if not self._verify_password(password, user.password_hash):
            return False, "メールアドレスまたはパスワードが正しくありません"

        st.session_state.auth_user = user
        return True, "ログインしました"

    def logout(self):
        """ログアウト"""
        st.session_state.auth_user = None

    def get_current_user(self) -> Optional[User]:
        """現在のユーザーを取得"""
        return st.session_state.auth_user

    def is_authenticated(self) -> bool:
        """認証済みかどうか"""
        return st.session_state.auth_user is not None

    def get_plan_limits(self) -> PlanLimits:
        """現在のユーザーのプラン制限を取得"""
        user = self.get_current_user()
        if user is None:
            return PLAN_LIMITS[PlanType.FREE]
        return PLAN_LIMITS[user.plan]

    def can_execute_query(self) -> tuple[bool, str]:
        """クエリ実行可否をチェック

        Returns:
            (実行可能フラグ, メッセージ)
        """
        limits = self.get_plan_limits()
        current_count = self.usage_tracker.get_query_count()

        if current_count >= limits.daily_queries:
            return False, f"本日のクエリ上限（{limits.daily_queries}回）に達しました。プランをアップグレードしてください。"

        return True, ""

    def can_upload_file(self, file_size_bytes: int) -> tuple[bool, str]:
        """ファイルアップロード可否をチェック

        Args:
            file_size_bytes: ファイルサイズ（バイト）

        Returns:
            (アップロード可能フラグ, メッセージ)
        """
        limits = self.get_plan_limits()
        max_bytes = limits.max_file_size_mb * 1024 * 1024

        if file_size_bytes > max_bytes:
            return False, f"ファイルサイズが上限（{limits.max_file_size_mb}MB）を超えています。プランをアップグレードしてください。"

        return True, ""

    def can_use_charts(self) -> bool:
        """チャート機能が使えるか"""
        return self.get_plan_limits().charts_enabled

    def can_use_llm(self) -> bool:
        """LLM機能が使えるか"""
        return self.get_plan_limits().llm_enabled

    def update_plan(self, user_email: str, new_plan: PlanType,
                    stripe_customer_id: Optional[str] = None,
                    stripe_subscription_id: Optional[str] = None) -> bool:
        """ユーザーのプランを更新

        Args:
            user_email: ユーザーのメールアドレス
            new_plan: 新しいプラン
            stripe_customer_id: Stripe顧客ID
            stripe_subscription_id: StripeサブスクリプションID

        Returns:
            成功フラグ
        """
        email_lower = user_email.lower()

        if email_lower not in st.session_state.auth_users_db:
            return False

        user = st.session_state.auth_users_db[email_lower]
        user.plan = new_plan

        if stripe_customer_id:
            user.stripe_customer_id = stripe_customer_id
        if stripe_subscription_id:
            user.stripe_subscription_id = stripe_subscription_id

        # 現在ログイン中のユーザーなら更新
        if st.session_state.auth_user and st.session_state.auth_user.email == email_lower:
            st.session_state.auth_user = user

        return True


def render_auth_ui():
    """認証UIを描画

    Returns:
        認証済みならTrue
    """
    auth_manager = AuthManager()

    if auth_manager.is_authenticated():
        # ログイン済み: ユーザー情報を表示
        user = auth_manager.get_current_user()
        with st.sidebar:
            st.divider()
            st.markdown(f"👤 **{user.email}**")
            st.caption(f"プラン: {user.plan.value.upper()}")

            # 使用量表示
            limits = auth_manager.get_plan_limits()
            query_count = auth_manager.usage_tracker.get_query_count()
            st.progress(
                min(query_count / limits.daily_queries, 1.0),
                text=f"クエリ: {query_count}/{limits.daily_queries}"
            )

            if st.button("ログアウト", key="logout_btn"):
                auth_manager.logout()
                st.rerun()

        return True

    # 未ログイン: ログイン/登録フォーム
    with st.sidebar:
        st.divider()
        st.subheader("🔐 アカウント")

        tab1, tab2 = st.tabs(["ログイン", "新規登録"])

        with tab1:
            email = st.text_input("メールアドレス", key="login_email")
            password = st.text_input("パスワード", type="password", key="login_password")

            if st.button("ログイン", key="login_btn"):
                success, message = auth_manager.login(email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with tab2:
            reg_email = st.text_input("メールアドレス", key="reg_email")
            reg_password = st.text_input("パスワード", type="password", key="reg_password")
            reg_password_confirm = st.text_input("パスワード（確認）", type="password", key="reg_password_confirm")

            if st.button("登録", key="register_btn"):
                if reg_password != reg_password_confirm:
                    st.error("パスワードが一致しません")
                else:
                    success, message = auth_manager.register(reg_email, reg_password)
                    if success:
                        st.success(message)
                        # 自動ログイン
                        auth_manager.login(reg_email, reg_password)
                        st.rerun()
                    else:
                        st.error(message)

        st.divider()
        st.caption("💡 アカウントなしでも無料プランで利用できます")

    return False


def require_auth(func):
    """認証必須デコレータ

    Usage:
        @require_auth
        def my_feature():
            ...
    """
    def wrapper(*args, **kwargs):
        auth_manager = AuthManager()
        if not auth_manager.is_authenticated():
            st.warning("この機能を使用するにはログインが必要です")
            return None
        return func(*args, **kwargs)
    return wrapper


def require_plan(min_plan: PlanType):
    """最低プラン要求デコレータ

    Usage:
        @require_plan(PlanType.BASIC)
        def my_premium_feature():
            ...
    """
    plan_order = [PlanType.FREE, PlanType.BASIC, PlanType.PRO, PlanType.ENTERPRISE]

    def decorator(func):
        def wrapper(*args, **kwargs):
            auth_manager = AuthManager()
            user = auth_manager.get_current_user()

            if user is None:
                current_plan = PlanType.FREE
            else:
                current_plan = user.plan

            current_idx = plan_order.index(current_plan)
            required_idx = plan_order.index(min_plan)

            if current_idx < required_idx:
                st.warning(f"この機能を使用するには{min_plan.value.upper()}プラン以上が必要です")
                return None

            return func(*args, **kwargs)
        return wrapper
    return decorator
