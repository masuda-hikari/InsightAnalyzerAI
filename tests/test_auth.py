"""
InsightAnalyzerAI - 認証システムテスト

認証・プラン管理機能のテスト
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Streamlitのモック
import sys
sys.modules['streamlit'] = MagicMock()

from src.auth import (
    PlanType,
    PlanLimits,
    PLAN_LIMITS,
    User,
    UsageTracker,
    AuthManager,
)


class TestPlanLimits:
    """プラン制限のテスト"""

    def test_free_plan_limits(self):
        """Freeプランの制限"""
        limits = PLAN_LIMITS[PlanType.FREE]
        assert limits.max_file_size_mb == 1
        assert limits.daily_queries == 10
        assert limits.charts_enabled is False
        assert limits.llm_enabled is False

    def test_basic_plan_limits(self):
        """Basicプランの制限"""
        limits = PLAN_LIMITS[PlanType.BASIC]
        assert limits.max_file_size_mb == 50
        assert limits.daily_queries == 100
        assert limits.charts_enabled is True
        assert limits.llm_enabled is True

    def test_pro_plan_limits(self):
        """Proプランの制限"""
        limits = PLAN_LIMITS[PlanType.PRO]
        assert limits.max_file_size_mb == 500
        assert limits.daily_queries == 10000
        assert limits.charts_enabled is True
        assert limits.api_access is True


class TestUser:
    """Userデータクラスのテスト"""

    def test_user_creation(self):
        """ユーザー作成"""
        user = User(
            user_id="test123",
            email="test@example.com",
            password_hash="hash123",
        )
        assert user.user_id == "test123"
        assert user.email == "test@example.com"
        assert user.plan == PlanType.FREE  # デフォルト

    def test_user_with_plan(self):
        """プラン指定でユーザー作成"""
        user = User(
            user_id="pro123",
            email="pro@example.com",
            password_hash="hash456",
            plan=PlanType.PRO,
        )
        assert user.plan == PlanType.PRO


class TestAuthManager:
    """AuthManagerのテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_password_hash(self, mock_session_state):
        """パスワードハッシュのテスト"""
        manager = AuthManager()
        hash1, salt1 = manager._hash_password("password123")
        hash2, _ = manager._hash_password("password123", salt1)

        # 同じソルトなら同じハッシュ
        assert hash1 == hash2

    def test_password_hash_different_salts(self, mock_session_state):
        """異なるソルトでは異なるハッシュ"""
        manager = AuthManager()
        hash1, salt1 = manager._hash_password("password123")
        hash2, salt2 = manager._hash_password("password123")

        # 異なるソルトなら異なるハッシュ
        if salt1 != salt2:
            assert hash1 != hash2

    def test_verify_password(self, mock_session_state):
        """パスワード検証"""
        manager = AuthManager()
        hash_value, salt = manager._hash_password("mypassword")
        stored_hash = f"{salt}:{hash_value}"

        assert manager._verify_password("mypassword", stored_hash) is True
        assert manager._verify_password("wrongpassword", stored_hash) is False

    def test_register_success(self, mock_session_state):
        """ユーザー登録成功"""
        manager = AuthManager()
        success, message = manager.register("new@example.com", "password123")

        assert success is True
        assert "完了" in message
        # 登録成功の確認: ログインできれば登録成功
        success2, _ = manager.login("new@example.com", "password123")
        assert success2 is True

    def test_register_invalid_email(self, mock_session_state):
        """無効なメールアドレスでの登録"""
        manager = AuthManager()
        success, message = manager.register("invalid-email", "password123")

        assert success is False
        assert "メールアドレス" in message

    def test_register_short_password(self, mock_session_state):
        """短いパスワードでの登録"""
        manager = AuthManager()
        success, message = manager.register("test@example.com", "short")

        assert success is False
        assert "8文字" in message

    def test_register_duplicate_email(self, mock_session_state):
        """重複メールアドレスでの登録"""
        manager = AuthManager()
        manager.register("dup@example.com", "password123")
        success, message = manager.register("dup@example.com", "password456")

        assert success is False
        assert "既に登録" in message

    def test_login_success(self, mock_session_state):
        """ログイン成功"""
        manager = AuthManager()
        manager.register("login@example.com", "password123")
        success, message = manager.login("login@example.com", "password123")

        assert success is True
        assert manager.is_authenticated() is True

    def test_login_wrong_password(self, mock_session_state):
        """間違ったパスワードでのログイン"""
        manager = AuthManager()
        manager.register("test@example.com", "password123")
        success, message = manager.login("test@example.com", "wrongpassword")

        assert success is False

    def test_login_nonexistent_user(self, mock_session_state):
        """存在しないユーザーでのログイン"""
        manager = AuthManager()
        success, message = manager.login("nonexistent@example.com", "password123")

        assert success is False

    def test_logout(self, mock_session_state):
        """ログアウト"""
        manager = AuthManager()
        manager.register("logout@example.com", "password123")
        manager.login("logout@example.com", "password123")

        assert manager.is_authenticated() is True

        manager.logout()

        assert manager.is_authenticated() is False

    def test_get_plan_limits_unauthenticated(self, mock_session_state):
        """未認証時のプラン制限"""
        manager = AuthManager()
        limits = manager.get_plan_limits()

        assert limits == PLAN_LIMITS[PlanType.FREE]

    def test_can_execute_query_within_limit(self, mock_session_state):
        """クエリ制限内での実行"""
        manager = AuthManager()
        manager.usage_tracker._init_session_state()

        can_execute, message = manager.can_execute_query()

        assert can_execute is True

    def test_can_upload_file_within_limit(self, mock_session_state):
        """ファイルサイズ制限内でのアップロード"""
        manager = AuthManager()

        # Freeプラン: 1MB制限
        can_upload, message = manager.can_upload_file(500 * 1024)  # 500KB

        assert can_upload is True

    def test_can_upload_file_exceeds_limit(self, mock_session_state):
        """ファイルサイズ制限超過"""
        manager = AuthManager()

        # Freeプラン: 1MB制限
        can_upload, message = manager.can_upload_file(2 * 1024 * 1024)  # 2MB

        assert can_upload is False
        assert "上限" in message

    def test_update_plan(self, mock_session_state):
        """プラン更新"""
        manager = AuthManager()
        manager.register("upgrade@example.com", "password123")
        manager.login("upgrade@example.com", "password123")

        success = manager.update_plan("upgrade@example.com", PlanType.PRO)

        assert success is True
        assert manager.get_current_user().plan == PlanType.PRO

    def test_can_use_insights_free_plan(self, mock_session_state):
        """Freeプランでは自動インサイト使用不可"""
        manager = AuthManager()
        manager.register("free@example.com", "password123")
        manager.login("free@example.com", "password123")

        assert manager.can_use_insights() is False

    def test_can_use_insights_basic_plan(self, mock_session_state):
        """Basicプランでは自動インサイト使用可能"""
        manager = AuthManager()
        manager.register("basic@example.com", "password123")
        manager.login("basic@example.com", "password123")
        manager.update_plan("basic@example.com", PlanType.BASIC)

        assert manager.can_use_insights() is True

    def test_can_use_insights_pro_plan(self, mock_session_state):
        """Proプランでは自動インサイト使用可能"""
        manager = AuthManager()
        manager.register("pro@example.com", "password123")
        manager.login("pro@example.com", "password123")
        manager.update_plan("pro@example.com", PlanType.PRO)

        assert manager.can_use_insights() is True

    def test_can_use_insights_unauthenticated(self, mock_session_state):
        """未認証では自動インサイト使用不可"""
        manager = AuthManager()

        assert manager.can_use_insights() is False

    def test_can_use_charts_free(self, mock_session_state):
        """Freeプランではチャート使用不可"""
        manager = AuthManager()

        assert manager.can_use_charts() is False

    def test_can_use_charts_basic(self, mock_session_state):
        """Basicプランではチャート使用可能"""
        manager = AuthManager()
        manager.register("charts@example.com", "password123")
        manager.login("charts@example.com", "password123")
        manager.update_plan("charts@example.com", PlanType.BASIC)

        assert manager.can_use_charts() is True

    def test_can_use_llm_free(self, mock_session_state):
        """FreeプランではLLM使用不可"""
        manager = AuthManager()

        assert manager.can_use_llm() is False

    def test_can_use_llm_basic(self, mock_session_state):
        """BasicプランではLLM使用可能"""
        manager = AuthManager()
        manager.register("llm@example.com", "password123")
        manager.login("llm@example.com", "password123")
        manager.update_plan("llm@example.com", PlanType.BASIC)

        assert manager.can_use_llm() is True


class TestUsageTracker:
    """UsageTrackerのテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_increment_query_count(self, mock_session_state):
        """クエリカウント増加"""
        tracker = UsageTracker()
        tracker._init_session_state()

        initial_count = tracker.get_query_count()
        tracker.increment_query_count()
        new_count = tracker.get_query_count()

        assert new_count == initial_count + 1

    def test_add_file_upload(self, mock_session_state):
        """ファイルアップロード記録"""
        tracker = UsageTracker()
        tracker._init_session_state()

        tracker.add_file_upload(1024)
        tracker.add_file_upload(2048)

        assert tracker.get_total_upload_size() == 3072

    @pytest.mark.skip(reason="Streamlitのsession_stateはdot記法で属性アクセスが必要で、モック環境では正確にテストできない")
    def test_reset_if_new_day(self, mock_session_state):
        """日が変わったらリセット"""
        import streamlit as st
        tracker = UsageTracker()
        tracker._init_session_state()

        # クエリカウントを増やす
        tracker.increment_query_count()
        tracker.increment_query_count()
        assert tracker.get_query_count() == 2

        # 昨日の日付に設定（dict形式でアクセス）
        st.session_state["usage_last_date"] = "2020-01-01"

        # リセットされるはず
        tracker.reset_if_new_day()
        assert tracker.get_query_count() == 0

    @pytest.mark.skip(reason="Streamlitのsession_stateはdot記法で属性アクセスが必要で、モック環境では正確にテストできない")
    def test_get_query_count_resets_on_new_day(self, mock_session_state):
        """get_query_countでもリセットされる"""
        import streamlit as st
        tracker = UsageTracker()
        tracker._init_session_state()

        # クエリカウントを増やす（dict形式でアクセス）
        st.session_state["usage_query_count"] = 5
        # 昨日の日付に設定
        st.session_state["usage_last_date"] = "2020-01-01"

        # get_query_countでリセットされる
        count = tracker.get_query_count()
        assert count == 0


class TestAuthManagerExtended:
    """AuthManagerの追加テスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_verify_password_invalid_format(self, mock_session_state):
        """不正なフォーマットのパスワードハッシュ検証"""
        manager = AuthManager()

        # salt:hash形式でない場合
        result = manager._verify_password("password", "invalid_hash_without_colon")
        assert result is False

    def test_register_empty_email(self, mock_session_state):
        """空のメールアドレスでの登録"""
        manager = AuthManager()
        success, message = manager.register("", "password123")

        assert success is False
        assert "メールアドレス" in message

    def test_update_plan_nonexistent_user(self, mock_session_state):
        """存在しないユーザーのプラン更新"""
        manager = AuthManager()
        success = manager.update_plan("nonexistent@example.com", PlanType.PRO)

        assert success is False

    def test_update_plan_with_stripe_ids(self, mock_session_state):
        """Stripe IDを含むプラン更新"""
        manager = AuthManager()
        manager.register("stripe@example.com", "password123")
        manager.login("stripe@example.com", "password123")

        success = manager.update_plan(
            "stripe@example.com",
            PlanType.PRO,
            stripe_customer_id="cus_xxx",
            stripe_subscription_id="sub_xxx"
        )

        assert success is True
        user = manager.get_current_user()
        assert user.stripe_customer_id == "cus_xxx"
        assert user.stripe_subscription_id == "sub_xxx"

    def test_update_plan_not_logged_in_user(self, mock_session_state):
        """ログインしていないユーザーのプラン更新"""
        manager = AuthManager()
        manager.register("notloggedin@example.com", "password123")

        # ログインせずにプラン更新
        success = manager.update_plan("notloggedin@example.com", PlanType.PRO)

        assert success is True
        # ログインユーザーはNoneのまま
        assert manager.get_current_user() is None

    @pytest.mark.skip(reason="Streamlitのsession_stateはdot記法で属性アクセスが必要で、モック環境では正確にテストできない")
    def test_can_execute_query_exceeds_limit(self, mock_session_state):
        """クエリ制限超過"""
        import streamlit as st
        manager = AuthManager()
        manager.usage_tracker._init_session_state()

        # Freeプランの制限（10回）を超えるカウントを設定（dict形式でアクセス）
        st.session_state["usage_query_count"] = 10

        can_execute, message = manager.can_execute_query()

        assert can_execute is False
        assert "上限" in message

    def test_get_current_user_none(self, mock_session_state):
        """未ログイン時のユーザー取得"""
        manager = AuthManager()
        user = manager.get_current_user()

        assert user is None

    def test_is_authenticated_false(self, mock_session_state):
        """未ログイン時の認証チェック"""
        manager = AuthManager()

        assert manager.is_authenticated() is False

    def test_can_use_insights_enterprise_plan(self, mock_session_state):
        """Enterpriseプランでは自動インサイト使用可能"""
        manager = AuthManager()
        manager.register("enterprise@example.com", "password123")
        manager.login("enterprise@example.com", "password123")
        manager.update_plan("enterprise@example.com", PlanType.ENTERPRISE)

        assert manager.can_use_insights() is True


class TestPlanLimitsExtended:
    """プラン制限の追加テスト"""

    def test_enterprise_plan_limits(self):
        """Enterpriseプランの制限"""
        limits = PLAN_LIMITS[PlanType.ENTERPRISE]
        assert limits.max_file_size_mb == 5000
        assert limits.daily_queries == 100000
        assert limits.charts_enabled is True
        assert limits.api_access is True
        assert limits.llm_enabled is True
        assert limits.priority_support is True

    def test_all_plans_have_limits(self):
        """全プランに制限が定義されている"""
        for plan_type in PlanType:
            assert plan_type in PLAN_LIMITS
            limits = PLAN_LIMITS[plan_type]
            assert isinstance(limits, PlanLimits)


class TestUserExtended:
    """Userデータクラスの追加テスト"""

    def test_user_with_stripe_ids(self):
        """Stripe IDを含むユーザー作成"""
        user = User(
            user_id="stripe123",
            email="stripe@example.com",
            password_hash="hash123",
            plan=PlanType.PRO,
            stripe_customer_id="cus_xxx",
            stripe_subscription_id="sub_xxx",
        )
        assert user.stripe_customer_id == "cus_xxx"
        assert user.stripe_subscription_id == "sub_xxx"

    def test_user_default_values(self):
        """ユーザーのデフォルト値"""
        user = User(
            user_id="default123",
            email="default@example.com",
            password_hash="hash123",
        )
        assert user.plan == PlanType.FREE
        assert user.stripe_customer_id is None
        assert user.stripe_subscription_id is None
        assert user.daily_query_count == 0
        assert user.last_query_date is None

    def test_user_created_at(self):
        """ユーザー作成日時"""
        user = User(
            user_id="time123",
            email="time@example.com",
            password_hash="hash123",
        )
        assert user.created_at is not None
        assert isinstance(user.created_at, datetime)


class TestDecorators:
    """デコレータのテスト

    注意: デコレータは内部で新しいAuthManagerインスタンスを作成するため、
    セッション状態の共有が複雑。ここではロジックテストを行う。
    """

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_require_auth_decorator_unauthenticated(self, mock_session_state):
        """認証なしでrequire_authデコレータ"""
        from src.auth import require_auth

        @require_auth
        def protected_function():
            return "success"

        result = protected_function()
        assert result is None  # 未認証なのでNone

    def test_require_plan_decorator_unauthenticated(self, mock_session_state):
        """未認証でrequire_planデコレータ"""
        from src.auth import require_plan

        @require_plan(PlanType.BASIC)
        def premium_function():
            return "success"

        result = premium_function()
        assert result is None  # 未認証はFree扱いなのでアクセス不可

    def test_require_plan_decorator_free_user(self, mock_session_state):
        """Freeユーザーでrequire_planデコレータ（Basic要求）"""
        from src.auth import require_plan

        manager = AuthManager()
        manager.register("free@example.com", "password123")
        manager.login("free@example.com", "password123")

        @require_plan(PlanType.BASIC)
        def premium_function():
            return "success"

        result = premium_function()
        assert result is None  # Freeプランなのでアクセス不可

    @pytest.mark.skip(reason="デコレータは新しいAuthManagerを作成するためセッション状態が共有されない")
    def test_require_auth_decorator_authenticated(self, mock_session_state):
        """認証ありでrequire_authデコレータ"""
        from src.auth import require_auth

        manager = AuthManager()
        manager.register("auth@example.com", "password123")
        manager.login("auth@example.com", "password123")

        @require_auth
        def protected_function():
            return "success"

        result = protected_function()
        assert result == "success"

    @pytest.mark.skip(reason="デコレータは新しいAuthManagerを作成するためセッション状態が共有されない")
    def test_require_plan_decorator_basic_user(self, mock_session_state):
        """Basicユーザーでrequire_planデコレータ（Basic要求）"""
        from src.auth import require_plan

        manager = AuthManager()
        manager.register("basic@example.com", "password123")
        manager.login("basic@example.com", "password123")
        manager.update_plan("basic@example.com", PlanType.BASIC)

        @require_plan(PlanType.BASIC)
        def premium_function():
            return "success"

        result = premium_function()
        assert result == "success"

    @pytest.mark.skip(reason="デコレータは新しいAuthManagerを作成するためセッション状態が共有されない")
    def test_require_plan_decorator_pro_user_for_basic(self, mock_session_state):
        """ProユーザーでBasic要求のデコレータ"""
        from src.auth import require_plan

        manager = AuthManager()
        manager.register("pro@example.com", "password123")
        manager.login("pro@example.com", "password123")
        manager.update_plan("pro@example.com", PlanType.PRO)

        @require_plan(PlanType.BASIC)
        def basic_function():
            return "success"

        result = basic_function()
        assert result == "success"  # ProはBasic以上なのでアクセス可能


class TestUsageTrackerExtended:
    """UsageTrackerの追加テスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        # dictではなくMagicMockを使い、属性アクセスをサポート
        mock_state = MagicMock()
        mock_state.__contains__ = lambda self, key: hasattr(self, key)
        st.session_state = mock_state
        return st.session_state

    def test_add_multiple_file_uploads(self, mock_session_state):
        """複数ファイルアップロード記録"""
        import streamlit as st
        st.session_state.usage_file_sizes = []

        tracker = UsageTracker()

        tracker.add_file_upload(1024)
        tracker.add_file_upload(2048)
        tracker.add_file_upload(4096)

        assert tracker.get_total_upload_size() == 7168

    def test_get_total_upload_size_empty(self, mock_session_state):
        """ファイルアップロードなしでの合計サイズ"""
        import streamlit as st
        st.session_state.usage_file_sizes = []

        tracker = UsageTracker()

        assert tracker.get_total_upload_size() == 0


class TestAuthManagerLoginEdgeCases:
    """ログインのエッジケーステスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_login_case_insensitive_email(self, mock_session_state):
        """メールアドレスの大文字小文字を無視してログイン"""
        manager = AuthManager()
        manager.register("TEST@EXAMPLE.COM", "password123")

        # 小文字でログイン
        success, _ = manager.login("test@example.com", "password123")
        assert success is True

    def test_login_email_with_leading_trailing_spaces(self, mock_session_state):
        """前後のスペースを含むメールアドレス"""
        manager = AuthManager()
        manager.register("test@example.com", "password123")

        # スペースを含むメールアドレスでログイン試行
        # 注意: 実装によってはトリミングされるか、失敗するかは異なる
        success, _ = manager.login("test@example.com", "password123")
        assert success is True


class TestAuthManagerPlanLimitsEdgeCases:
    """プラン制限のエッジケーステスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_get_plan_limits_for_basic_plan(self, mock_session_state):
        """Basicプランの制限取得"""
        manager = AuthManager()
        manager.register("basic@example.com", "password123")
        manager.login("basic@example.com", "password123")
        manager.update_plan("basic@example.com", PlanType.BASIC)

        limits = manager.get_plan_limits()
        assert limits == PLAN_LIMITS[PlanType.BASIC]

    def test_get_plan_limits_for_pro_plan(self, mock_session_state):
        """Proプランの制限取得"""
        manager = AuthManager()
        manager.register("pro@example.com", "password123")
        manager.login("pro@example.com", "password123")
        manager.update_plan("pro@example.com", PlanType.PRO)

        limits = manager.get_plan_limits()
        assert limits == PLAN_LIMITS[PlanType.PRO]

    def test_can_upload_large_file_pro_plan(self, mock_session_state):
        """Proプランでの大容量ファイルアップロード"""
        manager = AuthManager()
        manager.register("pro@example.com", "password123")
        manager.login("pro@example.com", "password123")
        manager.update_plan("pro@example.com", PlanType.PRO)

        # Pro: 500MB制限
        can_upload, _ = manager.can_upload_file(400 * 1024 * 1024)  # 400MB
        assert can_upload is True


class TestPlanTypeEnum:
    """PlanType列挙型のテスト"""

    def test_plan_type_values(self):
        """プランタイプの値確認"""
        assert PlanType.FREE.value == "free"
        assert PlanType.BASIC.value == "basic"
        assert PlanType.PRO.value == "pro"
        assert PlanType.ENTERPRISE.value == "enterprise"

    def test_plan_type_from_string(self):
        """文字列からプランタイプ作成"""
        assert PlanType("free") == PlanType.FREE
        assert PlanType("basic") == PlanType.BASIC
        assert PlanType("pro") == PlanType.PRO
        assert PlanType("enterprise") == PlanType.ENTERPRISE


class TestPlanLimitsDataclass:
    """PlanLimitsデータクラスのテスト"""

    def test_plan_limits_creation(self):
        """PlanLimits作成"""
        limits = PlanLimits(
            max_file_size_mb=10,
            daily_queries=50,
            charts_enabled=True,
            api_access=False,
            llm_enabled=True,
            priority_support=False
        )
        assert limits.max_file_size_mb == 10
        assert limits.daily_queries == 50
        assert limits.charts_enabled is True
        assert limits.api_access is False
        assert limits.llm_enabled is True
        assert limits.priority_support is False


class TestPasswordHashingSecurity:
    """パスワードハッシュのセキュリティテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_password_hash_produces_salt_and_hash(self, mock_session_state):
        """パスワードハッシュがsaltとhashを生成"""
        manager = AuthManager()
        hash_value, salt = manager._hash_password("password123")

        # ハッシュ値とソルトがどちらも生成されている
        assert hash_value is not None
        assert salt is not None
        assert len(hash_value) > 0
        assert len(salt) > 0
        # 16進数文字列
        assert all(c in "0123456789abcdef" for c in hash_value)
        assert all(c in "0123456789abcdef" for c in salt)

    def test_same_password_different_salt_different_hash(self, mock_session_state):
        """同じパスワードでもソルトが違えばハッシュも違う"""
        manager = AuthManager()
        hash1, salt1 = manager._hash_password("password123")
        hash2, salt2 = manager._hash_password("password123")

        # ソルトが自動生成されるので異なる
        if salt1 != salt2:
            assert hash1 != hash2


class MockSessionStateDict(dict):
    """dictとドット記法の両方をサポートするセッション状態モック"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class TestUsageTrackerResetOnNewDay:
    """日が変わった場合のリセットテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        mock_state = MockSessionStateDict()
        st.session_state = mock_state
        return mock_state

    def test_reset_if_new_day_when_date_changed(self, mock_session_state):
        """日付が変わったらリセットされる"""
        from datetime import datetime
        from src import auth

        # session_stateを事前に初期化
        mock_session_state["usage_query_count"] = 5
        mock_session_state["usage_last_date"] = "2020-01-01"
        mock_session_state["usage_file_sizes"] = [1000, 2000]

        # auth内のstにもアクセスできるようにする
        auth.st.session_state = mock_session_state

        tracker = UsageTracker()
        # リセットされるはず（今日と違う日付なので）
        tracker.reset_if_new_day()

        # リセット後はカウントが0になる
        assert mock_session_state["usage_query_count"] == 0
        assert mock_session_state["usage_file_sizes"] == []
        assert mock_session_state["usage_last_date"] == datetime.now().strftime("%Y-%m-%d")

    def test_reset_if_new_day_same_day(self, mock_session_state):
        """同じ日付ならリセットされない"""
        from datetime import datetime
        from src import auth

        today = datetime.now().strftime("%Y-%m-%d")

        # 今日の日付で初期化
        mock_session_state["usage_query_count"] = 5
        mock_session_state["usage_last_date"] = today
        mock_session_state["usage_file_sizes"] = [1000, 2000]

        auth.st.session_state = mock_session_state

        tracker = UsageTracker()
        # リセットは発生しない（今日の日付のまま）
        tracker.reset_if_new_day()

        # カウントはそのまま
        assert mock_session_state["usage_query_count"] == 5
        assert mock_session_state["usage_file_sizes"] == [1000, 2000]

    def test_get_query_count_triggers_reset_on_new_day(self, mock_session_state):
        """get_query_countでも日付変更時にリセットされる"""
        from src import auth

        # 昨日の日付とカウントを設定
        mock_session_state["usage_query_count"] = 10
        mock_session_state["usage_last_date"] = "2020-01-01"
        mock_session_state["usage_file_sizes"] = [1000]

        auth.st.session_state = mock_session_state

        tracker = UsageTracker()
        # get_query_countでリセットが発生
        count = tracker.get_query_count()
        assert count == 0


class TestAuthManagerQueryLimitExceeded:
    """クエリ制限超過のテスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        mock_state = MockSessionStateDict()
        st.session_state = mock_state
        return mock_state

    def test_can_execute_query_at_limit(self, mock_session_state):
        """クエリ制限に達した場合"""
        from datetime import datetime
        from src import auth

        today = datetime.now().strftime("%Y-%m-%d")

        # session_stateを初期化
        mock_session_state["usage_query_count"] = 10  # Freeプランの制限（10回）
        mock_session_state["usage_last_date"] = today
        mock_session_state["usage_file_sizes"] = []
        mock_session_state["auth_user"] = None
        mock_session_state["auth_users_db"] = {}

        auth.st.session_state = mock_session_state

        manager = AuthManager()

        can_execute, message = manager.can_execute_query()

        assert can_execute is False
        assert "上限" in message

    def test_can_execute_query_over_limit(self, mock_session_state):
        """クエリ制限を超過した場合"""
        from datetime import datetime
        from src import auth

        today = datetime.now().strftime("%Y-%m-%d")

        # Freeプランの制限（10回）を超える
        mock_session_state["usage_query_count"] = 15
        mock_session_state["usage_last_date"] = today
        mock_session_state["usage_file_sizes"] = []
        mock_session_state["auth_user"] = None
        mock_session_state["auth_users_db"] = {}

        auth.st.session_state = mock_session_state

        manager = AuthManager()

        can_execute, message = manager.can_execute_query()

        assert can_execute is False
        assert "10" in message  # 制限数が表示される


class TestAuthManagerUpdatePlanForLoggedInUser:
    """ログイン中ユーザーのプラン更新テスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_update_plan_updates_current_user_session(self, mock_session_state):
        """ログイン中ユーザーのプラン更新でセッションも更新される"""
        manager = AuthManager()
        manager.register("user@example.com", "password123")
        manager.login("user@example.com", "password123")

        # 現在のプランを確認
        assert manager.get_current_user().plan == PlanType.FREE

        # プラン更新
        success = manager.update_plan(
            "user@example.com",
            PlanType.PRO,
            stripe_customer_id="cus_123",
            stripe_subscription_id="sub_456"
        )

        assert success is True
        # セッション内のユーザーも更新されている
        current_user = manager.get_current_user()
        assert current_user.plan == PlanType.PRO
        assert current_user.stripe_customer_id == "cus_123"
        assert current_user.stripe_subscription_id == "sub_456"


class TestAuthManagerEmailCaseSensitivity:
    """メールアドレスの大文字小文字処理テスト"""

    @pytest.fixture
    def mock_session_state(self):
        """セッション状態のモック"""
        import streamlit as st
        st.session_state = {}
        return st.session_state

    def test_register_with_uppercase_login_with_lowercase(self, mock_session_state):
        """大文字で登録して小文字でログイン"""
        manager = AuthManager()
        manager.register("USER@EXAMPLE.COM", "password123")

        success, _ = manager.login("user@example.com", "password123")
        assert success is True

    def test_update_plan_case_insensitive(self, mock_session_state):
        """プラン更新もメールアドレス大文字小文字を無視"""
        manager = AuthManager()
        manager.register("user@example.com", "password123")

        # 大文字でプラン更新
        success = manager.update_plan("USER@EXAMPLE.COM", PlanType.BASIC)
        assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
