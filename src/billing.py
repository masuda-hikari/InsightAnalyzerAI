"""
InsightAnalyzerAI - 課金システム

Stripe統合による課金管理
Phase 5: 収益化機能（課金）
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import streamlit as st

from src.auth import AuthManager, PlanType

# Stripe SDK（オプション）
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None


# 料金プラン定義（Stripe Price ID）
@dataclass
class PriceConfig:
    """価格設定"""
    plan: PlanType
    price_jpy: int  # 月額（円）
    stripe_price_id: Optional[str]  # Stripe Price ID
    features: list[str]


# 価格設定
PRICE_CONFIGS: dict[PlanType, PriceConfig] = {
    PlanType.FREE: PriceConfig(
        plan=PlanType.FREE,
        price_jpy=0,
        stripe_price_id=None,
        features=[
            "1MB/ファイル",
            "10クエリ/日",
            "基本統計のみ",
        ],
    ),
    PlanType.BASIC: PriceConfig(
        plan=PlanType.BASIC,
        price_jpy=2980,
        stripe_price_id=os.getenv("STRIPE_PRICE_BASIC"),
        features=[
            "50MB/ファイル",
            "100クエリ/日",
            "チャート生成",
            "LLM分析",
        ],
    ),
    PlanType.PRO: PriceConfig(
        plan=PlanType.PRO,
        price_jpy=9800,
        stripe_price_id=os.getenv("STRIPE_PRICE_PRO"),
        features=[
            "500MB/ファイル",
            "無制限クエリ",
            "全機能解放",
            "API連携",
            "優先サポート",
        ],
    ),
    PlanType.ENTERPRISE: PriceConfig(
        plan=PlanType.ENTERPRISE,
        price_jpy=0,  # 要見積
        stripe_price_id=None,  # カスタム
        features=[
            "無制限",
            "オンプレミス対応",
            "カスタム機能",
            "専任サポート",
        ],
    ),
}


class BillingManager:
    """課金マネージャー"""

    def __init__(self):
        """課金マネージャーを初期化"""
        self.auth_manager = AuthManager()
        self._init_stripe()

    def _init_stripe(self):
        """Stripeを初期化"""
        self.stripe_available = False

        if not STRIPE_AVAILABLE:
            return

        # Stripe APIキーを取得
        stripe_key = None

        # 環境変数から取得
        stripe_key = os.getenv("STRIPE_SECRET_KEY")

        # Streamlit secretsから取得
        if stripe_key is None:
            try:
                stripe_key = st.secrets.get("STRIPE_SECRET_KEY")
            except Exception:
                pass

        if stripe_key:
            stripe.api_key = stripe_key
            self.stripe_available = True

    def create_checkout_session(self, plan: PlanType) -> Optional[str]:
        """Stripeチェックアウトセッションを作成

        Args:
            plan: 購入するプラン

        Returns:
            チェックアウトURL（Stripe利用不可の場合None）
        """
        if not self.stripe_available:
            return None

        price_config = PRICE_CONFIGS.get(plan)
        if not price_config or not price_config.stripe_price_id:
            return None

        user = self.auth_manager.get_current_user()
        if not user:
            return None

        try:
            # 成功/キャンセルURLを取得
            success_url = os.getenv("STRIPE_SUCCESS_URL", "http://localhost:8501?payment=success")
            cancel_url = os.getenv("STRIPE_CANCEL_URL", "http://localhost:8501?payment=cancel")

            # チェックアウトセッション作成
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": price_config.stripe_price_id,
                    "quantity": 1,
                }],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=user.email,
                metadata={
                    "user_id": user.user_id,
                    "plan": plan.value,
                },
            )

            return session.url

        except Exception as e:
            st.error(f"チェックアウトセッション作成エラー: {str(e)}")
            return None

    def get_subscription_status(self) -> Optional[dict]:
        """現在のサブスクリプション状態を取得

        Returns:
            サブスクリプション情報（なければNone）
        """
        if not self.stripe_available:
            return None

        user = self.auth_manager.get_current_user()
        if not user or not user.stripe_subscription_id:
            return None

        try:
            subscription = stripe.Subscription.retrieve(user.stripe_subscription_id)
            return {
                "status": subscription.status,
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "cancel_at_period_end": subscription.cancel_at_period_end,
            }
        except Exception:
            return None

    def cancel_subscription(self) -> bool:
        """サブスクリプションをキャンセル

        Returns:
            成功フラグ
        """
        if not self.stripe_available:
            return False

        user = self.auth_manager.get_current_user()
        if not user or not user.stripe_subscription_id:
            return False

        try:
            # 期間終了時にキャンセル
            stripe.Subscription.modify(
                user.stripe_subscription_id,
                cancel_at_period_end=True,
            )
            return True
        except Exception as e:
            st.error(f"キャンセルエラー: {str(e)}")
            return False

    def handle_webhook(self, payload: str, sig_header: str) -> bool:
        """Stripeウェブフックを処理

        Args:
            payload: ウェブフックペイロード
            sig_header: Stripe署名ヘッダー

        Returns:
            処理成功フラグ
        """
        if not self.stripe_available:
            return False

        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        if not webhook_secret:
            return False

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
        except Exception:
            return False

        # イベントタイプ別処理
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            self._handle_checkout_completed(session)

        elif event["type"] == "customer.subscription.updated":
            subscription = event["data"]["object"]
            self._handle_subscription_updated(subscription)

        elif event["type"] == "customer.subscription.deleted":
            subscription = event["data"]["object"]
            self._handle_subscription_deleted(subscription)

        return True

    def _handle_checkout_completed(self, session: dict):
        """チェックアウト完了を処理"""
        user_id = session.get("metadata", {}).get("user_id")
        plan_str = session.get("metadata", {}).get("plan")
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")

        if not user_id or not plan_str:
            return

        plan = PlanType(plan_str)

        # ユーザーDBを検索してプラン更新
        # 注意: 本番環境では適切なDBアクセスが必要
        for email, user in st.session_state.get("auth_users_db", {}).items():
            if user.user_id == user_id:
                self.auth_manager.update_plan(
                    email,
                    plan,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                )
                break

    def _handle_subscription_updated(self, subscription: dict):
        """サブスクリプション更新を処理"""
        # サブスクリプション状態に応じてプラン更新
        pass

    def _handle_subscription_deleted(self, subscription: dict):
        """サブスクリプション削除を処理"""
        # Freeプランにダウングレード
        subscription_id = subscription.get("id")

        for email, user in st.session_state.get("auth_users_db", {}).items():
            if user.stripe_subscription_id == subscription_id:
                self.auth_manager.update_plan(email, PlanType.FREE)
                break


def render_pricing_ui():
    """料金プランUIを描画"""
    st.subheader("💰 料金プラン")

    auth_manager = AuthManager()
    billing_manager = BillingManager()
    current_user = auth_manager.get_current_user()
    current_plan = current_user.plan if current_user else PlanType.FREE

    # 3カラムでプラン表示
    cols = st.columns(3)

    plans_to_show = [PlanType.FREE, PlanType.BASIC, PlanType.PRO]

    for i, plan in enumerate(plans_to_show):
        config = PRICE_CONFIGS[plan]
        is_current = plan == current_plan

        with cols[i]:
            # カードスタイル
            container = st.container(border=True)

            with container:
                # プラン名
                if is_current:
                    st.markdown(f"### 🎯 {plan.value.upper()}")
                    st.caption("現在のプラン")
                else:
                    st.markdown(f"### {plan.value.upper()}")

                # 価格
                if config.price_jpy == 0:
                    st.markdown("## 無料")
                else:
                    st.markdown(f"## ¥{config.price_jpy:,}/月")

                st.divider()

                # 機能リスト
                for feature in config.features:
                    st.write(f"✓ {feature}")

                st.divider()

                # ボタン
                if is_current:
                    st.button("現在のプラン", disabled=True, key=f"btn_{plan.value}")
                elif plan == PlanType.FREE:
                    # Freeへのダウングレードは別途処理
                    if current_plan != PlanType.FREE:
                        if st.button("ダウングレード", key=f"btn_{plan.value}"):
                            st.warning("ダウングレードは現在のサブスクリプション終了後に適用されます")
                else:
                    # 有料プランへのアップグレード
                    if st.button(
                        "アップグレード",
                        type="primary" if plan == PlanType.BASIC else "secondary",
                        key=f"btn_{plan.value}",
                    ):
                        if not current_user:
                            st.warning("プランを変更するにはログインが必要です")
                        elif billing_manager.stripe_available:
                            checkout_url = billing_manager.create_checkout_session(plan)
                            if checkout_url:
                                st.markdown(f"[お支払いページへ]({checkout_url})")
                            else:
                                st.error("チェックアウトセッションの作成に失敗しました")
                        else:
                            st.info("Stripe設定が完了していません。管理者にお問い合わせください。")
                            # デモ用: 即座にプランを変更
                            if st.button("デモ: プラン変更", key=f"demo_{plan.value}"):
                                auth_manager.update_plan(current_user.email, plan)
                                st.success(f"{plan.value.upper()}プランに変更しました")
                                st.rerun()

    # エンタープライズプラン
    st.divider()
    st.markdown("### 🏢 Enterpriseプラン")
    st.write("大規模組織向けのカスタムプランです。お問い合わせください。")
    st.button("お問い合わせ", key="btn_enterprise")


def render_billing_status():
    """課金状態を表示"""
    auth_manager = AuthManager()
    billing_manager = BillingManager()

    user = auth_manager.get_current_user()
    if not user:
        return

    if user.plan == PlanType.FREE:
        st.info("💡 有料プランにアップグレードすると、より多くの機能が使えます")
        return

    # サブスクリプション状態を取得
    status = billing_manager.get_subscription_status()

    if status:
        with st.expander("📋 サブスクリプション情報"):
            st.write(f"**状態**: {status['status']}")
            st.write(f"**次回更新日**: {status['current_period_end'].strftime('%Y/%m/%d')}")

            if status["cancel_at_period_end"]:
                st.warning("キャンセル予約済み（期間終了時に終了）")
            else:
                if st.button("サブスクリプションをキャンセル"):
                    if billing_manager.cancel_subscription():
                        st.success("キャンセルしました（期間終了まで利用可能）")
                        st.rerun()
