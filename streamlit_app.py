import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from ui_components import inject_dashboard_css
from page_overview import render_page1
from page_site import render_page2

from app_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_WEATHER_FC_PATH,
    TARGET_OPTIONS,
    load_pollution_data,
    load_weather_forecast,
    attach_site_clusters,
    normalize_columns,
    to_datetime_safe,
    precompute_all_prophet,
    safe_get_secret,
    load_alert_state,
    save_alert_state,
    send_slack,
    fmt_date,
    get_query_params,
    set_query_params_safe,
    has_requests,
)

st.set_page_config(
    page_title="Air Quality Decision Dashboard",
    layout="wide",
)

MODEL_PATHS = {
    "O3": "models/model_o3.joblib",
    "NO2": "models/model_no2.joblib",
    "CO": "models/model_co.joblib",
    "SO2": "models/model_so2.joblib",
}

SPIKE_THRESHOLD = 0.30

FUTURE_INPUT_DEFAULT = "future_input_2024_01_01_to_07_all_sites_MODELREADY.csv"
FUTURE_INPUT_PATTERNS = [
    "future_input_*MODELREADY*.csv",
    "future_input_*.csv",
]

class DummySpikeModel:
    _is_dummy = True
    feature_name_ = []
    def predict_proba(self, X):
        n = 0 if X is None else len(X)
        return np.column_stack([np.ones(n), np.zeros(n)])

def _get_model_features(model) -> list[str]:
    feats = getattr(model, "feature_name_", None)
    if feats is None:
        feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return []
    return list(feats)

def _resolve_existing_path(path_str: str, patterns: list[str]) -> Path:
    base = Path(__file__).resolve().parent
    search_dirs = [
        Path.cwd(),
        base,
        base.parent,
        base / "data",
        base.parent / "data",
        Path.cwd() / "data",
    ]

    if path_str:
        p = Path(path_str)
        if p.is_file():
            return p
        for d in search_dirs:
            cand = d / path_str
            if cand.is_file():
                return cand

    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            hits = sorted([p for p in d.glob(pat) if p.is_file()])
            if hits:
                return hits[0]

    raise FileNotFoundError(
        f"미래 입력 CSV를 찾지 못했습니다. 입력='{path_str}'. "
        f"탐색폴더={[str(x) for x in search_dirs]}"
    )

@st.cache_data(show_spinner=False)
def load_future_input(path_str: str) -> pd.DataFrame:
    p = _resolve_existing_path(path_str, FUTURE_INPUT_PATTERNS)
    df = pd.read_csv(p)
    df = normalize_columns(df)
    if "date" not in df.columns or "site" not in df.columns:
        raise ValueError("미래 입력 CSV에 date, site 컬럼이 필요합니다.")
    df["date"] = to_datetime_safe(df["date"])
    df = df.dropna(subset=["date", "site"]).copy()
    return df.sort_values(["site", "date"]).reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_spike_models_safe():
    models = {}
    errors = []
    for k, path in MODEL_PATHS.items():
        try:
            models[k] = joblib.load(path)
        except Exception as e:
            models[k] = DummySpikeModel()
            errors.append(f"{k}: {type(e).__name__}")
    return models, errors

def ensure_required_features(
    df_future_site: pd.DataFrame,
    df_site_hist: pd.DataFrame,
    required_features: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df_future_site.copy()
    filled: dict[str, str] = {}

    fut_map = {c.lower(): c for c in out.columns}
    hist_map = {c.lower(): c for c in df_site_hist.columns}

    def _last_nonnull(df_: pd.DataFrame, col_: str) -> float:
        s = pd.to_numeric(df_[col_], errors="coerce")
        if s.notna().any():
            return float(s.dropna().iloc[-1])
        return 0.0

    for feat in required_features:
        if feat in out.columns:
            continue
        key = feat.lower()

        if key in fut_map:
            src = fut_map[key]
            out[feat] = out[src]
            filled[feat] = f"alias_from_future:{src}"
            continue

        if key in hist_map:
            src = hist_map[key]
            out[feat] = _last_nonnull(df_site_hist, src)
            filled[feat] = f"constant_from_hist:{src}"
            continue

        out[feat] = 0.0
        filled[feat] = "default:0"

    for feat in required_features:
        out[feat] = pd.to_numeric(out[feat], errors="coerce")
        if out[feat].isna().all():
            out[feat] = 0.0
        else:
            out[feat] = out[feat].fillna(out[feat].median())

    return out, filled

def main():
    # ===== 전역 UI 상태 기본값 =====
    if "ui_dark_mode" not in st.session_state:
        st.session_state["ui_dark_mode"] = False
    if "mute_alerts" not in st.session_state:
        st.session_state["mute_alerts"] = False
    if "sidebar_color_hex" not in st.session_state:
        st.session_state["sidebar_color_hex"] = "#0B1F3A"  # 기본 네이비
    if "sidebar_color_mode" not in st.session_state:
        st.session_state["sidebar_color_mode"] = "프리셋"
    if "sidebar_color_preset" not in st.session_state:
        st.session_state["sidebar_color_preset"] = "Navy (#0B1F3A)"
    if "sidebar_color_custom" not in st.session_state:
        st.session_state["sidebar_color_custom"] = "#0B1F3A"

    # ===== CSS 적용 =====
    FIXED_SIDEBAR_BG = "#0B3A3A"

    inject_dashboard_css(
        sidebar_bg=FIXED_SIDEBAR_BG,
        dark_mode=bool(st.session_state["ui_dark_mode"]),
    )

    st.markdown(
        """
        <style>
        div.block-container { padding-top: 0.6rem !important; padding-bottom: 1.2rem !important; }
        section.main > div { padding-top: 0.6rem !important; }
        header { margin-bottom: 0rem !important; }
        main { padding-top: 0rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ================= Sidebar =================
    with st.sidebar:
        st.header("설정")

        # --- 데이터 경로
        data_path = st.text_input("대기 데이터 경로", value=DEFAULT_DATA_PATH)
        weather_fc_path = st.text_input("미래 기상 경로(optional)", value=DEFAULT_WEATHER_FC_PATH)

        st.divider()

        

        future_path_default = st.session_state.get("future_input_path", FUTURE_INPUT_DEFAULT)
        st.text_input(
            "미래 입력 CSV 경로",
            value=future_path_default,
            key="future_input_path",
            help="예: data/future_input_....csv (없으면 자동 탐색)",
        )


        st.divider()

        # --- Slack
        st.subheader("🔔 Slack 알림")
        slack_url = safe_get_secret("SLACK_WEBHOOK_URL")

        if not slack_url:
            st.error("Slack Webhook 미설정 (secrets 필요)")
            slack_enabled = False
        else:
            slack_enabled = st.checkbox("Slack 알림 활성화", value = True)

        # ✅ 우측 상단 음소거 토글(session_state) 우선 적용
        if bool(st.session_state.get("mute_alerts", False)):
            slack_enabled = False
            st.caption("🔕 알림 음소거(ON) → Slack 알림 OFF")

        if slack_url and not has_requests():
            st.warning("Slack 발송을 쓰려면 `pip install requests` 필요")

        st.divider()

        qp = get_query_params()
        page = qp.get("page", ["overview"])[0]

        nav = st.radio(
            "페이지",
            ["overview", "site"],
            index=0 if page == "overview" else 1,
            format_func=lambda x: "Page 1 (Overview)" if x == "overview" else "Page 2 (Site)",
        )

    # ================= Load Data =================
    df_raw = load_pollution_data(data_path)
    df_all = attach_site_clusters(df_raw)

    anchor = pd.to_datetime(df_all["date"].max())
    weather_fc = load_weather_forecast(weather_fc_path)

# ================= Spike Summary (앱 시작 시 1회 계산) =================
    if "SPIKE_DF" not in st.session_state:

        with st.spinner("⚡ 스파이크 요약 최초 1회 계산 중..."):

            df_future = load_future_input(
                st.session_state.get("future_input_path", FUTURE_INPUT_DEFAULT)
            )
            models, model_errors = load_spike_models_safe()

            if model_errors:
                st.warning(f"스파이크 모델 일부 로드 실패: {', '.join(model_errors)}")

            spike_rows = []

            for s in df_all["site"].unique():
                df_f_site = df_future[df_future["site"].astype(str) == str(s)]
                df_h_site = df_all[df_all["site"].astype(str) == str(s)]

                if df_f_site.empty:
                    spike_rows.append({"site": s, "spike_days": 0, "priority": "LOW"})
                    continue

                required_union = []
                for m in models.values():
                    required_union += _get_model_features(m)
                required_union = list(dict.fromkeys(required_union))

                df_f_site_filled, _ = ensure_required_features(
                    df_f_site, df_h_site, required_union
                )

                total_spikes = 0
                for m in models.values():
                    if getattr(m, "_is_dummy", False):
                        continue
                    feats = _get_model_features(m)
                    if not feats:
                        continue
                    probs = m.predict_proba(df_f_site_filled[feats])[:, 1]
                    total_spikes += int((probs >= SPIKE_THRESHOLD).sum())

                priority = (
                    "HIGH" if total_spikes >= 9 else
                    "MEDIUM" if total_spikes >= 6 else
                    "LOW"
                )

                spike_rows.append({
                    "site": s,
                    "spike_days": total_spikes,
                    "priority": priority,
                })

            st.session_state["SPIKE_DF"] = pd.DataFrame(spike_rows)

    spike_df = st.session_state["SPIKE_DF"]
    spike_status = "ON"

    # ================= Slack Alert (옵션) =================
    if spike_status == "ON" and slack_url and slack_enabled:
        alert_state = load_alert_state()

        for _, r in spike_df.iterrows():
            s = r["site"]
            if r["priority"] != "HIGH":
                continue
            if not can_send(alert_state, s):
                continue

            msg = (
                f"🚨 *대기질 스파이크 감지*\n"
                f"- 관측소: *{s}*\n"
                f"- 예상 스파이크 일수: *{int(r['spike_days'])}일*\n"
                f"- 우선순위: *HIGH*\n"
                f"- 조치 필요"
            )

            try:
                send_slack(slack_url, msg)
                mark_sent(alert_state, s)
            except Exception as e:
                st.error(f"Slack 전송 실패: {e}")

        save_alert_state(alert_state)

        # ================= Prophet 전체 선계산 (딱 1번) =================
    prophet_sig = (
        data_path,
        weather_fc_path,
        anchor.date(),  # 하루 단위 고정
        tuple(TARGET_OPTIONS),
    )

    if st.session_state.get("PROPHET_SIG") != prophet_sig:
        st.session_state["PROPHET_SIG"] = prophet_sig
        st.session_state.pop("ALL_PROPHET", None)

    if "ALL_PROPHET" not in st.session_state:
        with st.spinner("⏳ Prophet 전체 예측 최초 1회 계산 중..."):
            st.session_state["ALL_PROPHET"] = precompute_all_prophet(
                df_all=df_all,
                targets=TARGET_OPTIONS,
                anchor=anchor,
                horizon=7,
                interval_width=0.90,
                weather_fc=weather_fc,
            )

    ALL_PROPHET = st.session_state["ALL_PROPHET"]

    # ================= Router =================
    st.sidebar.info(f"오늘(기준일): {fmt_date(anchor)}")
    st.sidebar.caption(f"스파이크 요약 상태: {spike_status}")

    if nav == "overview":
        set_query_params_safe(page="overview")
        render_page1(
            df_all=df_all,
            spike_df=spike_df,
            map_show_state=False,
        )
    else:
        qp = get_query_params()
        site = qp.get("site", [None])[0] or st.session_state.get("p1_site_sel", None)

        if not site:
            st.warning("Page 1에서 관측소를 먼저 선택하세요.")
            return

        set_query_params_safe(page="site", site=site)
        render_page2(
            df_all=df_all,
            site=str(site),
            target="o3_mean",
            anchor=anchor,
            horizon=7,
            interval_width=0.90,
            weather_fc=weather_fc,
            thr_config={},
            ALL_PROPHET=ALL_PROPHET,
        )

if __name__ == "__main__":
    main()
