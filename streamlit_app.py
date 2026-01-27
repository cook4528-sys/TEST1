# streamlit_app.py
# ============================================================
# 2-page Air Quality Decision Dashboard (Streamlit)
# Page 1: Situation Overview & Map (선별)
#  - 지도: 클러스터 색상(circle) + (오늘 State Risk) ⚠/❗ + (예측 임계치 초과) ❕/❗(오프셋)
#  - 우측 패널: 선택 관측소 요약 + (전체) 예측 임계치 경고 사이트 목록 + "초과 예상 t+몇일" 컬럼
# Page 2: Site Analysis & Action Support (판단·결정)
#  - 통합 시계열 + 미래 예측(라인 확대) 포함
# Slack:
#  - secrets.toml 안전 처리 + 테스트 버튼 + 쿨다운
# Threshold:
#  - 기본: 사이트 P95 ON / 고정값 OFF / 계절 가변 OFF
# ============================================================

import os
import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prophet
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# Slack
try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


# =========================
# App Config
# =========================
st.set_page_config(page_title="Air Quality Decision Dashboard", layout="wide")

DEFAULT_DATA_PATH = "data/pollution_2018_2023_8.csv"
DEFAULT_WEATHER_FC_PATH = ""  # optional

AQI_COLS = ["o3_aqi", "no2_aqi", "co_aqi", "so2_aqi"]
MEAN_COLS = ["o3_mean", "no2_mean", "co_mean", "so2_mean"]
TARGET_OPTIONS = MEAN_COLS

# site cluster 5개 → 3단계 축약 (지도 색상)
SITE_CLUSTER_MAP_5 = {
    0: "moderate",
    1: "risk",
    2: "high-risk",
    3: "safe",
    4: "moderate-episodic",
}
SITE_CLUSTER_3LVL = {
    "safe": "Stable",
    "moderate": "Stable",
    "risk": "Risk",
    "moderate-episodic": "Risk",
    "high-risk": "High-risk",
}

STATE_RISK_ICON = {"None": "", "Medium": "⚠", "High": "❗"}
SPIKE_RISK_ICON = {"None": "—", "Watch": "🟡 Watch", "Warn": "🔴 Warn"}

# 지도 색상(고정)
CLUSTER_COLORS = {
    "Stable": "#2ca02c",
    "Risk": "#ff7f0e",
    "High-risk": "#d62728",
    "Unknown": "#7f7f7f",
}

# ✅ Spike 아이콘 오프셋(겹침 최소화)
# - 데이터/줌레벨 따라 체감이 다르므로 필요 시 조정
SPIKE_ICON_OFFSET = {
    "Watch": (0.05, -0.05),  # (lat_offset, lon_offset)
    "Warn":  (0.06,  0.06),
}

# Slack alert cooldown state
ALERT_STATE_PATH = Path(".cache") / "slack_alert_state.json"
ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
def safe_get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    if "date" not in df.columns:
        for cand in ["day", "datetime", "timestamp"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break

    if "site" not in df.columns:
        for cand in ["address", "site_name", "station", "station_name"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "site"})
                break

    return df


def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")


def parse_point_geometry(val: str) -> Tuple[Optional[float], Optional[float]]:
    if pd.isna(val):
        return (None, None)
    try:
        obj = json.loads(val)
        if obj.get("type") != "Point":
            return (None, None)
        lon, lat = obj.get("coordinates", [None, None])
        return (float(lat), float(lon))
    except Exception:
        return (None, None)


def coerce_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == "object":
            out[c] = (
                out[c].astype(str)
                .str.replace(",", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    u = pd.to_numeric(u, errors="coerce")
    v = pd.to_numeric(v, errors="coerce")
    return np.sqrt(u * u + v * v)


def fmt_date(d) -> str:
    if pd.isna(d):
        return "-"
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def get_query_params() -> Dict[str, List[str]]:
    try:
        return {k: [v] if isinstance(v, str) else list(v) for k, v in dict(st.query_params).items()}
    except Exception:
        return st.experimental_get_query_params()


def set_query_params_safe(**kwargs):
    qp = get_query_params()
    new_qp = {k: ([v] if isinstance(v, str) else list(v)) for k, v in kwargs.items()}

    def _norm(d):
        out = {}
        for k, v in d.items():
            if v is None:
                out[k] = [""]
            elif isinstance(v, list):
                out[k] = [str(x) for x in v]
            else:
                out[k] = [str(v)]
        return out

    if _norm(qp) == _norm(new_qp):
        return

    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)


# =========================
# Load & Preprocess
# =========================
@st.cache_data(show_spinner=False)
def load_pollution_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)

    if "date" not in df.columns or "site" not in df.columns:
        raise ValueError("필수 컬럼(date, site)을 찾지 못했습니다. 원본 컬럼명을 확인하세요.")

    df["date"] = to_datetime_safe(df["date"])
    df = df.dropna(subset=["date", "site"]).copy()

    if "geometry" in df.columns:
        latlon = df["geometry"].apply(parse_point_geometry)
        df["lat"] = [t[0] for t in latlon]
        df["lon"] = [t[1] for t in latlon]
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    exclude = ["site", "date", "geometry", "state", "county", "city", "region_name"]
    df = coerce_numeric_columns(df, exclude=exclude)

    if "met_wind_u" in df.columns and "met_wind_v" in df.columns:
        df["wind_speed"] = wind_speed(df["met_wind_u"], df["met_wind_v"])

    # site-date 중복 제거(수치=mean, 범주=first)
    key_cols = ["site", "date"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in key_cols]
    other_cols = [c for c in df.columns if c not in key_cols + numeric_cols]
    agg = {**{c: "mean" for c in numeric_cols}, **{c: "first" for c in other_cols}}

    df = df.groupby(key_cols, as_index=False).agg(agg)
    return df.sort_values(["site", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_weather_forecast(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None

    w = pd.read_csv(path)
    w = normalize_columns(w)

    if "date" not in w.columns or "site" not in w.columns:
        raise ValueError("기상 파일에 date, site 컬럼이 필요합니다.")

    w["date"] = to_datetime_safe(w["date"])
    w = w.dropna(subset=["date", "site"]).copy()

    exclude = ["site", "date", "geometry", "state", "county", "city", "region_name"]
    w = coerce_numeric_columns(w, exclude=exclude)

    if "met_wind_u" in w.columns and "met_wind_v" in w.columns:
        w["wind_speed"] = wind_speed(w["met_wind_u"], w["met_wind_v"])

    key_cols = ["site", "date"]
    numeric_cols = w.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in key_cols]
    other_cols = [c for c in w.columns if c not in key_cols + numeric_cols]
    agg = {**{c: "mean" for c in numeric_cols}, **{c: "first" for c in other_cols}}
    w = w.groupby(key_cols, as_index=False).agg(agg)

    return w.sort_values(["site", "date"]).reset_index(drop=True)


# =========================
# Clustering (Map Color)
# =========================
@st.cache_resource(show_spinner=False)
def fit_day_cluster(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    work = df.dropna(subset=AQI_COLS).copy()
    if work.empty:
        raise ValueError(f"AQI 컬럼({AQI_COLS}) 결측으로 day-cluster를 학습할 수 없습니다.")

    X = work[AQI_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(Xs)
    return km, scaler


@st.cache_resource(show_spinner=False)
def fit_site_cluster(
    df: pd.DataFrame,
    _day_km: KMeans,
    _day_scaler: StandardScaler,
    n_clusters: int = 5,
    random_state: int = 42,
):
    work = df.dropna(subset=AQI_COLS).copy()
    X_day = _day_scaler.transform(work[AQI_COLS].values)
    work["day_cluster"] = _day_km.predict(X_day)
    work["total_aqi"] = work[AQI_COLS].sum(axis=1)

    site_features = (
        work.groupby("site")
        .agg(
            mean_total_aqi=("total_aqi", "mean"),
            std_total_aqi=("total_aqi", "std"),
            **{f"pct_day_cluster_{i}": ("day_cluster", lambda x, i=i: (x == i).mean()) for i in range(_day_km.n_clusters)},
        )
        .reset_index()
        .fillna(0)
    )

    feat_cols = [c for c in site_features.columns if c != "site"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(site_features[feat_cols].values)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(Xs)
    return km, scaler


@st.cache_data(show_spinner=False)
def attach_site_clusters(df: pd.DataFrame) -> pd.DataFrame:
    day_km, day_scaler = fit_day_cluster(df)
    site_km, site_scaler = fit_site_cluster(df, day_km, day_scaler)

    work = df.dropna(subset=AQI_COLS).copy()
    X_day = day_scaler.transform(work[AQI_COLS].values)
    work["day_cluster"] = day_km.predict(X_day)
    work["total_aqi"] = work[AQI_COLS].sum(axis=1)

    site_feat = (
        work.groupby("site")
        .agg(
            mean_total_aqi=("total_aqi", "mean"),
            std_total_aqi=("total_aqi", "std"),
            **{f"pct_day_cluster_{i}": ("day_cluster", lambda x, i=i: (x == i).mean()) for i in range(day_km.n_clusters)},
        )
        .reset_index()
        .fillna(0)
    )

    feat_cols = [c for c in site_feat.columns if c != "site"]
    Xs = site_scaler.transform(site_feat[feat_cols].values)
    site_feat["cluster_k5"] = site_km.predict(Xs).astype(int)

    site_feat["cluster_5name"] = site_feat["cluster_k5"].map(SITE_CLUSTER_MAP_5).fillna("risk")
    site_feat["cluster_3name"] = site_feat["cluster_5name"].map(SITE_CLUSTER_3LVL).fillna("Risk")

    return df.merge(site_feat[["site", "cluster_k5", "cluster_5name", "cluster_3name"]], on="site", how="left")


# =========================
# Threshold Policy
# =========================
@st.cache_data(show_spinner=False)
def compute_threshold_tables(df: pd.DataFrame, target: str, site_q: float, season_q: float) -> Tuple[pd.Series, pd.DataFrame]:
    hist = df.dropna(subset=[target]).copy()
    if hist.empty:
        return pd.Series(dtype=float), pd.DataFrame(columns=["site", "month", "thr_season"])

    hist["month"] = hist["date"].dt.month.astype(int)
    site_thr = hist.groupby("site")[target].quantile(site_q)

    season_thr = (
        hist.groupby(["site", "month"])[target]
        .quantile(season_q)
        .rename("thr_season")
        .reset_index()
    )
    return site_thr, season_thr


def threshold_for(
    site: str,
    date: pd.Timestamp,
    target: str,
    fixed_value: float,
    use_fixed: bool,
    site_thr: pd.Series,
    use_site: bool,
    season_thr_df: pd.DataFrame,
    use_season: bool,
) -> float:
    vals = []
    if use_fixed and np.isfinite(fixed_value):
        vals.append(float(fixed_value))

    if use_site and site in site_thr.index and np.isfinite(site_thr.loc[site]):
        vals.append(float(site_thr.loc[site]))

    if use_season and not season_thr_df.empty:
        m = int(pd.to_datetime(date).month)
        hit = season_thr_df[(season_thr_df["site"] == site) & (season_thr_df["month"] == m)]
        if not hit.empty and np.isfinite(hit["thr_season"].iloc[0]):
            vals.append(float(hit["thr_season"].iloc[0]))

    return float(np.max(vals)) if vals else np.nan


def calc_state_risk_today(y_today: float, thr_today: float, medium_ratio: float = 0.90) -> str:
    if not np.isfinite(y_today) or not np.isfinite(thr_today):
        return "None"
    if y_today >= thr_today:
        return "High"
    if y_today >= (thr_today * medium_ratio):
        return "Medium"
    return "None"


# =========================
# Prophet Forecast + Spike Risk
# =========================
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def fill_exog_climatology(df_site: pd.DataFrame, exog_cols: List[str]) -> pd.DataFrame:
    if not exog_cols:
        return df_site
    out = df_site.copy()
    out["doy"] = out["date"].dt.dayofyear
    for c in exog_cols:
        if c not in out.columns:
            out[c] = np.nan
        clim = out.groupby("doy")[c].mean()
        out[c] = out[c].fillna(out["doy"].map(clim))
        out[c] = out[c].fillna(out[c].mean())
    return out.drop(columns=["doy"])


@st.cache_data(show_spinner=False)
def prophet_predict_site(
    df_site: pd.DataFrame,
    target: str,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    weather_fc: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if not _HAS_PROPHET:
        raise RuntimeError("Prophet이 설치되어 있지 않습니다. pip install prophet")

    exog_candidates = ["temp_c", "pressure_pa", "met_rain_mm", "wind_speed", "ndbi_mean", "ndvi_mean"]
    exog_cols = [c for c in exog_candidates if c in df_site.columns]

    hist = df_site[df_site["date"] <= anchor].copy()
    hist = hist.dropna(subset=[target]).copy()

    if hist["date"].nunique() < 60:
        return pd.DataFrame(columns=["date", "y", "yhat", "yhat_lower", "yhat_upper"])

    hist = ensure_columns(hist, exog_cols)
    hist = fill_exog_climatology(hist, exog_cols)

    future_dates = pd.date_range(anchor + pd.Timedelta(days=1), anchor + pd.Timedelta(days=horizon), freq="D")
    fut = pd.DataFrame({"date": future_dates})
    fut = ensure_columns(fut, exog_cols)

    if weather_fc is not None and not weather_fc.empty:
        site_key = str(df_site["site"].iloc[0])
        wf = weather_fc[weather_fc["site"].astype(str) == site_key].copy()
        wf = wf[wf["date"].isin(future_dates)].copy()
        if not wf.empty:
            keep = ["date"] + [c for c in exog_cols if c in wf.columns]
            fut = fut.merge(wf[keep], on="date", how="left", suffixes=("", "_wf"))
            for c in exog_cols:
                if f"{c}_wf" in fut.columns:
                    fut[c] = fut[c].combine_first(fut[f"{c}_wf"])
                    fut = fut.drop(columns=[f"{c}_wf"])

    for c in ["ndbi_mean", "ndvi_mean"]:
        if c in exog_cols:
            const_val = pd.to_numeric(hist[c], errors="coerce").dropna()
            const_val = float(const_val.iloc[-1]) if not const_val.empty else np.nan
            fut[c] = fut[c].fillna(const_val)

    if exog_cols:
        tmp = pd.concat([hist[["date"] + exog_cols], fut[["date"] + exog_cols]], ignore_index=True)
        tmp = fill_exog_climatology(tmp, exog_cols)
        fut[exog_cols] = tmp.iloc[-len(fut):][exog_cols].values

    train_cols = ["ds", "y"] + exog_cols
    train = hist.rename(columns={"date": "ds", target: "y"})[train_cols]

    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    for c in exog_cols:
        m.add_regressor(c)
    m.fit(train)

    all_dates = pd.date_range(hist["date"].min(), anchor + pd.Timedelta(days=horizon), freq="D")
    base = pd.DataFrame({"date": all_dates})
    merge_cols = ["date", target] + exog_cols
    base = base.merge(hist[merge_cols], on="date", how="left")
    base = ensure_columns(base, exog_cols)
    base = fill_exog_climatology(base, exog_cols)

    if not fut.empty and exog_cols:
        base = base.merge(fut[["date"] + exog_cols], on="date", how="left", suffixes=("", "_fut"))
        for c in exog_cols:
            if f"{c}_fut" in base.columns:
                base.loc[base["date"] > anchor, c] = base.loc[base["date"] > anchor, f"{c}_fut"]
                base = base.drop(columns=[f"{c}_fut"])

    pred_in = base.rename(columns={"date": "ds"})[["ds"] + exog_cols]
    fc = m.predict(pred_in)

    out = pd.DataFrame({
        "date": pd.to_datetime(pred_in["ds"]),
        "y": pd.to_numeric(base[target], errors="coerce"),
        "yhat": fc["yhat"].values,
        "yhat_lower": fc["yhat_lower"].values,
        "yhat_upper": fc["yhat_upper"].values,
    })
    return out


@st.cache_data(show_spinner=False)
def compute_spike_risk_all_sites(
    df: pd.DataFrame,
    target: str,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    fixed_value: float,
    use_fixed: bool,
    use_site: bool,
    site_q: float,
    use_season: bool,
    season_q: float,
    warn_days: int,
    watch_days: int,
    weather_fc: Optional[pd.DataFrame],
    medium_ratio: float,
) -> pd.DataFrame:
    site_thr, season_thr_df = compute_threshold_tables(df, target=target, site_q=site_q, season_q=season_q)

    sites = df["site"].astype(str).unique().tolist()
    rows = []

    for site in sites:
        df_site = df[df["site"].astype(str) == str(site)].copy().sort_values("date")

        today_row = df_site[df_site["date"] == anchor]
        if today_row.empty:
            today_row = df_site[df_site["date"] <= anchor].tail(1)
        if today_row.empty:
            continue

        y_today = float(pd.to_numeric(today_row[target].iloc[0], errors="coerce")) if target in today_row.columns else np.nan
        thr_today = threshold_for(site, anchor, target, fixed_value, use_fixed, site_thr, use_site, season_thr_df, use_season)
        state_risk = calc_state_risk_today(y_today, thr_today, medium_ratio=medium_ratio)

        spike_level = "None"
        exceed_days = 0
        max_upper = np.nan
        max_thr = np.nan

        # ✅ 추가: 초과 예상 t+몇일
        exceed_tplus = ""  # 예: "t+2,t+3"
        exceed_first_tplus = np.nan

        try:
            pred = prophet_predict_site(df_site, target, anchor, horizon, interval_width, weather_fc)
            if not pred.empty:
                fut = pred[(pred["date"] > anchor) & (pred["date"] <= anchor + pd.Timedelta(days=horizon))].copy()
                if not fut.empty:
                    fut["thr"] = [
                        threshold_for(site, d_, target, fixed_value, use_fixed, site_thr, use_site, season_thr_df, use_season)
                        for d_ in fut["date"]
                    ]
                    fut["exceed"] = (fut["yhat_upper"] > fut["thr"])

                    exceed_rows = fut[fut["exceed"] == True].copy()
                    exceed_days = int(exceed_rows.shape[0])

                    if exceed_days > 0:
                        offsets = sorted({int((d - anchor).days) for d in exceed_rows["date"].tolist()})
                        exceed_first_tplus = float(offsets[0]) if offsets else np.nan
                        # 너무 길어지면 최대 7개까지만
                        exceed_tplus = ",".join([f"t+{o}" for o in offsets[:7]])

                    max_upper = float(np.nanmax(fut["yhat_upper"].values))
                    max_thr = float(np.nanmax(fut["thr"].values))

                    if exceed_days >= warn_days:
                        spike_level = "Warn"
                    elif exceed_days >= watch_days:
                        spike_level = "Watch"
                    else:
                        spike_level = "None"
        except Exception:
            spike_level = "None"

        lat = float(pd.to_numeric(today_row["lat"].iloc[0], errors="coerce")) if "lat" in today_row.columns else np.nan
        lon = float(pd.to_numeric(today_row["lon"].iloc[0], errors="coerce")) if "lon" in today_row.columns else np.nan

        state = str(today_row["state"].iloc[0]) if "state" in today_row.columns and pd.notna(today_row["state"].iloc[0]) else ""
        county = str(today_row["county"].iloc[0]) if "county" in today_row.columns and pd.notna(today_row["county"].iloc[0]) else ""
        city = str(today_row["city"].iloc[0]) if "city" in today_row.columns and pd.notna(today_row["city"].iloc[0]) else ""
        cluster_3 = str(today_row["cluster_3name"].iloc[0]) if "cluster_3name" in today_row.columns and pd.notna(today_row["cluster_3name"].iloc[0]) else "Risk"

        rows.append({
            "site": site,
            "cluster_3name": cluster_3,
            "lat": lat,
            "lon": lon,
            "state": state,
            "county": county,
            "city": city,
            "today_y": y_today,
            "today_thr": thr_today,
            "state_risk": state_risk,
            "spike_exceed_days": exceed_days,
            "spike_risk": spike_level,
            "max_yhat_upper_7d": max_upper,
            "max_thr_7d": max_thr,
            # ✅ 신규 컬럼
            "exceed_tplus": exceed_tplus,
            "exceed_first_tplus": exceed_first_tplus,
        })

    return pd.DataFrame(rows)


# =========================
# Figures (Map + TS)
# =========================
def _hover_text(row: pd.Series) -> str:
    tplus = row.get("exceed_tplus", "")
    tplus_txt = f"<br>Exceed t+: {tplus}" if isinstance(tplus, str) and tplus else ""
    return (
        f"<b>{row['site']}</b><br>"
        f"Cluster: {row.get('cluster_3name','-')}<br>"
        f"State Risk: {row.get('state_risk','-')}<br>"
        f"Spike Risk(7d): {row.get('spike_risk','-')} (days={int(row.get('spike_exceed_days',0))})"
        f"{tplus_txt}<br>"
        f"Today y: {row.get('today_y',np.nan):.5g}<br>"
        f"Today thr: {row.get('today_thr',np.nan):.5g}<br>"
        f"{row.get('state','')} {row.get('county','')} {row.get('city','')}"
    )


def build_map_figure(snap: pd.DataFrame) -> go.Figure:
    """
    - Base: 클러스터 색상(circle)
    - Overlay1: 오늘 State Risk 아이콘(⚠/❗) markers+text
    - Overlay2: 예측 임계치 초과(Spike Risk) 아이콘(❕/❗) text + ✅ 오프셋
    """
    s = snap.dropna(subset=["lat", "lon"]).copy()
    fig = go.Figure()

    if s.empty:
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    # Base layer: cluster별
    for cl in ["Stable", "Risk", "High-risk"]:
        sub = s[s["cluster_3name"].astype(str) == cl].copy()
        if sub.empty:
            continue

        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"],
            lon=sub["lon"],
            mode="markers",
            name=f"{cl}",
            marker=dict(
                size=10,
                color=CLUSTER_COLORS.get(cl, CLUSTER_COLORS["Unknown"]),
                opacity=0.75,
                allowoverlap=True,
            ),
            customdata=sub[["site"]].values,
            hovertext=sub.apply(_hover_text, axis=1),
            hoverinfo="text",
        ))

    # Overlay1: 오늘 State Risk
    for risk_level in ["Medium", "High"]:
        sub = s[s["state_risk"].astype(str) == risk_level].copy()
        if sub.empty:
            continue

        for cl in ["Stable", "Risk", "High-risk"]:
            sub2 = sub[sub["cluster_3name"].astype(str) == cl].copy()
            if sub2.empty:
                continue

            icon = STATE_RISK_ICON.get(risk_level, "")
            fig.add_trace(go.Scattermapbox(
                lat=sub2["lat"],
                lon=sub2["lon"],
                mode="markers+text",
                name=f"{risk_level} (today)",
                marker=dict(
                    size=16 if risk_level == "High" else 14,
                    color=CLUSTER_COLORS.get(cl, CLUSTER_COLORS["Unknown"]),
                    opacity=0.95,
                    allowoverlap=True,
                ),
                text=[icon] * len(sub2),
                textposition="middle center",
                textfont=dict(size=18),
                customdata=sub2[["site"]].values,
                hovertext=sub2.apply(_hover_text, axis=1),
                hoverinfo="text",
                showlegend=False,
            ))

    # Overlay2: 예측 임계치 초과(Spike Risk) + ✅ 오프셋
    SPIKE_MAP_ICON = {"Watch": "❕", "Warn": "❗"}
    for spike_level in ["Watch", "Warn"]:
        sub = s[s["spike_risk"].astype(str) == spike_level].copy()
        if sub.empty:
            continue

        icon = SPIKE_MAP_ICON.get(spike_level, "❗")
        dlat, dlon = SPIKE_ICON_OFFSET.get(spike_level, (0.05, 0.05))

        fig.add_trace(go.Scattermapbox(
            lat=(sub["lat"] + dlat),
            lon=(sub["lon"] + dlon),
            mode="text",
            name=f"{spike_level} (forecast)",
            text=[icon] * len(sub),
            textposition="middle center",
            textfont=dict(size=22 if spike_level == "Warn" else 20),
            customdata=sub[["site"]].values,
            hovertext=sub.apply(_hover_text, axis=1),
            hoverinfo="text",
            showlegend=False,
        ))

    center_lat = float(np.nanmedian(s["lat"].values))
    center_lon = float(np.nanmedian(s["lon"].values))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4,
        ),
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def build_timeseries_figure(pred: pd.DataFrame, anchor: pd.Timestamp, horizon: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["date"], y=pred["y"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=pred["date"], y=pred["yhat"], mode="lines", name="Prophet baseline", line=dict(dash="dash")))

    hist = pred[pred["date"] <= anchor].copy()
    anom = hist[hist["y"].notna() & ((hist["y"] > hist["yhat_upper"]) | (hist["y"] < hist["yhat_lower"]))].copy()
    fig.add_trace(go.Scatter(
        x=anom["date"], y=anom["y"], mode="markers", name="Anomaly",
        marker=dict(color="red", size=8)
    ))

    start = anchor + pd.Timedelta(days=1)
    end = anchor + pd.Timedelta(days=horizon)
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor="rgba(255,0,0,0.08)",
        line_width=0,
        annotation_text="Spike window (t+1 ~ t+H)",
        annotation_position="top left",
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Value",
    )
    return fig


def build_forecast_zoom_figure(fut: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if fut.empty:
        fig.update_layout(height=560)
        return fig

    fig.add_trace(go.Scatter(
        x=fut["date"], y=fut["yhat_upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
        name="Upper",
    ))
    fig.add_trace(go.Scatter(
        x=fut["date"], y=fut["yhat_lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,0,0,0.10)",
        showlegend=False, hoverinfo="skip",
        name="Lower",
    ))

    fig.add_trace(go.Scatter(
        x=fut["date"], y=fut["yhat"],
        mode="lines", name="Forecast (yhat)",
    ))

    if "thr" in fut.columns and fut["thr"].notna().any():
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["thr"],
            mode="lines", name="Threshold",
            line=dict(dash="dot"),
        ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Forecast",
    )
    return fig


# =========================
# Slack Alert (Cooldown)
# =========================
def load_alert_state() -> Dict[str, str]:
    if ALERT_STATE_PATH.exists():
        try:
            return json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_alert_state(state: Dict[str, str]):
    ALERT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def can_send(site: str, alert_key: str, cooldown_hours: int, state: Dict[str, str]) -> bool:
    k = f"{site}::{alert_key}"
    last = state.get(k)
    if not last:
        return True
    try:
        last_dt = dt.datetime.fromisoformat(last)
    except Exception:
        return True
    return (dt.datetime.now() - last_dt) >= dt.timedelta(hours=cooldown_hours)


def mark_sent(site: str, alert_key: str, state: Dict[str, str]):
    k = f"{site}::{alert_key}"
    state[k] = dt.datetime.now().isoformat(timespec="seconds")


def send_slack(webhook_url: str, text: str) -> Tuple[bool, str]:
    if not _HAS_REQUESTS:
        return False, "requests 미설치(pip install requests)"
    if not webhook_url:
        return False, "Slack webhook URL이 비어있습니다."
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        if 200 <= r.status_code < 300:
            return True, "OK"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


# =========================
# Pages
# =========================
def render_page1(snap: pd.DataFrame, anchor: pd.Timestamp):
    st.title("Page 1. Situation Overview & Map")
    st.caption("목표: ‘지금 어디를 봐야 하는지’ 관측소를 선별합니다. (시계열/정책메시지 없음)")

    high_n = int((snap["state_risk"] == "High").sum())
    med_n = int((snap["state_risk"] == "Medium").sum())
    spike_n = int((snap["spike_risk"].isin(["Watch", "Warn"])).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("State Risk High (오늘)", high_n)
    c2.metric("State Risk Medium (오늘)", med_n)
    c3.metric("Spike Risk Warning (7일 내)", spike_n)

    left, right = st.columns([3.2, 1.2], gap="large")

    selected_site = None
    with left:
        st.subheader("Map")
        fig = build_map_figure(snap)
        try:
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
            )
            if event and getattr(event, "selection", None) and event.selection.get("points"):
                p0 = event.selection["points"][0]
                cd = p0.get("customdata", None)
                if cd is not None:
                    if isinstance(cd, (list, tuple)) and len(cd) > 0:
                        selected_site = cd[0] if not isinstance(cd[0], (list, tuple)) else cd[0][0]
                    else:
                        selected_site = cd
        except Exception:
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("관측소 선택")
        site_list = snap["site"].astype(str).sort_values().tolist()
        if not site_list:
            st.error("관측소 목록이 비어 있습니다.")
            return

        default_site = st.session_state.get("selected_site", site_list[0])
        if default_site not in site_list:
            default_site = site_list[0]

        sel = st.selectbox("볼 관측소", site_list, index=site_list.index(default_site))
        if selected_site:
            sel = str(selected_site)

        st.session_state["selected_site"] = sel

        row = snap[snap["site"].astype(str) == str(sel)].iloc[0]
        st.markdown("#### 요약 패널")
        st.write(f"**{row['site']}**")
        st.write(f"- Cluster: **{row.get('cluster_3name','-')}**")
        st.write(f"- 오늘 State Risk: **{STATE_RISK_ICON.get(row['state_risk'],'')} {row['state_risk']}**")
        st.write(f"- 예측 임계치 경고(7d): **{SPIKE_RISK_ICON.get(row['spike_risk'],'-')}**")
        st.write(f"- 예측 초과일수: **{int(row.get('spike_exceed_days',0))}일**")
        if str(row.get("exceed_tplus", "")):
            st.write(f"- 초과 예상(t+): **{row.get('exceed_tplus')}**")

        st.divider()

        # ✅ 예측 임계치 경고 사이트 목록(전체) + "초과 예상 t+몇일"
        warn_df = snap[snap["spike_risk"].isin(["Warn", "Watch"])].copy()
        st.markdown("#### 7일 임계치 경고 사이트 목록")

        if warn_df.empty:
            st.info("예측 구간(t+1~t+H)에서 임계치 초과 가능 사이트가 없습니다.")
        else:
            warn_df["spike_sev"] = warn_df["spike_risk"].map({"Warn": 2, "Watch": 1}).fillna(0).astype(int)
            warn_df = warn_df.sort_values(
                ["spike_sev", "spike_exceed_days"],
                ascending=[False, False],
            )

            pick = st.selectbox(
                "경고 사이트 빠른 선택",
                warn_df["site"].astype(str).tolist(),
                index=0,
                key="warn_site_pick",
            )
            if st.button("선택한 경고 사이트 자세히 보기 → Page 2", use_container_width=True):
                set_query_params_safe(page="site", site=str(pick))
                st.session_state["selected_site"] = str(pick)
                st.rerun()

            show_cols = [
                "site", "cluster_3name", "state_risk",
                "spike_risk", "spike_exceed_days",
                "exceed_tplus",
                "max_yhat_upper_7d", "max_thr_7d",
            ]
            show_cols = [c for c in show_cols if c in warn_df.columns]
            st.dataframe(
                warn_df[show_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        if st.button("자세히 보기 → Page 2", use_container_width=True):
            set_query_params_safe(page="site", site=sel)
            st.rerun()


def render_page2(
    df_all: pd.DataFrame,
    site: str,
    target: str,
    anchor: pd.Timestamp,
    horizon: int,
    interval_width: float,
    weather_fc: Optional[pd.DataFrame],
    thr_config: Dict,
):
    st.title("Page 2. Site Analysis & Action Support")
    st.caption("목표: 선택된 1개 관측소를 분석하고 조치 여부를 판단합니다.")

    df_site = df_all[df_all["site"].astype(str) == str(site)].copy().sort_values("date")
    if df_site.empty:
        st.error("관측소 데이터를 찾지 못했습니다.")
        return

    site_thr, season_thr_df = compute_threshold_tables(df_all, target, thr_config["site_q"], thr_config["season_q"])

    today_row = df_site[df_site["date"] == anchor]
    if today_row.empty:
        today_row = df_site[df_site["date"] <= anchor].tail(1)

    y_today = float(pd.to_numeric(today_row[target].iloc[0], errors="coerce")) if not today_row.empty else np.nan
    thr_today = threshold_for(
        site, anchor, target,
        thr_config["fixed_value"], thr_config["use_fixed"],
        site_thr, thr_config["use_site"],
        season_thr_df, thr_config["use_season"]
    )
    state_risk = calc_state_risk_today(y_today, thr_today, medium_ratio=thr_config["medium_ratio"])

    pred = prophet_predict_site(df_site, target, anchor, horizon, interval_width, weather_fc)

    fut = pred[(pred["date"] > anchor) & (pred["date"] <= anchor + pd.Timedelta(days=horizon))].copy()
    if not fut.empty:
        fut["thr"] = [
            threshold_for(
                site, d_, target,
                thr_config["fixed_value"], thr_config["use_fixed"],
                site_thr, thr_config["use_site"],
                season_thr_df, thr_config["use_season"]
            )
            for d_ in fut["date"]
        ]
        fut["exceed"] = (fut["yhat_upper"] > fut["thr"])
        exceed_days = int(fut["exceed"].sum())
    else:
        exceed_days = 0

    spike_level = "None"
    if exceed_days >= thr_config["warn_days"]:
        spike_level = "Warn"
    elif exceed_days >= thr_config["watch_days"]:
        spike_level = "Watch"

    meta = df_site.tail(1).iloc[0]
    cluster_3 = str(meta.get("cluster_3name", "Risk"))

    a, b, c, d = st.columns(4)
    a.metric("관측소", site)
    b.metric("클러스터", cluster_3)
    c.metric("오늘 State Risk", f"{STATE_RISK_ICON.get(state_risk,'')} {state_risk}")
    d.metric("Spike Risk (7d 최고)", SPIKE_RISK_ICON.get(spike_level, spike_level))

    st.divider()

    st.subheader("통합 시계열 (Actual + Prophet baseline + Anomaly + Spike window)")
    st.plotly_chart(build_timeseries_figure(pred, anchor=anchor, horizon=horizon), use_container_width=True)

    st.subheader("미래 예측(라인) 확대 보기")
    st.caption("t+1 ~ t+H 구간 예측선(yhat)과 예측구간을 확대 표시합니다.")
    st.plotly_chart(build_forecast_zoom_figure(fut), use_container_width=True)

    st.subheader("해석 + 대응 가이드")
    recent = pred[(pred["date"] <= anchor) & (pred["date"] > anchor - pd.Timedelta(days=30))].copy()
    anom_cnt = int((recent["y"].notna() & ((recent["y"] > recent["yhat_upper"]) | (recent["y"] < recent["yhat_lower"]))).sum())

    if fut.empty:
        risk_msg = "미래 구간 예측을 산출하지 못했습니다."
    else:
        max_upper = float(np.nanmax(fut["yhat_upper"].values))
        max_thr = float(np.nanmax(fut["thr"].values)) if "thr" in fut.columns else np.nan
        risk_msg = (
            f"향후 {horizon}일 중 **{exceed_days}일**이 ‘예측구간 상단(yhat_upper)’ 기준 임계 초과 가능성이 있습니다. "
            f"(max upper={max_upper:.5g}, max thr={max_thr:.5g})"
        )

    st.write(f"최근 30일 이상탐지(예측구간 이탈) **{anom_cnt}회**. {risk_msg}")

    st.divider()
    if st.button("← Page 1로 돌아가기", use_container_width=True):
        set_query_params_safe(page="overview")
        st.rerun()


# =========================
# Main
# =========================
def main():
    with st.sidebar:
        st.header("설정")

        data_path = st.text_input("대기 데이터 경로", value=DEFAULT_DATA_PATH)
        weather_fc_path = st.text_input("미래 7일 기상 경로(optional)", value=DEFAULT_WEATHER_FC_PATH)

        if not _HAS_PROPHET:
            st.error("Prophet 미설치: `pip install prophet` 필요")
        if not _HAS_REQUESTS:
            st.warning("Slack 발송을 쓰려면 `pip install requests` 필요")

        st.divider()

        target = st.selectbox("대기지표(Target)", TARGET_OPTIONS, index=TARGET_OPTIONS.index("co_mean"))
        st.session_state["target"] = target

        st.subheader("임계치 정책(기본=사이트 P95)")
        use_fixed = st.checkbox("고정값 적용", value=False)
        fixed_default = 0.07 if target == "o3_mean" else 0.0
        fixed_value = st.number_input("고정값", value=float(fixed_default), step=0.001, format="%.5f")

        use_site = st.checkbox("사이트 분위수 적용", value=True)
        site_q = st.selectbox("사이트 분위수", options=[0.90, 0.95], index=1)

        use_season = st.checkbox("계절(월) 분위수 적용", value=False)
        season_q = st.selectbox("계절(월) 분위수", options=[0.90, 0.95], index=1)

        medium_ratio = st.slider("State Risk Medium 기준(임계치 대비 비율)", 0.70, 0.99, 0.90, 0.01)

        st.subheader("Spike Risk 규칙")
        horizon = st.number_input("예측 기간(일)", 1, 14, 7, 1)
        interval_width = st.selectbox("Prophet 예측구간 폭(interval_width)", options=[0.80, 0.90, 0.95], index=1)
        warn_days = st.number_input("Warn 기준(초과일수 ≥)", 1, 7, 2, 1)
        watch_days = st.number_input("Watch 기준(초과일수 ≥)", 1, 7, 1, 1)

        st.divider()
        st.subheader("Slack 알람")
        webhook_url = st.text_input("Slack Webhook URL", value=safe_get_secret("SLACK_WEBHOOK_URL", ""))
        cooldown_hours = st.number_input("알람 쿨다운(시간)", 1, 72, 6, 1)
        include_watch = st.checkbox("Watch도 알람 발송", value=False)

        st.subheader("Slack 테스트")
        test_msg = st.text_input("테스트 메시지", value="Slack 연결 테스트: OK")
        if st.button("테스트 알람 보내기", use_container_width=True):
            ok, msg = send_slack(webhook_url, test_msg)
            if ok:
                st.success("테스트 발송 성공")
            else:
                st.error(f"테스트 발송 실패: {msg}")

        st.divider()
        qp = get_query_params()
        page = qp.get("page", ["overview"])[0]
        nav = st.radio(
            "페이지",
            ["overview", "site"],
            index=0 if page == "overview" else 1,
            format_func=lambda x: "Page 1 (Overview & Map)" if x == "overview" else "Page 2 (Site Analysis)",
        )

    # Load
    df_raw = load_pollution_data(data_path)
    df_all = attach_site_clusters(df_raw)

    # 오늘(기준일) = 데이터 마지막 일자
    anchor = pd.to_datetime(df_all["date"].max())
    st.sidebar.info(f"오늘(기준일): {fmt_date(anchor)} (데이터 마지막 일자)")

    weather_fc = load_weather_forecast(weather_fc_path)

    thr_config = dict(
        use_fixed=bool(use_fixed),
        fixed_value=float(fixed_value),
        use_site=bool(use_site),
        site_q=float(site_q),
        use_season=bool(use_season),
        season_q=float(season_q),
        medium_ratio=float(medium_ratio),
        warn_days=int(warn_days),
        watch_days=int(watch_days),
    )

    with st.spinner("사이트별 Spike Risk(Prophet) 산출 중..."):
        snap = compute_spike_risk_all_sites(
            df=df_all,
            target=target,
            anchor=anchor,
            horizon=int(horizon),
            interval_width=float(interval_width),
            fixed_value=float(fixed_value),
            use_fixed=bool(use_fixed),
            use_site=bool(use_site),
            site_q=float(site_q),
            use_season=bool(use_season),
            season_q=float(season_q),
            warn_days=int(warn_days),
            watch_days=int(watch_days),
            weather_fc=weather_fc,
            medium_ratio=float(medium_ratio),
        )

    # Slack 발송(룰 기반)
    st.sidebar.subheader("Slack 발송 실행")
    if st.sidebar.button("Slack 알람 발송", use_container_width=True):
        if not webhook_url:
            st.sidebar.error("Webhook URL을 입력하거나 secrets.toml에 설정하세요.")
        else:
            state = load_alert_state()
            sent = 0
            failed = 0

            targets = snap.copy()
            targets["send_flag"] = False
            targets.loc[targets["state_risk"] == "High", "send_flag"] = True
            targets.loc[targets["spike_risk"] == "Warn", "send_flag"] = True
            if include_watch:
                targets.loc[targets["spike_risk"] == "Watch", "send_flag"] = True
            targets = targets[targets["send_flag"] == True].copy()

            for _, r in targets.iterrows():
                site = str(r["site"])
                alert_key = f"{fmt_date(anchor)}::{target}::{r['state_risk']}::{r['spike_risk']}"
                if not can_send(site, alert_key, int(cooldown_hours), state):
                    continue

                tplus = r.get("exceed_tplus", "")
                tplus_line = f"\n- exceed(t+): {tplus}" if isinstance(tplus, str) and tplus else ""

                text = (
                    f"[AirQ Alert] 기준일={fmt_date(anchor)} | target={target}\n"
                    f"- site: {site}\n"
                    f"- cluster: {r.get('cluster_3name','-')}\n"
                    f"- today: {STATE_RISK_ICON.get(r['state_risk'],'')} {r['state_risk']} "
                    f"(y={r['today_y']:.5g}, thr={r['today_thr']:.5g})\n"
                    f"- spike(7d): {SPIKE_RISK_ICON.get(r['spike_risk'],'-')} | exceed_days={int(r.get('spike_exceed_days',0))}"
                    f"{tplus_line}\n"
                    f"- max_yhat_upper_7d={r.get('max_yhat_upper_7d',np.nan):.5g} | max_thr_7d={r.get('max_thr_7d',np.nan):.5g}"
                )

                ok, _msg = send_slack(webhook_url, text)
                if ok:
                    sent += 1
                    mark_sent(site, alert_key, state)
                else:
                    failed += 1

            save_alert_state(state)
            st.sidebar.success(f"발송 성공 {sent}건 / 실패 {failed}건 (쿨다운 {cooldown_hours}h 적용)")

    # Router
    qp = get_query_params()
    if nav == "overview":
        set_query_params_safe(page="overview")
        render_page1(snap=snap, anchor=anchor)
    else:
        site = qp.get("site", [None])[0] or st.session_state.get("selected_site", None)
        if not site:
            st.warning("Page 1에서 관측소를 먼저 선택하세요.")
            return

        set_query_params_safe(page="site", site=site)
        render_page2(
            df_all=df_all,
            site=str(site),
            target=target,
            anchor=anchor,
            horizon=int(horizon),
            interval_width=float(interval_width),
            weather_fc=weather_fc,
            thr_config=thr_config,
        )


if __name__ == "__main__":
    main()
