import math
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="テーマ株スコアボード", layout="wide")

APP_NAME = "テーマ株スコアボード"
st.title(APP_NAME)
st.caption("フィジカルAI / 防衛安保 / 半導体：テーマ内相対スコアでランキング＋テーマ指数チャート")

st.markdown("""
<style>
/* 全体 */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1180px; }
html, body, [class*="css"]  { font-family: -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN",
  "Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans JP", "Segoe UI", Roboto, sans-serif; }

/* 見出しを日本サイトっぽく */
h1 { letter-spacing: 0.02em; }
h2, h3 { border-left: 6px solid rgba(0,0,0,0.15); padding-left: .6rem; margin-top: 1.0rem; }

/* データフレームの見た目：余白・枠・ヘッダ */
div[data-testid="stDataFrame"] { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; overflow: hidden; }
div[data-testid="stDataFrame"] div[role="columnheader"] {
  font-weight: 700;
  background: rgba(0,0,0,0.03);
}

/* metricカードを少し“サイト風”に */
div[data-testid="stMetric"] { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 10px 12px; }
</style>
""", unsafe_allow_html=True)

themes = pd.read_csv("themes.csv")
themes["code"] = themes["code"].astype(str)

# -------------------------
# UI（左：設定）
# -------------------------
left, right = st.columns([1, 2], gap="large")
with left:
    theme = st.selectbox("テーマ", sorted(themes["theme"].unique()))
    show_top = st.slider("表示件数", 5, 50, 20, 1)
    min_vol = st.number_input("出来高の下限（任意）", min_value=0, value=0, step=10000)

    st.divider()
    st.markdown("### スコア重み（合計は自動で正規化）")
    # “PV向き”＝トレンド継続寄りの配分を初期値に
    w_mom = st.slider("モメンタム（1日）", 0.0, 1.0, 0.25, 0.05)
    w_vs = st.slider("出来高急増（vs 20日平均）", 0.0, 1.0, 0.30, 0.05)
    w_ma = st.slider("トレンド（20日MA乖離）", 0.0, 1.0, 0.30, 0.05)
    w_high = st.slider("高値圏（期間高値からの乖離）", 0.0, 1.0, 0.15, 0.05)

    st.divider()
    index_lookback = st.selectbox("テーマ指数の表示期間", ["1mo", "3mo", "6mo"], index=1)
    st.info("スコアは“テーマ内の相対評価(0〜100)”です。テーマ間比較にはテーマ指数を使うのがオススメ。")

codes_df = themes.loc[themes["theme"] == theme, ["code", "name"]].copy()
pairs = list(zip(codes_df["code"].tolist(), codes_df["name"].tolist()))

# -------------------------
# 取得（キャッシュ）
# -------------------------
@st.cache_data(ttl=60 * 30)
def fetch_histories(code_name_pairs, period="6mo"):
    """
    指標計算のために共通で6ヶ月分程度を取る。
    """
    out = {}
    for code, name in code_name_pairs:
        try:
            t = yf.Ticker(f"{code}.T")
            hist = t.history(period=period, auto_adjust=False)
            if hist is None or hist.empty or len(hist) < 30:
                continue
            hist = hist.copy()
            hist["code"] = code
            hist["name"] = name
            out[code] = hist
        except Exception:
            continue
    return out

hists = fetch_histories(pairs, period="6mo")

# -------------------------
# ユーティリティ
# -------------------------
def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn = float(s.min())
    mx = float(s.max())
    if math.isclose(mx, mn):
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def safe_pct(a, b):
    if b == 0 or pd.isna(b) or pd.isna(a):
        return float("nan")
    return (a / b - 1.0) * 100.0

# -------------------------
# 指標テーブル作成（最新日ベース）
# -------------------------
rows = []
for code, hist in hists.items():
    try:
        close = hist["Close"].astype(float)
        high = hist["High"].astype(float)
        vol = hist["Volume"].astype(float)

        last = float(close.iloc[-1])
        prev = float(close.iloc[-2])

        chg1d = safe_pct(last, prev)

        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
        ma20_dev = safe_pct(last, ma20)  # 20MA乖離%

        # 出来高急増：今日 / 20日平均
        vol20 = float(vol.rolling(20).mean().iloc[-1])
        vol_surge = (float(vol.iloc[-1]) / vol20) if vol20 and not pd.isna(vol20) else float("nan")
        log_vol_surge = math.log10(vol_surge + 1.0) if not pd.isna(vol_surge) else float("nan")

        # 期間高値からの乖離（6moの範囲で高値）
        period_high = float(high.max())
        from_high = safe_pct(last, period_high)  # 0に近いほど高値圏

        # ざっくりリターン（営業日換算）
        def ret_n(n):
            if len(close) <= n:
                return float("nan")
            return safe_pct(last, float(close.iloc[-(n+1)]))

        ret_1w = ret_n(5)
        ret_1m = ret_n(21)
        ret_3m = ret_n(63)

        rows.append({
            "code": code,
            "name": str(hist["name"].iloc[-1]),
            "close": last,
            "chg1d%": chg1d,
            "vol": float(vol.iloc[-1]),
            "vol_surge": vol_surge,
            "ma20_dev%": ma20_dev,
            "from_high%": from_high,
            "ret_1w%": ret_1w,
            "ret_1m%": ret_1m,
            "ret_3m%": ret_3m,
            "ma20": ma20,
            "ma50": ma50,
        })
    except Exception:
        continue

df = pd.DataFrame(rows)

with right:
    if df.empty:
        st.error("データが取得できませんでした。少し時間を置いて再読み込みしてみてください。")
        st.stop()

    if min_vol > 0:
        df = df[df["vol"] >= float(min_vol)]

    if df.empty:
        st.warning("出来高フィルタ後に銘柄が0件になりました。条件を緩めてください。")
        st.stop()

    # -------------------------
    # スコア計算（テーマ内相対：0〜100）
    # -------------------------
    # “良い方向”に揃える：
    #  - chg1d%：高いほど良い
    #  - vol_surge（log）：高いほど良い
    #  - ma20_dev%：高いほど良い（上昇トレンド）
    #  - from_high%：0に近いほど良い → 反転
    n_mom = minmax(df["chg1d%"].fillna(df["chg1d%"].median()))
    n_vs = minmax(df["vol_surge"].apply(lambda x: math.log10(x + 1.0) if pd.notna(x) else float("nan")).fillna(0.0))
    n_ma = minmax(df["ma20_dev%"].fillna(df["ma20_dev%"].median()))
    n_high = 1.0 - minmax(df["from_high%"].fillna(df["from_high%"].median()))  # 反転

    w_sum = w_mom + w_vs + w_ma + w_high
    if w_sum == 0:
        w_mom_n, w_vs_n, w_ma_n, w_high_n = 1.0, 0.0, 0.0, 0.0
    else:
        w_mom_n, w_vs_n, w_ma_n, w_high_n = w_mom / w_sum, w_vs / w_sum, w_ma / w_sum, w_high / w_sum

    df["score"] = (w_mom_n * n_mom + w_vs_n * n_vs + w_ma_n * n_ma + w_high_n * n_high) * 100.0

    # 表示整形
    view = df.copy()
    view["close"] = view["close"].round(1)
    view["chg1d%"] = view["chg1d%"].round(2)
    view["vol"] = view["vol"].astype(int)
    view["vol_surge"] = view["vol_surge"].round(2)
    view["ma20_dev%"] = view["ma20_dev%"].round(2)
    view["from_high%"] = view["from_high%"].round(2)
    view["ret_1w%"] = view["ret_1w%"].round(2)
    view["ret_1m%"] = view["ret_1m%"].round(2)
    view["ret_3m%"] = view["ret_3m%"].round(2)
    view["score"] = view["score"].round(1)

    view = view.sort_values("score", ascending=False).head(show_top)

    # -------------------------
    # 見た目：上位カード
    # -------------------------
    top1 = view.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("注目銘柄", f'{top1["code"]} {top1["name"]}')
    c2.metric("スコア", f'{top1["score"]}')
    c3.metric("前日比", f'{top1["chg1d%"]}%')
    c4.metric("出来高急増", f'{top1["vol_surge"]}x')

    st.subheader(f"スコアランキング：{theme}")
table = view[["code","name","score","close","chg1d%","vol","vol_surge","ma20_dev%","from_high%","ret_1w%","ret_1m%","ret_3m%"]].copy()

# 日本語ヘッダに
table = table.rename(columns={
    "code":"コード",
    "name":"銘柄",
    "score":"スコア",
    "close":"終値",
    "chg1d%":"前日比(%)",
    "vol":"出来高",
    "vol_surge":"出来高急増(x)",
    "ma20_dev%":"20MA乖離(%)",
    "from_high%":"高値乖離(%)",
    "ret_1w%":"1週(%)",
    "ret_1m%":"1ヶ月(%)",
    "ret_3m%":"3ヶ月(%)",
})

# まず表示用に整形（桁区切り・小数）
def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return ""

def fmt_f1(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return ""

def fmt_f2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""

# 右寄せ＋Zebra＋上昇/下落の控えめ強調
def zebra(row_idx):
    return "background-color: rgba(0,0,0,0.02)" if row_idx % 2 == 1 else ""

def color_pct(v):
    try:
        v = float(v)
    except Exception:
        return ""
    if v > 0:
        return "color: #d32f2f; font-weight: 600;"   # 上昇＝赤（日本株っぽさ）
    if v < 0:
        return "color: #1976d2; font-weight: 600;"   # 下落＝青
    return "color: rgba(0,0,0,0.75);"

styler = (
    table.style
    .apply(lambda s: [zebra(i) for i in range(len(s))], axis=0)
    .set_properties(**{"text-align":"right"})
    .set_properties(subset=["コード","銘柄"], **{"text-align":"left"})
    .format({
        "スコア": fmt_f1,
        "終値": fmt_f1,
        "前日比(%)": fmt_f2,
        "出来高": fmt_int,
        "出来高急増(x)": fmt_f2,
        "20MA乖離(%)": fmt_f2,
        "高値乖離(%)": fmt_f2,
        "1週(%)": fmt_f2,
        "1ヶ月(%)": fmt_f2,
        "3ヶ月(%)": fmt_f2,
    })
    .applymap(color_pct, subset=["前日比(%)","1週(%)","1ヶ月(%)","3ヶ月(%)","20MA乖離(%)"])
)

st.dataframe(styler, use_container_width=True, hide_index=True)

st.download_button(
  "CSVダウンロード（表示分）",
  view.to_csv(index=False).encode("utf-8"),
  file_name=f"{theme}_score.csv",
  mime="text/csv"
)

# -------------------------
# テーマ指数（等ウェイトの簡易インデックス）
# -------------------------
st.subheader("テーマ指数（等ウェイト・指数化）")

# lookbackの日数（ざっくり営業日）
lookback_map = {"1mo": 25, "3mo": 70, "6mo": 140}
lb = lookback_map.get(index_lookback, 70)

closes = []
for code, hist in hists.items():
s = hist["Close"].astype(float)
if len(s) < 30:
    continue
s = s.tail(lb)
s = s / float(s.iloc[0]) * 100.0  # 先頭を100に
closes.append(s.rename(code))

if len(closes) >= 2:
    idx = pd.concat(closes, axis=1).dropna(how="all")
    theme_index = idx.mean(axis=1).rename("ThemeIndex")
    st.line_chart(theme_index)
    idx_chg = safe_pct(float(theme_index.iloc[-1]), float(theme_index.iloc[0]))
    st.caption(f"期間（{index_lookback}）のテーマ指数変化：{idx_chg:.2f}%（先頭=100の等ウェイト指数）")
else:
    st.info("テーマ指数を作るのに十分な銘柄データがありませんでした（取得失敗が多い場合に起きます）。")

# -------------------------
# 銘柄詳細（クリック代わりに選択式）
# -------------------------
st.subheader("銘柄詳細（チャート）")
pick = st.selectbox("銘柄を選択", view["code"].tolist(), index=0)
if pick in hists:
    hist = hists[pick].copy()
    hist = hist.tail(lb)
    chart = hist["Close"].astype(float)
    st.line_chart(chart)

    # 追加の小メトリクス
    last = float(hist["Close"].iloc[-1])
    ma20 = float(hist["Close"].astype(float).rolling(20).mean().iloc[-1])
    vol = float(hist["Volume"].astype(float).iloc[-1])
    vol20 = float(hist["Volume"].astype(float).rolling(20).mean().iloc[-1])
    vs = (vol / vol20) if vol20 else float("nan")

    m1, m2, m3 = st.columns(3)
    m1.metric("終値", f"{last:.1f}")
    m2.metric("20MA乖離", f"{safe_pct(last, ma20):.2f}%")
    m3.metric("出来高急増", f"{vs:.2f}x")

st.caption("注：無料データ取得のため、取得できない銘柄が混ざることがあります。")
