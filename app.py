import math
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="テーマ×株スコアランキング", layout="wide")
st.title("テーマ×株 スコアランキング（フィジカルAI / 防衛安保 / 半導体）")

themes = pd.read_csv("themes.csv")
themes["code"] = themes["code"].astype(str)

left, right = st.columns([1, 2])

with left:
    theme = st.selectbox("テーマ", sorted(themes["theme"].unique()))
    period = st.selectbox("期間（価格取得）", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)

    st.markdown("### スコア重み（合計は自動で正規化）")
    w_chg = st.slider("上昇率（chg%）", 0.0, 1.0, 0.55, 0.05)
    w_vol = st.slider("出来高（log）", 0.0, 1.0, 0.25, 0.05)
    w_high = st.slider("高値圏（from_high%）", 0.0, 1.0, 0.20, 0.05)

    min_vol = st.number_input("出来高の下限（任意）", min_value=0, value=0, step=10000)
    show_top = st.slider("表示件数", 5, 50, 20, 1)

codes_df = themes.loc[themes["theme"] == theme, ["code", "name"]].copy()

@st.cache_data(ttl=60 * 30)
def fetch_quotes(code_name_pairs, period):
    rows = []
    for code, name in code_name_pairs:
        try:
            t = yf.Ticker(f"{code}.T")  # 東証
            hist = t.history(period=period)
            if hist is None or len(hist) < 2:
                continue

            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            chg = (last / prev - 1.0) * 100.0
            vol = float(hist["Volume"].iloc[-1])
            high = float(hist["High"].max())
            from_high = (last / high - 1.0) * 100.0 if high else 0.0

            rows.append({
                "code": code,
                "name": name,
                "close": last,
                "chg%": chg,
                "volume": vol,
                "from_high%": from_high,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn = float(s.min())
    mx = float(s.max())
    if math.isclose(mx, mn):
        # 全部同じ値のときは中間点に寄せる
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

pairs = list(zip(codes_df["code"].tolist(), codes_df["name"].tolist()))
df = fetch_quotes(pairs, period)

with right:
    st.subheader(f"テーマ：{theme}")

    if df.empty:
        st.warning("データが取得できませんでした（無料データ側の制限や一時不調の可能性があります）。")
        st.stop()

    if min_vol > 0:
        df = df[df["volume"] >= float(min_vol)]

    if df.empty:
        st.warning("出来高フィルタ後にデータが空になりました。条件を緩めてください。")
        st.stop()

    # --- スコア計算（テーマ内で相対評価 0〜100） ---
    # 出来高は log で圧縮（極端な大型株の影響を抑える）
    df["log_volume"] = df["volume"].apply(lambda x: math.log10(x + 1.0))

    # 正規化（0〜1）
    n_chg = minmax(df["chg%"])                  # 高いほど良い
    n_vol = minmax(df["log_volume"])            # 高いほど良い
    n_high = 1.0 - minmax(df["from_high%"])     # 0に近いほど良い（=高値圏）→ 反転

    # 重みは合計1に正規化
    w_sum = w_chg + w_vol + w_high
    if w_sum == 0:
        w_chg_n, w_vol_n, w_high_n = 1.0, 0.0, 0.0
    else:
        w_chg_n, w_vol_n, w_high_n = w_chg / w_sum, w_vol / w_sum, w_high / w_sum

    df["score_0_100"] = (
        (w_chg_n * n_chg) +
        (w_vol_n * n_vol) +
        (w_high_n * n_high)
    ) * 100.0

    # 表示用
    out = df[["code", "name", "close", "chg%", "volume", "from_high%", "score_0_100"]].copy()
    out["close"] = out["close"].round(1)
    out["chg%"] = out["chg%"].round(2)
    out["from_high%"] = out["from_high%"].round(2)
    out["score_0_100"] = out["score_0_100"].round(1)
    out = out.sort_values("score_0_100", ascending=False).head(show_top)

    st.dataframe(out, use_container_width=True, hide_index=True)

    st.download_button(
        "CSVダウンロード（表示分）",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"{theme}_score_{period}.csv",
        mime="text/csv"
    )

st.caption("注：スコアはテーマ内の相対評価（0〜100）です。無料データ取得のため、取得できない銘柄が混ざることがあります。")
