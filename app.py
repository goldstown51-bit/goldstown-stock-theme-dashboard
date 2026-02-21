import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="テーマ×株ランキング", layout="wide")
st.title("テーマ×株ランキング（フィジカルAI / 防衛安保 / 半導体）")

themes = pd.read_csv("themes.csv")

left, right = st.columns([1, 2])
with left:
    theme = st.selectbox("テーマ", sorted(themes["theme"].unique()))
    period = st.selectbox("期間", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)
    min_vol = st.number_input("出来高の下限（目安）", min_value=0, value=0, step=10000)
    sort_key = st.selectbox("並び替え", ["chg%", "volume", "from_high%"], index=0)

codes_df = themes.loc[themes["theme"] == theme, ["code", "name"]].copy()
codes_df["code"] = codes_df["code"].astype(str)

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
            vol = int(hist["Volume"].iloc[-1])
            high = float(hist["High"].max())
            from_high = (last / high - 1.0) * 100.0 if high else 0.0

            rows.append({
                "code": code,
                "name": name,
                "close": last,
                "chg%": chg,
                "volume": vol,
                "from_high%": from_high
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

pairs = list(zip(codes_df["code"].tolist(), codes_df["name"].tolist()))
df = fetch_quotes(pairs, period)

with right:
    st.subheader(f"テーマ：{theme}")
    if df.empty:
        st.warning("データが取得できませんでした（無料データ側の制限や一時不調の可能性があります）。")
    else:
        if min_vol > 0:
            df = df[df["volume"] >= min_vol]

        ascending = False
        df = df.sort_values(sort_key, ascending=ascending)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "CSVダウンロード",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{theme}_{period}.csv",
            mime="text/csv"
        )

st.caption("注：無料データ取得のため、取得できない銘柄が混ざることがあります。")
