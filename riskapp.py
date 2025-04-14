"""
BMIX Portfolio Risk Attribution – v2.3‑full
-------------------------------------------------
• Waterfall charts for Volatility, VaR 95/99, CVaR 95/99
  – each includes a Diversification bar so totals reconcile
• Historical‑simulation component VaR/CVaR per position
  (Δyield × position → daily P/L → VaR/CVaR)
• Expanded AgGrid output (Vol, VaR & CVaR columns)
• Instrument master list fully embedded – no external CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="📈 BMIX Portfolio Risk Attribution", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def load_historical_data(excel_file: str) -> pd.DataFrame:
    """Read every sheet, concatenate, drop weekends."""
    try:
        xl = pd.ExcelFile(excel_file)
        frames = [xl.parse(s, index_col="date", parse_dates=True) for s in xl.sheet_names]
        df = pd.concat(frames, axis=1)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        return df[~df.index.dayofweek.isin([5, 6])]
    except Exception as exc:
        st.error(f"❌ Error loading Excel: {exc}")
        return pd.DataFrame()


def adjust_time_zones(df: pd.DataFrame, instr_to_country: dict) -> pd.DataFrame:
    """Shift non‑Asia instruments forward by one business day."""
    non_lag = ["JP", "AU", "SK", "CH"]
    countries = pd.Series([instr_to_country.get(c, "Other") for c in df.columns], index=df.columns)
    lag_cols = countries[~countries.isin(non_lag)].index
    out = df.copy()
    if len(lag_cols):
        out[lag_cols] = out[lag_cols].shift(1)
    return out.dropna()


def calculate_daily_changes_in_bps(df):  # Δyield × 100 → bps
    return df.diff().dropna() * 100


def fallback_mx_ois_data(changes):
    ois_map = {
        "MX 2Y Swap OIS": "MX 2Y Swap",
        "MX 5Y Swap OIS": "MX 5Y Swap",
        "MX 10Y Swap OIS": "MX 10Y Swap",
    }
    for ois, vanilla in ois_map.items():
        if ois in changes.columns and vanilla in changes.columns:
            changes[ois] = changes[ois].fillna(changes[vanilla])
    return changes


def calculate_covariance_matrix(changes, lookback):
    recent = changes.tail(lookback)
    return recent.cov() * 252 if not recent.empty else pd.DataFrame()


def guess_country(name: str) -> str:
    for code in [
        "AU","US","DE","UK","IT","CA","JP","CH",
        "BR","MX","SA","CZ","PO","SK","NZ","SW","NK"
    ]:
        if code in name:
            return code
    return "Other"


def make_waterfall(df, name_col, value_col, title):
    df = df.copy().sort_values(value_col, ascending=False)
    fig = go.Figure(go.Waterfall(
        x=df[name_col].tolist() + ["Total"],
        y=df[value_col].tolist() + [df[value_col].sum()],
        measure=["relative"] * len(df) + ["total"],
        decreasing={"marker": {"color": "#ff6961"}},
        increasing={"marker": {"color": "#77dd77"}},
        totals={"marker": {"color": "#6fa8dc"}},
    ))
    fig.update_layout(title=title, showlegend=False)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Full instrument master list (every line from original mapping)
# ──────────────────────────────────────────────────────────────────────────────
instrument_records = [
    # ---------- Futures ----------
    {"Ticker": "GACGB2 Index",  "Instrument": "AU 3Y Future",        "Portfolio": "DM"},
    {"Ticker": "GACGB10 Index", "Instrument": "AU 10Y Future",       "Portfolio": "DM"},
    {"Ticker": "TUAFWD Comdty", "Instrument": "US 2Y Future",        "Portfolio": "DM"},
    {"Ticker": "FVAFWD Comdty", "Instrument": "US 5Y Future",        "Portfolio": "DM"},
    {"Ticker": "TYAFWD Comdty", "Instrument": "US 10Y Future",       "Portfolio": "DM"},
    {"Ticker": "UXYAFWD Comdty","Instrument": "US 10Y Ultra Future", "Portfolio": "DM"},
    {"Ticker": "WNAFWD Comdty", "Instrument": "US 30Y Future",       "Portfolio": "DM"},
    {"Ticker": "DUAFWD Comdty", "Instrument": "DE 2Y Future",        "Portfolio": "DM"},
    {"Ticker": "OEAFWD Comdty", "Instrument": "DE 5Y Future",        "Portfolio": "DM"},
    {"Ticker": "RXAFWD Comdty", "Instrument": "DE 10Y Future",       "Portfolio": "DM"},
    {"Ticker": "GAFWD Comdty",  "Instrument": "UK 10Y Future",       "Portfolio": "DM"},
    {"Ticker": "IKAFWD Comdty", "Instrument": "IT 10Y Future",       "Portfolio": "DM"},
    {"Ticker": "CNAFWD Comdty", "Instrument": "CA 10Y Future",       "Portfolio": "DM"},
    {"Ticker": "JBAFWD Comdty", "Instrument": "JP 10Y Future",       "Portfolio": "DM"},
    # ---------- Swaps (all tenors; DM + EM) ----------
    # 1Y
    {"Ticker": "CCSWNI1 Curncy","Instrument": "CH 1Y Swap",          "Portfolio": "EM"},
    # 2Y
    {"Ticker": "ADSW2 Curncy",  "Instrument": "AU 2Y Swap",          "Portfolio": "DM"},
    {"Ticker": "CDSO2 Curncy",  "Instrument": "CA 2Y Swap",          "Portfolio": "DM"},
    {"Ticker": "USSW2 Curncy",  "Instrument": "US 2Y Swap",          "Portfolio": "DM"},
    {"Ticker": "EUSA2 Curncy",  "Instrument": "DE 2Y Swap",          "Portfolio": "DM"},
    {"Ticker": "BPSWS2 BGN Curncy","Instrument":"UK 2Y Swap",        "Portfolio": "DM"},
    {"Ticker": "NDSWAP2 BGN Curncy","Instrument":"NZ 2Y Swap",       "Portfolio": "DM"},
    {"Ticker": "I39302Y Index", "Instrument": "BR 2Y Swap",          "Portfolio": "EM"},
    {"Ticker": "MPSW2B BGN Curncy","Instrument":"MX 2Y Swap",        "Portfolio": "EM"},
    {"Ticker": "MPSWF2B Curncy","Instrument": "MX 2Y Swap OIS",      "Portfolio": "EM"},
    {"Ticker": "SAFR1I2 BGN Curncy","Instrument":"SA 2Y Swap",       "Portfolio": "EM"},
    {"Ticker": "CKSW2 BGN Curncy","Instrument":"CZ 2Y Swap",         "Portfolio": "EM"},
    {"Ticker": "PZSW2 BGN Curncy","Instrument":"PO 2Y Swap",         "Portfolio": "EM"},
    {"Ticker": "KWSWNI2 BGN Curncy","Instrument":"SK 2Y Swap",       "Portfolio": "EM"},
    {"Ticker": "CCSWNI2 CMPN Curncy","Instrument":"CH 2Y Swap",      "Portfolio": "EM"},
    # 5Y
    {"Ticker": "ADSW5 Curncy",  "Instrument": "AU 5Y Swap",          "Portfolio": "DM"},
    {"Ticker": "CDSO5 Curncy",  "Instrument": "CA 5Y Swap",          "Portfolio": "DM"},
    {"Ticker": "USSW5 Curncy",  "Instrument": "US 5Y Swap",          "Portfolio": "DM"},
    {"Ticker": "EUSA5 Curncy",  "Instrument": "DE 5Y Swap",          "Portfolio": "DM"},
    {"Ticker": "BPSWS5 BGN Curncy","Instrument":"UK 5Y Swap",        "Portfolio": "DM"},
    {"Ticker": "NDSWAP5 BGN Curncy","Instrument":"NZ 5Y Swap",       "Portfolio": "DM"},
    {"Ticker": "I39305Y Index", "Instrument": "BR 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "MPSW5E Curncy", "Instrument": "MX 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "MPSWF5E Curncy","Instrument": "MX 5Y Swap OIS",      "Portfolio": "EM"},
    {"Ticker": "SASW5 Curncy",  "Instrument": "SA 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "CKSW5 Curncy",  "Instrument": "CZ 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "PZSW5 Curncy",  "Instrument": "PO 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "KWSWNI5 Curncy","Instrument": "SK 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "CCSWNI5 Curncy","Instrument": "CH 5Y Swap",          "Portfolio": "EM"},
    {"Ticker": "JYSO5 Curncy",  "Instrument": "JP 5Y Swap",          "Portfolio": "DM"},
    # 10Y
    {"Ticker": "ADSW10 Curncy", "Instrument": "AU 10Y Swap",         "Portfolio": "DM"},
    {"Ticker": "CDSW10 Curncy", "Instrument": "CA 10Y Swap",         "Portfolio": "DM"},
    {"Ticker": "USSW10 Curncy", "Instrument": "US 10Y Swap",         "Portfolio": "DM"},
    {"Ticker": "EUSA10 Curncy", "Instrument": "DE 10Y Swap",         "Portfolio": "DM"},
    {"Ticker": "BPSWS10 BGN Curncy","Instrument":"UK 10Y Swap",      "Portfolio": "DM"},
    {"Ticker": "NDSWAP10 BGN Curncy","Instrument":"NZ 10Y Swap",     "Portfolio": "DM"},
    {"Ticker": "ADSW30 Curncy", "Instrument": "AU 30Y Swap",         "Portfolio": "DM"},
    {"Ticker": "CDSW30 Curncy", "Instrument": "CA 30Y Swap",         "Portfolio": "DM"},
    {"Ticker": "USSW30 Curncy", "Instrument": "US 30Y Swap",         "Portfolio": "DM"},
    {"Ticker": "EUSA30 Curncy", "Instrument": "DE 30Y Swap",         "Portfolio": "DM"},
    {"Ticker": "BPSWS30 BGN Curncy","Instrument":"UK 30Y Swap",      "Portfolio": "DM"},
    {"Ticker": "NDSWAP30 BGN Curncy","Instrument":"NZ 30Y Swap",     "Portfolio": "DM"},
    {"Ticker": "JYSO30 Curncy", "Instrument": "JP 30Y Swap",         "Portfolio": "DM"},
    {"Ticker": "MPSW10J BGN Curncy","Instrument":"MX 10Y Swap",      "Portfolio": "EM"},
    {"Ticker": "MPSWF10J BGN Curncy","Instrument":"MX 10Y Swap OIS", "Portfolio": "EM"},
    {"Ticker": "SASW10 Curncy", "Instrument": "SA 10Y Swap",         "Portfolio": "EM"},
    {"Ticker": "CKSW10 BGN Curncy","Instrument":"CZ 10Y Swap",       "Portfolio": "EM"},
    {"Ticker": "PZSW10 BGN Curncy","Instrument":"PO 10Y Swap",       "Portfolio": "EM"},
    {"Ticker": "KWSWNI10 BGN Curncy","Instrument":"SK 10Y Swap",     "Portfolio": "EM"},
    {"Ticker": "CCSWNI10 Curncy","Instrument": "CH 10Y Swap",        "Portfolio": "EM"},
    {"Ticker": "BPSWIT10 Curncy","Instrument": "UK 10Y Swap Inf",    "Portfolio": "DM"},
    {"Ticker": "SKSW10 Curncy", "Instrument": "SW 10Y Swap",         "Portfolio": "DM"},
    {"Ticker": "NKSW10 Curncy", "Instrument": "NK 10Y Swap",         "Portfolio": "DM"},
]

instruments_data = pd.DataFrame(instrument_records)
instr_to_country = {row.Instrument: guess_country(row.Instrument) for _, row in instruments_data.iterrows()}
dm_instruments = instruments_data[instruments_data.Portfolio == "DM"].Instrument.tolist()
em_instruments = instruments_data[instruments_data.Portfolio == "EM"].Instrument.tolist()

# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📈 BMIX Portfolio Risk Attribution – v2.3‑full")

    # ---- Load price history ----
    excel_file = "historical_data.xlsx"
    if not os.path.exists(excel_file):
        st.error("historical_data.xlsx not found in working directory")
        st.stop()
    raw_df = load_historical_data(excel_file)
    if raw_df.empty:
        st.error("No data loaded from Excel")
        st.stop()

    # ---- Sidebar controls ----
    st.sidebar.header("Configuration")
    sens_default = raw_df.columns.get_loc("US 10Y Future") if "US 10Y Future" in raw_df.columns else 0
    sensitivity_rate = st.sidebar.selectbox("Sensitivity instrument", raw_df.columns.tolist(), index=sens_default)
    vol_lookback = st.sidebar.slider("Volatility lookback (days)", 21, 1260, 252, step=21)
    var_lookback = st.sidebar.slider("VaR/CVaR lookback (days)", 252, 1260, 1260, step=252)

    tab_results, tab_input = st.tabs(["📊 Results", "📂 Input Positions"])

    # ---- Input Positions tab ----
    with tab_input:
        st.subheader("DM Positions")
        dm_df = pd.DataFrame({"Instrument": dm_instruments, "Outright": 0.0, "Curve": 0.0, "Spread": 0.0})
        grid_dm = AgGrid(dm_df, gridOptions=GridOptionsBuilder.from_dataframe(dm_df).build(),
                         height=450, width="100%")

        st.subheader("EM Positions")
        em_df = pd.DataFrame({"Instrument": em_instruments, "Outright": 0.0, "Curve": 0.0, "Spread": 0.0})
        grid_em = AgGrid(em_df, gridOptions=GridOptionsBuilder.from_dataframe(em_df).build(),
                         height=450, width="100%")

    # ---- Results tab ----
    with tab_results:
        if st.button("🚀 Run Risk Attribution"):
            with st.spinner("Calculating…"):
                # 1️⃣ Pre‑process price data
                changes = fallback_mx_ois_data(
                    calculate_daily_changes_in_bps(
                        adjust_time_zones(raw_df, instr_to_country)
                    )
                )
                cov = calculate_covariance_matrix(changes, vol_lookback)

                # 2️⃣ Build positions vector
                def tidy(df_src, label):
                    df = pd.DataFrame(df_src).astype({"Outright": float, "Curve": float, "Spread": float})
                    df["Portfolio"] = label
                    return df

                pos_df = pd.concat(
                    [tidy(grid_dm["data"], "DM"), tidy(grid_em["data"], "EM")],
                    ignore_index=True,
                )

                exploded = []
                for _, row in pos_df.iterrows():
                    for ptype in ["Outright", "Curve", "Spread"]:
                        val = row[ptype]
                        if val != 0:
                            exploded.append(
                                {"Instrument": row.Instrument, "Position Type": ptype, "Position": val}
                            )
                if not exploded:
                    st.warning("Enter at least one non‑zero position.")
                    st.stop()

                pos_vec = pd.DataFrame(exploded).set_index(["Instrument", "Position Type"])["Position"]

                # 3️⃣ Expand covariance to position level
                valid_instr = pos_vec.index.get_level_values(0).unique().intersection(cov.index)
                cov_sub = cov.loc[valid_instr, valid_instr]
                idx = pos_vec.index
                cov_exp = pd.DataFrame(
                    [[cov_sub.loc[i[0], j[0]] for j in idx] for i in idx], index=idx, columns=idx
                )

                # 4️⃣ Volatility contributions
                port_var = float(pos_vec.values @ cov_exp.values @ pos_vec.values)
                port_vol = np.sqrt(port_var)
                marg = cov_exp @ pos_vec
                contrib_vol = pos_vec * marg

                # 5️⃣ Historical VaR/CVaR
                hist = changes.tail(var_lookback)
                pos_instr = pos_vec.groupby(level=0).sum()
                pl_matrix = hist[pos_instr.index] * pos_instr
                port_pl = pl_matrix.sum(axis=1)

                def var_cvar(series: pd.Series, level: int):
                    var_val = -np.percentile(series, 100 - level)
                    tail = series[series <= -var_val]
                    cvar_val = -tail.mean() if not tail.empty else np.nan
                    return var_val, cvar_val

                VaR95, CVaR95 = var_cvar(port_pl, 95)
                VaR99, CVaR99 = var_cvar(port_pl, 99)

                compVaR95 = pl_matrix.apply(lambda s: var_cvar(s, 95)[0])
                compVaR99 = pl_matrix.apply(lambda s: var_cvar(s, 99)[0])
                compCVaR95 = pl_matrix.apply(lambda s: var_cvar(s, 95)[1])
                compCVaR99 = pl_matrix.apply(lambda s: var_cvar(s, 99)[1])

                # 6️⃣ Waterfall charts
                for lvl, p_var, p_cvar, comp_var, comp_cvar in [
                    (95, VaR95, CVaR95, compVaR95, compCVaR95),
                    (99, VaR99, CVaR99, compVaR99, compCVaR99),
                ]:
                    # --- VaR waterfall
                    var_df = comp_var.reset_index(name="Value").rename(columns={"index": "Instrument"})
                    divers_gap = p_var - var_df.Value.sum()
                    var_df = pd.concat(
                        [var_df, pd.DataFrame({"Instrument": ["Diversification"], "Value": [divers_gap]})],
                        ignore_index=True,
                    )
                    st.subheader(f"Component VaR {lvl}% (+ Diversification)")
                    st.plotly_chart(make_waterfall(var_df, "Instrument", "Value", f"VaR {lvl}%"),
                                    use_container_width=True)

                    # --- CVaR waterfall
                    cvar_df = comp_cvar.reset_index(name="Value").rename(columns={"index": "Instrument"})
                    divers_cgap = p_cvar - cvar_df.Value.sum()
                    cvar_df = pd.concat(
                        [cvar_df, pd.DataFrame({"Instrument": ["Diversification"], "Value": [divers_cgap]})],
                        ignore_index=True,
                    )
                    st.subheader(f"Component CVaR {lvl}% (+ Diversification)")
                    st.plotly_chart(make_waterfall(cvar_df, "Instrument", "Value", f"CVaR {lvl}%"),
                                    use_container_width=True)

                # 7️⃣ Metric tiles
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Volatility", f"{port_vol:.2f} bps")
                col2.metric("VaR 95%", f"{VaR95:.2f} bps")
                col3.metric("VaR 99%", f"{VaR99:.2f} bps")
                col4.metric("CVaR 95%", f"{CVaR95:.2f} bps")


if __name__ == "__main__":
    main()


