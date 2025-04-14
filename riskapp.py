"""
BMIX Portfolio Risk Attribution – v2.2
-------------------------------------------------
* Waterfall charts for Volatility, VaR 95/99, CVaR 95/99
* Historical‑simulation component VaR/CVaR
* Expanded AgGrid output (vol, VaR & CVaR columns)
* Instrument master list hard‑coded (no external CSV)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

# ──────────────────────────────────────────────────────────────────────────────
#  Streamlit page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 BMIX Portfolio Risk Attribution", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def load_historical_data(excel_file: str) -> pd.DataFrame:
    """Read every sheet in *excel_file*, concatenate, drop weekends."""
    try:
        excel = pd.ExcelFile(excel_file)
        frames = [excel.parse(s, index_col="date", parse_dates=True) for s in excel.sheet_names]
        df = pd.concat(frames, axis=1)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        return df[~df.index.dayofweek.isin([5, 6])]
    except Exception as exc:
        st.error(f"❌ Error loading Excel: {exc}")
        return pd.DataFrame()


def adjust_time_zones(df: pd.DataFrame, instr_to_country: dict) -> pd.DataFrame:
    """Shift non‑Asia instruments forward by +1 business day."""
    non_lag = ["JP", "AU", "SK", "CH"]
    countries = pd.Series([instr_to_country.get(c, "Other") for c in df.columns], index=df.columns)
    lag_cols = countries[~countries.isin(non_lag)].index
    out = df.copy()
    if len(lag_cols):
        out[lag_cols] = out[lag_cols].shift(1)
    return out.dropna()


def calculate_daily_changes_in_bps(df: pd.DataFrame) -> pd.DataFrame:
    return df.diff().dropna() * 100  # pct‑pt → bp


def fallback_mx_ois_data(changes: pd.DataFrame) -> pd.DataFrame:
    ois_map = {
        "MX 2Y Swap OIS": "MX 2Y Swap",
        "MX 5Y Swap OIS": "MX 5Y Swap",
        "MX 10Y Swap OIS": "MX 10Y Swap",
    }
    for ois, vanilla in ois_map.items():
        if ois in changes.columns and vanilla in changes.columns:
            changes[ois] = changes[ois].fillna(changes[vanilla])
    return changes


def calculate_volatilities(changes: pd.DataFrame, lookback: int) -> pd.Series:
    recent = changes.tail(lookback)
    return recent.std() * np.sqrt(252) if not recent.empty else pd.Series(dtype=float)


def calculate_covariance_matrix(changes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    recent = changes.tail(lookback)
    return recent.cov() * 252 if not recent.empty else pd.DataFrame()


def guess_country(name: str) -> str:
    codes = {"AU":"AU","US":"US","DE":"DE","UK":"UK","IT":"IT","CA":"CA","JP":"JP","CH":"CH","BR":"BR","MX":"MX","SA":"SA","CZ":"CZ","PO":"PO","SK":"SK","NZ":"NZ","SW":"SW","NK":"NK"}
    for c in codes:
        if c in name:
            return codes[c]
    return "Other"


def make_waterfall(df: pd.DataFrame, name_col: str, value_col: str, title: str) -> go.Figure:
    df = df.copy().sort_values(value_col, ascending=False)
    x = df[name_col].tolist() + ["Total"]
    y = df[value_col].tolist() + [df[value_col].sum()]
    measure = ["relative"] * len(df) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=x,
        y=y,
        measure=measure,
        decreasing={"marker": {"color": "#ff6961"}},
        increasing={"marker": {"color": "#77dd77"}},
        totals={"marker": {"color": "#6fa8dc"}},
    ))
    fig.update_layout(title=title, showlegend=False)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
#  Instrument master list (hard‑coded)
# ──────────────────────────────────────────────────────────────────────────────

instruments_data = pd.read_json(
    """
    [
      {"Ticker":"GACGB2 Index","Instrument":"AU 3Y Future","Portfolio":"DM"},
      {"Ticker":"GACGB10 Index","Instrument":"AU 10Y Future","Portfolio":"DM"},
      {"Ticker":"TUAFWD Comdty","Instrument":"US 2Y Future","Portfolio":"DM"},
      {"Ticker":"FVAFWD Comdty","Instrument":"US 5Y Future","Portfolio":"DM"},
      {"Ticker":"TYAFWD Comdty","Instrument":"US 10Y Future","Portfolio":"DM"},
      {"Ticker":"UXYAFWD Comdty","Instrument":"US 10Y Ultra Future","Portfolio":"DM"},
      {"Ticker":"WNAFWD Comdty","Instrument":"US 30Y Future","Portfolio":"DM"},
      {"Ticker":"DUAFWD Comdty","Instrument":"DE 2Y Future","Portfolio":"DM"},
      {"Ticker":"OEAFWD Comdty","Instrument":"DE 5Y Future","Portfolio":"DM"},
      {"Ticker":"RXAFWD Comdty","Instrument":"DE 10Y Future","Portfolio":"DM"},
      {"Ticker":"GAFWD Comdty","Instrument":"UK 10Y Future","Portfolio":"DM"},
      {"Ticker":"IKAFWD Comdty","Instrument":"IT 10Y Future","Portfolio":"DM"},
      {"Ticker":"CNAFWD Comdty","Instrument":"CA 10Y Future","Portfolio":"DM"},
      {"Ticker":"JBAFWD Comdty","Instrument":"JP 10Y Future","Portfolio":"DM"},
      {"Ticker":"CCSWNI1 Curncy","Instrument":"CH 1Y Swap","Portfolio":"EM"},
      {"Ticker":"ADSW2 Curncy","Instrument":"AU 2Y Swap","Portfolio":"DM"},
      {"Ticker":"CDSO2 Curncy","Instrument":"CA 2Y Swap","Portfolio":"DM"},
      {"Ticker":"USSW2 Curncy","Instrument":"US 2Y Swap","Portfolio":"DM"},
      {"Ticker":"EUSA2 Curncy","Instrument":"DE 2Y Swap","Portfolio":"DM"},
      {"Ticker":"BPSWS2 BGN Curncy","Instrument":"UK 2Y Swap","Portfolio":"DM"},
      {"Ticker":"NDSWAP2 BGN Curncy","Instrument":"NZ 2Y Swap","Portfolio":"DM"},
      {"Ticker":"I39302Y Index","Instrument":"BR 2Y Swap","Portfolio":"EM"},
      {"Ticker":"MPSW2B BGN Curncy","Instrument":"MX 2Y Swap","Portfolio":"EM"},
      {"Ticker":"MPSWF2B Curncy","Instrument":"MX 2Y Swap OIS","Portfolio":"EM"},
      {"Ticker":"SAFR1I2 BGN Curncy","Instrument":"SA 2Y Swap","Portfolio":"EM"},
      {"Ticker":"CKSW2 BGN Curncy","Instrument":"CZ 2Y Swap","Portfolio":"EM"},
      {"Ticker":"PZSW2 BGN Curncy","Instrument":"PO 2Y Swap","Portfolio":"EM"},
      {"Ticker":"KWSWNI2 BGN Curncy","Instrument":"SK 2Y Swap","Portfolio":"EM"},
      {"Ticker":"CCSWNI2 CMPN Curncy","Instrument":"CH 2Y Swap","Portfolio":"EM"},
      {"Ticker":"ADSW5 Curncy","Instrument":"AU 5Y Swap","Portfolio":"DM"},
      {"Ticker":"CDSO5 Curncy","Instrument":"CA 5Y Swap","Portfolio":"DM"},
      {"Ticker":"USSW5 Curncy","Instrument":"US 5Y Swap","Portfolio":"DM"},
      {"Ticker":"EUSA5 Curncy","Instrument":"DE 5Y Swap","Portfolio":"DM"},
      {"Ticker":"BPSWS5 BGN Curncy","Instrument":"UK 5Y Swap","Portfolio":"DM"},
      {"Ticker":"NDSWAP5 BGN Curncy","Instrument":"NZ 5Y Swap","Portfolio":"DM"},
      {"Ticker":"I39305Y Index","Instrument":"BR 5Y Swap","Portfolio":"EM"},
      {"Ticker":"MPSW5E Curncy","Instrument":"MX 5Y Swap","Portfolio":"EM"},
      {"Ticker":"MPSWF5E Curncy","Instrument":"MX 5Y Swap OIS","Portfolio":"EM"},
      {"Ticker":"SASW5 Curncy","Instrument":"SA 5Y Swap","Portfolio":"EM"},
      {"Ticker":"CKSW5 Curncy","Instrument":"CZ 5Y Swap","Portfolio":"EM"},
      {"Ticker":"PZSW5 Curncy","Instrument":"PO 5Y Swap","Portfolio":"EM"},
      {"Ticker":"KWSWNI5 Curncy","Instrument":"SK 5Y Swap","Portfolio":"EM"},
      {"Ticker":"CCSWNI5 Curncy","Instrument":"CH 5Y Swap","Portfolio":"EM"},
      {"Ticker":"JYSO5 Curncy","Instrument":"JP 5Y Swap","Portfolio":"DM"},
      {"Ticker":"ADSW10 Curncy","Instrument":"AU 10Y Swap","Portfolio":"DM"},
      {"Ticker":"CDSW10 Curncy","Instrument":"CA 10Y Swap","Portfolio":"DM"},
      {"Ticker":"USSW10 Curncy","Instrument":"US 10Y Swap","Portfolio":"DM"},
      {"Ticker":"EUSA10 Curncy","Instrument":"DE 10Y Swap","Portfolio":"DM"},
      {"Ticker":"BPSWS10 BGN Curncy","Instrument":"UK 10Y Swap","Portfolio":"DM"},
      {"Ticker":"NDSWAP10 BGN Curncy","Instrument":"NZ 10Y Swap","Portfolio":"DM"},
      {"Ticker":"ADSW30 Curncy","Instrument":"AU 30Y Swap","Portfolio":"DM"},
      {"Ticker":"CDSW30 Curncy","Instrument":"CA 30Y Swap","Portfolio":"DM"},
      {"Ticker":"USSW30 Curncy","Instrument":"US 30Y Swap","Portfolio":"DM"},
      {"Ticker":"EUSA30 Curncy","Instrument":"DE 30Y Swap","Portfolio":"DM"},
      {"Ticker":"BPSWS30 BGN Curncy","Instrument":"UK 30Y Swap","Portfolio":"DM"},
      {"Ticker":"NDSWAP30 BGN Curncy","Instrument":"NZ 30Y Swap","Portfolio":"DM"},
      {"Ticker":"JYSO30 Curncy","Instrument":"JP 30Y Swap","Portfolio":"DM"},
      {"Ticker":"MPSW10J BGN Curncy","Instrument":"MX 10Y Swap","Portfolio":"EM"},
      {"Ticker":"MPSWF10J BGN Curncy","Instrument":"MX 10Y Swap OIS","Portfolio":"EM"},
      {"Ticker":"SASW10 Curncy","Instrument":"SA 10Y Swap","Portfolio":"EM"},
      {"Ticker":"CKSW10 BGN Curncy","Instrument":"CZ 10Y Swap","Portfolio":"EM"},
      {"Ticker":"PZSW10 BGN Curncy","Instrument":"PO 10Y Swap","Portfolio":"EM"},
      {"Ticker":"KWSWNI10 BGN Curncy","Instrument":"SK 10Y Swap","Portfolio":"EM"},
      {"Ticker":"CCSWNI10 Curncy","Instrument":"CH 10Y Swap","Portfolio":"EM"},
      {"Ticker":"BPSWIT10 Curncy","Instrument":"UK 10Y Swap Inf","Portfolio":"DM"},
      {"Ticker":"SKSW10 Curncy","Instrument":"SW 10Y Swap","Portfolio":"DM"},
      {"Ticker":"NKSW10 Curncy","Instrument":"NK 10Y Swap","Portfolio":"DM"}
    ]
    """
)

# Map Instrument → Portfolio & build country map on the fly
instrument_to_portfolio = instruments_data.set_index("Instrument")["Portfolio"].to_dict()

# We'll derive instrument → country lazily using guess_country()
instr_to_country = {instr: guess_country(instr) for instr in instruments_data["Instrument"].tolist()}

# Lists for default positions sheets
dm_instruments = instruments_data[instruments_data["Portfolio"] == "DM"]["Instrument"].tolist()
em_instruments = instruments_data[instruments_data["Portfolio"] == "EM"]["Instrument"].tolist()

# ──────────────────────────────────────────────────────────────────────────────
#  Main app logic
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 BMIX Portfolio Risk Attribution – v2.2")

    # ---------------- Sidebar settings ----------------
    excel_file = "historical_data.xlsx"
    if not os.path.exists(excel_file):
        st.sidebar.error(f"❌ '{excel_file}' not found in working directory.")
        st.stop()

    raw_df = load_historical_data(excel_file)
    if raw_df.empty:
        st.error("No data loaded from Excel file.")
        st.stop()

    st.sidebar.header("Sensitivity Instrument")
    sens_default = raw_df.columns.tolist().index("US 10Y Future") if "US 10Y Future" in raw_df.columns else 0
    sensitivity_rate = st.sidebar.selectbox("Choose instrument:", raw_df.columns.tolist(), index=sens_default)

    st.sidebar.header("Lookback Windows")
    vol_lookback = st.sidebar.slider("Volatility lookback (days)", 21, 1260, 252, step=21)
    var_lookback = st.sidebar.slider("VaR/CVaR lookback (days)", 252, 1260, 1260, step=252)

    # ---------------- Tabs ----------------
    tab_results, tab_input = st.tabs(["📊 Results", "📂 Input Positions"])

    # ---------------- Input Positions Tab ----------------
    with tab_input:
        st.subheader("DM Portfolio Positions")
        dm_df = pd.DataFrame({"Instrument": dm_instruments, "Outright": 0.0, "Curve": 0.0, "Spread": 0.0})
        gb_dm = GridOptionsBuilder.from_dataframe(dm_df)
        gb_dm.configure_default_column(editable=True, resizable=True)
        dm_data = AgGrid(dm_df, gridOptions=gb_dm.build(), height=600, width="100%")
        positions_dm = dm_data["data"]

        st.subheader("EM Portfolio Positions")
        em_df = pd.DataFrame({"Instrument": em_instruments, "Outright": 0.0, "Curve": 0.0, "Spread": 0.0})
        gb_em = GridOptionsBuilder.from_dataframe(em_df)
        gb_em.configure_default_column(editable=True, resizable=True)
        em_data = AgGrid(em_df, gridOptions=gb_em.build(), height=600, width="100%")
        positions_em = em_data["data"]

    # ---------------- Results Tab ----------------
    with tab_results:
        if st.button("🚀 Run Risk Attribution"):
            with st.spinner("Calculating risk metrics…"):
                # 1️⃣ Price data prep
                df_adj = adjust_time_zones(raw_df, instr_to_country)
                changes = fallback_mx_ois_data(calculate_daily_changes_in_bps(df_adj))
                if changes.empty:
                    st.error("Daily changes dataframe is empty after preprocessing.")
                    st.stop()

                vols = calculate_volatilities(changes, vol_lookback)
                cov = calculate_covariance_matrix(changes, vol_lookback)

                # 2️⃣ Positions vector
                def tidy_positions(pos_df, portfolio_label):
                    pos_df = pd.DataFrame(pos_df).astype({"Outright": float, "Curve": float, "Spread": float})
                    pos_df["Portfolio"] = portfolio_label
                    return pos_df

                positions_df = pd.concat([
                    tidy_positions(positions_dm, "DM"),
                    tidy_positions(positions_em, "EM"),
                ], ignore_index=True)

                exploded = []
                for _, row in positions_df.iterrows():
                    for ptype in ["Outright", "Curve", "Spread"]:
                        val = row[ptype]
                        if val != 0:
                            exploded.append({"Instrument": row["Instrument"], "Position Type": ptype, "Position": val, "Portfolio": row["Portfolio"]})

                if not exploded:
                    st.warning("No non‑zero positions entered.")
                    st.stop()

                pos_df = pd.DataFrame(exploded)
                pos_vector = pos_df.set_index(["Instrument", "Position Type"])["Position"]

                # Align covariance
                valid_instr = pos_vector.index.get_level_values("Instrument").unique().intersection(cov.index)
                if valid_instr.empty:
                    st.error("None of the instruments have history in the covariance window.")
                    st.stop()
                cov_sub = cov.loc[valid_instr, valid_instr]
                idx = pos_vector.index
                cov_expanded = pd.DataFrame(0.0, index=idx, columns=idx)
                for i in idx:
                    for j in idx:
                        cov_expanded.loc[i, j] = cov_sub.loc[i[0], j[0]]

                # 3️⃣ Volatility contributions
                port_var = float(pos_vector.values @ cov_expanded.values @ pos_vector.values)
                if port_var <= 0:
                    st.error("Computed portfolio variance non‑positive.")
                    st.stop()
                port_vol = np.sqrt(port_var)
                marginal = cov_expanded @ pos_vector
                contrib_var = pos_vector * marginal
                contrib_vol = contrib_var / port_vol
                pct_contrib = contrib_var / port_var * 100

                # 4️⃣ Historical VaR/CVaR
                hist_window = changes.tail(var_lookback)
                pos_by_instr = pos_vector.groupby("Instrument").sum()
                valid_for_var = pos_by_instr.index.intersection(hist_window.columns)
                hist_window = hist_window[valid_for_var]

                # P/L matrix (bps) – each instrument already scaled by position size
                pl_matrix = hist_window.mul(pos_by_instr.loc[valid_for_var], axis=1)
                port_pl   = pl_matrix.sum(axis=1)

                def calc_var_cvar(series: pd.Series, level: int):
                    var_val = -np.percentile(series, 100 - level)
                    tail    = series[series <= -var_val]
                    cvar_val = -tail.mean() if not tail.empty else np.nan
                    return var_val, cvar_val

                VaR95, CVaR95 = calc_var_cvar(port_pl, 95)
                VaR99, CVaR99 = calc_var_cvar(port_pl, 99)

                def component_series(level: int):
                    comp_var, comp_cvar = {}, {}
                    for instr in pl_matrix.columns:
                        v, c = calc_var_cvar(pl_matrix[instr], level)
                        comp_var[instr]  = v
                        comp_cvar[instr] = c
                    return pd.Series(comp_var), pd.Series(comp_cvar)

                compVaR95, compCVaR95 = component_series(95)
                compVaR99, compCVaR99 = component_series(99)

                # 5️⃣ Build output table
                out_tbl = pos_df.copy()
                out_tbl["Contribution to Volatility (bps)"] = contrib_vol.values
                out_tbl["Percent Contribution (%)"] = pct_contrib.values
                out_tbl["Contribution to VaR 95 (bps)"] = out_tbl["Instrument"].map(compVaR95)
                out_tbl["Contribution to CVaR 95 (bps)"] = out_tbl["Instrument"].map(compCVaR95)
                out_tbl["Contribution to VaR 99 (bps)"] = out_tbl["Instrument"].map(compVaR99)
                out_tbl["Contribution to CVaR 99 (bps)"] = out_tbl["Instrument"].map(compCVaR99)

                # 6️⃣ Charts – Volatility waterfalls
                st.subheader("Volatility Contribution – by Instrument")
                inst_vol_df = out_tbl.groupby("Instrument")["Contribution to Volatility (bps)"].sum().reset_index()
                st.plotly_chart(make_waterfall(inst_vol_df, "Instrument", "Contribution to Volatility (bps)", "Instrument Volatility"), use_container_width=True)

                st.subheader("Volatility Contribution – Country/Bucket")
                cb_df = out_tbl.copy()
                cb_df["Label"] = cb_df["Instrument"].apply(guess_country) + " - " + cb_df["Position Type"]
                cb_grp = cb_df.groupby("Label")["Contribution to Volatility (bps)"].sum().reset_index()
                st.plotly_chart(make_waterfall(cb_grp, "Label", "Contribution to Volatility (bps)", "Country & Bucket Volatility"), use_container_width=True)

                # VaR/CVaR waterfalls
                for lvl, port_var_val, port_cvar_val, comp_var, comp_cvar in [
                    (95, VaR95, CVaR95, compVaR95, compCVaR95),
                    (99, VaR99, CVaR99, compVaR99, compCVaR99),
                ]:
                    if not comp_var.empty:
                        def _series_to_df(s):
                            df_tmp = s.reset_index()
                            df_tmp.columns = ["Instrument", "Value"]
                            return df_tmp

                        # ----- VaR waterfall with Diversification bar -----
                        var_df = _series_to_df(comp_var)
                        divers_gap = port_var_val - var_df["Value"].sum()
                        var_df = pd.concat([
                            var_df,
                            pd.DataFrame({"Instrument": ["Diversification"], "Value": [divers_gap]})
                        ], ignore_index=True)
                        st.subheader(f"Component VaR {lvl}% – Instrument (+ Diversification)")
                        st.plotly_chart(
                            make_waterfall(var_df, "Instrument", "Value", f"Component VaR {lvl}%"),
                            use_container_width=True,
                        )(_series_to_df(comp_var), "Instrument", "Value", f"Component VaR {lvl}%"),
                            use_container_width=True,
                        )
                        # ----- CVaR waterfall with Diversification bar -----
                        cvar_df = _series_to_df(comp_cvar)
                        divers_cgap = port_cvar_val - cvar_df["Value"].sum()
                        cvar_df = pd.concat([
                            cvar_df,
                            pd.DataFrame({"Instrument": ["Diversification"], "Value": [divers_cgap]})
                        ], ignore_index=True)
                        st.subheader(f"Component CVaR {lvl}% – Instrument (+ Diversification)")
                        st.plotly_chart(
                            make_waterfall(cvar_df, "Instrument", "Value", f"Component CVaR {lvl}%"),
                            use_container_width=True,
                        )(_series_to_df(comp_cvar), "Instrument", "Value", f"Component CVaR {lvl}%"),
                            use_container_width=True,
                        )

                # 7️⃣ Metric tiles
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Portfolio Volatility", f"{port_vol:.2f} bps")
                col2.metric("VaR 95%", f"{VaR95:.2f} bps")
                col3.metric("VaR 99%", f"{VaR99:.2f} bps")
                col4.metric("CVaR 95%", f"{CVaR95:.2f} bps"):.2f} bps" if not compCVaR95.empty else "N/A")

                # 8️⃣ Detailed table
                st.subheader("Detailed Contributions Table")
                numeric_cols = [c for c in out_tbl.columns if c not in ["Instrument", "Position Type", "Portfolio"]]
                out_tbl[numeric_cols] = out_tbl[numeric_cols].round(2)
                gb = GridOptionsBuilder.from_dataframe(out_tbl)
                gb.configure_default_column(editable=False, resizable=True)
                AgGrid(out_tbl, gridOptions=gb.build(), height=500, width="100%")

                st.download_button("📥 Download CSV", data=out_tbl.to_csv(index=False).encode(), file_name="risk_contributions.csv")


if __name__ == "__main__":
    main()


