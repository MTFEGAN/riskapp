"""
BMIX Portfolio Risk Attribution – v2.0
-------------------------------------------------
Full Streamlit application with:
* Waterfall charts (instrument & country/bucket) for volatility contributions
* Historical‑simulation component VaR/CVaR (95 % & 99 %) and their waterfalls
* Expanded AgGrid table with new VaR/CVaR contribution columns
* Same position‑input workflow as v1
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

# ──────────────────────────────────────────────────────────────────────────────
#  Streamlit page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 BMIX Portfolio Risk Attribution", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def load_historical_data(excel_file: str) -> pd.DataFrame:
    """Read all sheets from *excel_file* and concatenate by columns."""
    try:
        excel = pd.ExcelFile(excel_file)
        frames = [excel.parse(s, index_col="date", parse_dates=True) for s in excel.sheet_names]
        df = pd.concat(frames, axis=1)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        # Drop weekends
        df = df[~df.index.dayofweek.isin([5, 6])]
        return df
    except Exception as exc:
        st.error(f"❌ Error loading Excel: {exc}")
        return pd.DataFrame()


def adjust_time_zones(df: pd.DataFrame, instrument_country: dict) -> pd.DataFrame:
    """Shift non‑Asia instruments by +1 business day so they line up."""
    non_lag = ["JP", "AU", "SK", "CH"]
    countries = pd.Series([instrument_country.get(c, "Other") for c in df.columns], index=df.columns)
    lag_cols = countries[~countries.isin(non_lag)].index
    out = df.copy()
    if len(lag_cols):
        out[lag_cols] = out[lag_cols].shift(1)
    return out.dropna()


def calculate_daily_changes_in_bps(df: pd.DataFrame) -> pd.DataFrame:
    """First‑difference in percentage‑points × 100 → basis‑points."""
    return df.diff().dropna() * 100


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
    if changes.empty:
        return pd.Series(dtype=float)
    recent = changes.tail(lookback)
    return recent.std() * np.sqrt(252)


def calculate_covariance_matrix(changes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()
    recent = changes.tail(lookback)
    return recent.cov() * 252


def compute_beta(x: pd.Series, y: pd.Series, lookback: int) -> float:
    common = x.index.intersection(y.index)
    if common.empty:
        return np.nan
    x, y = x.loc[common].tail(lookback), y.loc[common].tail(lookback)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    cov = np.cov(x, y)[0, 1]
    var_y = np.var(y)
    return np.nan if var_y == 0 else cov / var_y


def guess_country_from_instrument_name(name: str) -> str:
    codes = {
        "AU": "AU", "US": "US", "DE": "DE", "UK": "UK", "IT": "IT", "CA": "CA", "JP": "JP", "CH": "CH",
        "BR": "BR", "MX": "MX", "SA": "SA", "CZ": "CZ", "PO": "PO", "SK": "SK", "NZ": "NZ", "SW": "SW", "NK": "NK",
    }
    for code in codes:
        if code in name:
            return codes[code]
    return "Other"


def make_waterfall(df: pd.DataFrame, name_col: str, value_col: str, title: str) -> go.Figure:
    df = df.copy().sort_values(value_col, ascending=False)
    measure = ["relative"] * len(df) + ["total"]
    x = df[name_col].tolist() + ["Total"]
    y = df[value_col].tolist() + [df[value_col].sum()]
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
#  Instrument → country mapping  (same as v1)
# ──────────────────────────────────────────────────────────────────────────────
#  *Truncated for brevity in comments – keep full dict here*

instrument_country = {
    "AU 3Y Future": "AU", "AU 10Y Future": "AU", "US 2Y Future": "US", "US 5Y Future": "US", "US 10Y Future": "US",
    "US 10Y Ultra Future": "US", "US 30Y Future": "US", "DE 2Y Future": "DE", "DE 5Y Future": "DE", "DE 10Y Future": "DE",
    "UK 10Y Future": "UK", "IT 10Y Future": "IT", "CA 10Y Future": "CA", "JP 10Y Future": "JP", "CH 1Y Swap": "CH",
    "AU 2Y Swap": "AU", "CA 2Y Swap": "CA", "US 2Y Swap": "US", "DE 2Y Swap": "DE", "UK 2Y Swap": "UK", "NZ 2Y Swap": "NZ",
    "BR 2Y Swap": "BR", "MX 2Y Swap": "MX", "MX 2Y Swap OIS": "MX", "SA 2Y Swap": "SA", "CZ 2Y Swap": "CZ", "PO 2Y Swap": "PO",
    "SK 2Y Swap": "SK", "CH 2Y Swap": "CH", "AU 5Y Swap": "AU", "CA 5Y Swap": "CA", "US 5Y Swap": "US", "DE 5Y Swap": "DE",
    "UK 5Y Swap": "UK", "NZ 5Y Swap": "NZ", "BR 5Y Swap": "BR", "MX 5Y Swap": "MX", "MX 5Y Swap OIS": "MX", "SA 5Y Swap": "SA",
    "CZ 5Y Swap": "CZ", "PO 5Y Swap": "PO", "SK 5Y Swap": "SK", "CH 5Y Swap": "CH", "JP 5Y Swap": "JP", "AU 10Y Swap": "AU",
    "CA 10Y Swap": "CA", "US 10Y Swap": "US", "DE 10Y Swap": "DE", "UK 10Y Swap": "UK", "NZ 10Y Swap": "NZ", "AU 30Y Swap": "AU",
    "CA 30Y Swap": "CA", "US 30Y Swap": "US", "DE 30Y Swap": "DE", "UK 30Y Swap": "UK", "NZ 30Y Swap": "NZ", "JP 30Y Swap": "JP",
    "MX 10Y Swap": "MX", "MX 10Y Swap OIS": "MX", "SA 10Y Swap": "SA", "CZ 10Y Swap": "CZ", "PO 10Y Swap": "PO", "SK 10Y Swap": "SK",
    "CH 10Y Swap": "CH", "UK 10Y Swap Inf": "UK", "SW 10Y Swap": "SW", "NK 10Y Swap": "NK",
}

# ──────────────────────────────────────────────────────────────────────────────
#  Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 BMIX Portfolio Risk Attribution – v2")

    # ------------------------------------------------------------------
    #  Sidebar – load historical data & select settings
    # ------------------------------------------------------------------
    excel_file = "historical_data.xlsx"
    if not os.path.exists(excel_file):
        st.sidebar.error(f"❌ '{excel_file}' not found.")
        st.stop()

    raw_df = load_historical_data(excel_file)
    if raw_df.empty:
        st.error("No data loaded from Excel.")
        st.stop()

    st.sidebar.header("🔍 Sensitivity Instrument")
    sens_default = raw_df.columns.tolist().index("US 10Y Future") if "US 10Y Future" in raw_df.columns else 0
    sensitivity_rate = st.sidebar.selectbox("Choose instrument:", raw_df.columns.tolist(), index=sens_default)

    st.sidebar.header("⚙️ Lookback Windows")
    vol_periods = {"📅 1M (~21d)": 21, "📆 3M (~63d)": 63, "📅 6M (~126d)": 126, "🗓️ 1Y (252d)": 252,
                   "📅 3Y (~756d)": 756, "📆 5Y (~1260d)": 1260}
    var_periods = {"🗓️ 1Y (252d)": 252, "📆 5Y (~1260d)": 1260}
    vol_lookback = vol_periods[st.sidebar.selectbox("Volatility:", list(vol_periods.keys()), index=3)]
    var_lookback = var_periods[st.sidebar.selectbox("VaR/CVaR:", list(var_periods.keys()), index=1)]

    # ------------------------------------------------------------------
    #  Instruments master list & blank position sheets
    # ------------------------------------------------------------------
    instruments_data = pd.read_csv("instruments_master.csv") if os.path.exists("instruments_master.csv") else None
    if instruments_data is None:
        st.error("Please provide 'instruments_master.csv' with Ticker, Instrument Name, Portfolio columns.")
        st.stop()

    dm_instruments = instruments_data[instruments_data["Portfolio"] == "DM"]["Instrument Name"].tolist()
    em_instruments = instruments_data[instruments_data["Portfolio"] == "EM"]["Instrument Name"].tolist()

    default_positions_dm = pd.DataFrame({"Instrument": dm_instruments, "Outright": 0.0, "Curve": 0.0, "Spread": 0.0})
    default_positions_em = pd.DataFrame({"Instrument": em_instruments, "Outright": 0.0, "Curve": 0.0, "Spread": 0.0})

    # ------------------------------------------------------------------
    #  Tabs
    # ------------------------------------------------------------------
    tab_results, tab_input, tab_settings = st.tabs(["📊 Results", "📂 Input Positions", "🔧 Extra Settings"])

    # ---- Input tab -----------------------------------------------------
    with tab_input:
        st.subheader("📈 DM Positions")
        gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
        gb_dm.configure_default_column(editable=True, resizable=True)
        dm_data = AgGrid(default_positions_dm, gridOptions=gb_dm.build(), height=600, width="100%")
        positions_data_dm = dm_data["data"]

        st.subheader("🌍 EM Positions")
        gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
        gb_em.configure_default_column(editable=True, resizable=True)
        em_data = AgGrid(default_positions_em, gridOptions=gb_em.build(), height=600, width="100%")
        positions_data_em = em_data["data"]

    # ---- Settings tab (just explanatory) -------------------------------
    with tab_settings:
        st.markdown("Use the sidebar to adjust lookbacks and sensitivity instrument.")

    # ---- Results tab ---------------------------------------------------
    with tab_results:
        if st.button("🚀 Run Risk Attribution"):
            with st.spinner("Crunching numbers..."):
                # -------------------------------- Pre‑processing --------------------------------
                df = adjust_time_zones(raw_df, instrument_country)
                changes = fallback_mx_ois_data(calculate_daily_changes_in_bps(df))
                if changes.empty:
                    st.warning("No daily changes after adjustments.")
                    st.stop()

                vols = calculate_volatilities(changes, vol_lookback)
                cov = calculate_covariance_matrix(changes, vol_lookback)

                # ----------------------------- Build positions vector ---------------------------
                def prep_positions(df_pos, portfolio_label):
                    df_pos = pd.DataFrame(df_pos).astype({"Outright": float, "Curve": float, "Spread": float})
                    df_pos["Portfolio"] = portfolio_label
                    return df_pos

                positions_df = pd.concat([
                    prep_positions(positions_data_dm, "DM"),
                    prep_positions(positions_data_em, "EM"),
                ], ignore_index=True)

                # Explode into separate rows per position type where value ≠ 0
                exploded = []
                for _, row in positions_df.iterrows():
                    for ptype in ["Outright", "Curve", "Spread"]:
                        val = row[ptype]
                        if val != 0:
                            exploded.append({
                                "Instrument": row["Instrument"],
                                "Position Type": ptype,
                                "Position": val,
                                "Portfolio": row["Portfolio"],
                            })
                if not exploded:
                    st.warning("No active positions entered.")
                    st.stop()
                exploded_df = pd.DataFrame(exploded)
                pos_vector = exploded_df.set_index(["Instrument", "Position Type"])["Position"]

                # --------------------------- Align covariance matrix ----------------------------
                valid_instr = pos_vector.index.get_level_values("Instrument").unique().intersection(cov.index)
                if valid_instr.empty:
                    st.error("None of the instruments have price history in the covariance window.")
                    st.stop()
                cov_sub = cov.loc[valid_instr, valid_instr]
                # Build expanded covariance (position‑type level) – each sub‑instrument block same cov
                idx = pos_vector.index
                expanded_cov = pd.DataFrame(0.0, index=idx, columns=idx)
                for i in idx:
                    for j in idx:
                        expanded_cov.loc[i, j] = cov_sub.loc[i[0], j[0]]

                # --------------------------- Volatility contributions ---------------------------
                port_var = float(pos_vector.values @ expanded_cov.values @ pos_vector.values)
                if port_var <= 0:
                    st.error("Portfolio variance non‑positive.")
                    st.stop()
                port_vol = np.sqrt(port_var)

                marginal = expanded_cov @ pos_vector
                contrib_var = pos_vector * marginal
                contrib_vol = contrib_var / port_vol
                pct_contrib = contrib_var / port_var * 100

                # --------------------------- Historical VaR / CVaR -----------------------------
                hist_window = changes.tail(var_lookback)
                pos_by_instr = pos_vector.groupby("Instrument").sum()
                valid_for_var = pos_by_instr.index.intersection(hist_window.columns)
                hist_window = hist_window[valid_for_var]
                port_pl = hist_window.mul(pos_by_instr.loc[valid_for_var], axis=1).sum(axis=1)

                def comp_var_cvar(level: int):
                    """Return tuple: portfolio VaR, component VaR Series, component CVaR Series"""
                    if port_pl.empty:
                        return np.nan, pd.Series(dtype=float), pd.Series(dtype=float)
                    var_val = -np.percentile(port_pl, 100 - level)
                    tail_mask = port_pl <= -var_val
                    tail_pl = hist_window.loc[tail_mask]
                    comp_var = -tail_pl.quantile(level / 100.0)
                    comp_cvar = -tail_pl.mean()
                    return var_val, comp_var, comp_cvar

                VaR95, compVaR95, compCVaR95 = comp_var_cvar(95)
                VaR99, compVaR99, compCVaR99 = comp_var_cvar(99)

                # --------------------------- Build output table --------------------------------
                out_tbl = exploded_df.copy()
                out_tbl["Contribution to Volatility (bps)"] = contrib_vol.values
                out_tbl["Percent Contribution (%)"] = pct_contrib.values
                out_tbl["Contribution to VaR 95 (bps)"] = out_tbl["Instrument"].map(compVaR95)
                out_tbl["Contribution to CVaR 95 (bps)"] = out_tbl["Instrument"].map(compCVaR95)
                out_tbl["Contribution to VaR 99 (bps)"] = out_tbl["Instrument"].map(compVaR99)
                out_tbl["Contribution to CVaR 99 (bps)"] = out_tbl["Instrument"].map(compCVaR99)

                # --------------------------- Waterfall charts -----------------------------------
                st.subheader("Volatility Contribution – Instrument")
                inst_vol = out_tbl.groupby("Instrument")["Contribution to Volatility (bps)"].sum().reset_index()
                st.plotly_chart(make_waterfall(inst_vol, "Instrument", "Contribution to Volatility (bps)", "Instrument Volatility"), use_container_width=True)

                st.subheader("Volatility Contribution – Country/Bucket")
                cb = out_tbl.copy()
                cb["Country"] = cb["Instrument"].apply(guess_country_from_instrument_name)
                cb["Label"] = cb["Country"] + " - " + cb["Position Type"]
                cb_grp = cb.groupby("Label")["Contribution to Volatility (bps)"].sum().reset_index()
                st.plotly_chart(make_waterfall(cb_grp, "Label", "Contribution to Volatility (bps)", "Country & Bucket Volatility"), use_container_width=True)

                for lvl, comp_var, comp_cvar in [(95, compVaR95, compCVaR95), (99, compVaR99, compCVaR99)]:
                    if not comp_var.empty:
                        st.subheader(f"Component VaR {lvl}% – Instrument")
                        df_var = comp_var.reset_index().rename(columns={0: "Value"})
                        st.plotly_chart(make_waterfall(df_var, "Instrument", "Value", f"Component VaR {lvl}%"), use_container_width=True)
                        st.subheader(f"Component CVaR {lvl}% – Instrument")
                        df_cvar = comp_cvar.reset_index().rename(columns={0: "Value"})
                        st.plotly_chart(make_waterfall(df_cvar, "Instrument", "Value", f"Component CVaR {lvl}%"), use_container_width=True)

                # --------------------------- Metric tiles ---------------------------------------
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Portfolio Volatility", f"{port_vol:.2f} bps")
                col2.metric("VaR 95%", f"{VaR95:.2f} bps")
                col3.metric("VaR 99%", f"{VaR99:.2f} bps")
                col4.metric("CVaR 95%", f"{compCVaR95.mean():.2f} bps" if not compCVaR95.empty else "N/A")

                # --------------------------- Detailed table -------------------------------------
                st.subheader("Detailed Contributions")
                num_cols = [c for c in out_tbl.columns if c not in ["Instrument", "Position Type", "Portfolio"]]
                out_tbl[num_cols] = out_tbl[num_cols].round(2)
                gb = GridOptionsBuilder.from_dataframe(out_tbl)
                gb.configure_default_column(editable=False, resizable=True)
                AgGrid(out_tbl, gridOptions=gb.build(), height=500, width="100%")

                st.download_button("📥 Download CSV", data=out_tbl.to_csv(index=False).encode(), file_name="risk_contributions.csv")


if __name__ == "__main__":
    main()