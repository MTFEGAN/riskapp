import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="📈 BMIX Portfolio Risk Attribution", layout="wide")

# ── Helper Functions ───────────────────────────────────────────────────────────

def fmt_val(x):
    return f"{x:.2f} bps" if (not np.isnan(x) and not np.isinf(x)) else "N/A"

def load_historical_data(excel_file):
    try:
        excel = pd.ExcelFile(excel_file)
        df_list = []
        for sheet in excel.sheet_names:
            df_sheet = excel.parse(sheet_name=sheet, index_col='date', parse_dates=True)
            df_list.append(df_sheet)
        df = pd.concat(df_list, axis=1)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.dayofweek.isin([5,6])]
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return pd.DataFrame()

def adjust_time_zones(df, instrument_country):
    non_lag = ['JP','AU','SK','CH']
    inst_ctry = pd.Series([instrument_country.get(i,'Other') for i in df.columns], index=df.columns)
    to_lag = inst_ctry[~inst_ctry.isin(non_lag)].index.tolist()
    df2 = df.copy()
    if to_lag:
        df2[to_lag] = df2[to_lag].shift(1)
    return df2.dropna()

def calculate_daily_changes_in_bps(df):
    return df.diff().dropna() * 100

def fallback_mx_ois_data(dc):
    ois_map = {
        'MX 2Y Swap OIS':'MX 2Y Swap',
        'MX 5Y Swap OIS':'MX 5Y Swap',
        'MX 10Y Swap OIS':'MX 10Y Swap'
    }
    for ois, non in ois_map.items():
        if ois in dc.columns and non in dc.columns:
            dc[ois] = dc[ois].fillna(dc[non])
    return dc

def calculate_volatilities(dc, lookback):
    if dc.empty: return pd.Series(dtype=float)
    recent = dc.tail(lookback)
    if recent.empty: return pd.Series(dtype=float)
    return recent.std() * np.sqrt(252)

def calculate_covariance_matrix(dc, lookback):
    if dc.empty: return pd.DataFrame()
    recent = dc.tail(lookback)
    if recent.empty: return pd.DataFrame()
    return recent.cov() * 252

def guess_country(instr):
    codes = ['AU','US','DE','UK','IT','CA','JP','CH','BR','MX','SA','CZ','PO','SK','NZ','SW','NK']
    for c in codes:
        if c in instr:
            return c
    return 'Other'

def create_waterfall_chart(labels, values, total, title, include_diversification=False):
    if include_diversification:
        div = total - sum(values)
        measures = ['relative']*len(values) + ['relative','total']
        x = labels + ['Diversification Impact','Total']
        y = values + [div,total]
        text = [f"{v:.2f}" for v in values] + [f"{div:.2f}",f"{total:.2f}"]
    else:
        measures = ['relative']*len(values) + ['total']
        x = labels + ['Total']
        y = values + [total]
        text = [f"{v:.2f}" for v in values] + [f"{total:.2f}"]
    fig = go.Figure(go.Waterfall(
        orientation='v', measure=measures, x=x, y=y, text=text,
        connector={'line':{'color':'gray'}}
    ))
    fig.update_layout(title=title, waterfallgroupgap=0.3)
    return fig

# ── Instrument-Country Mapping ────────────────────────────────────────────────
instrument_country = {
    "AU 3Y Future":"AU","AU 10Y Future":"AU","US 2Y Future":"US","US 5Y Future":"US",
    "US 10Y Future":"US","US 10Y Ultra Future":"US","US 30Y Future":"US","DE 2Y Future":"DE",
    "DE 5Y Future":"DE","DE 10Y Future":"DE","UK 10Y Future":"UK","IT 10Y Future":"IT",
    "CA 10Y Future":"CA","JP 10Y Future":"JP","CH 1Y Swap":"CH","AU 2Y Swap":"AU",
    "CA 2Y Swap":"CA","US 2Y Swap":"US","DE 2Y Swap":"DE","UK 2Y Swap":"UK",
    "NZ 2Y Swap":"NZ","BR 2Y Swap":"BR","MX 2Y Swap":"MX","MX 2Y Swap OIS":"MX",
    "SA 2Y Swap":"SA","CZ 2Y Swap":"CZ","PO 2Y Swap":"PO","SK 2Y Swap":"SK",
    "CH 2Y Swap":"CH","AU 5Y Swap":"AU","CA 5Y Swap":"CA","US 5Y Swap":"US",
    "DE 5Y Swap":"DE","UK 5Y Swap":"UK","NZ 5Y Swap":"NZ","BR 5Y Swap":"BR",
    "MX 5Y Swap":"MX","MX 5Y Swap OIS":"MX","SA 5Y Swap":"SA","CZ 5Y Swap":"CZ",
    "PO 5Y Swap":"PO","SK 5Y Swap":"SK","CH 5Y Swap":"CH","JP 5Y Swap":"JP",
    "AU 10Y Swap":"AU","CA 10Y Swap":"CA","US 10Y Swap":"US","DE 10Y Swap":"DE",
    "UK 10Y Swap":"UK","NZ 10Y Swap":"NZ","AU 30Y Swap":"AU","CA 30Y Swap":"CA",
    "US 30Y Swap":"US","DE 30Y Swap":"DE","UK 30Y Swap":"UK","NZ 30Y Swap":"NZ",
    "JP 30Y Swap":"JP","MX 10Y Swap":"MX","MX 10Y Swap OIS":"MX","SA 10Y Swap":"SA",
    "CZ 10Y Swap":"CZ","PO 10Y Swap":"PO","SK 10Y Swap":"SK","CH 10Y Swap":"CH",
    "UK 10Y Swap Inf":"UK","SW 10Y Swap":"SW","NK 10Y Swap":"NK"
}

# ── Main Application ─────────────────────────────────────────────────────────

def main():
    st.title("📈 BMIX Portfolio Risk Attribution")
    st.write("App initialized successfully.")

    # Load historical data
    excel_file = 'historical_data.xlsx'
    if not os.path.exists(excel_file):
        st.sidebar.error(f"❌ '{excel_file}' not found.")
        st.stop()
    raw_df = load_historical_data(excel_file)
    if raw_df.empty:
        st.error("No data loaded from Excel.")
        st.stop()
    available_instruments = raw_df.columns.tolist()

    # Sidebar config
    st.sidebar.header("🔍 Configuration")
    default_idx = available_instruments.index("US 10Y Future") if "US 10Y Future" in available_instruments else 0
    sensitivity_rate = st.sidebar.selectbox("Select sensitivity instrument:", available_instruments, index=default_idx)
    mode = st.sidebar.radio("Risk Attribution Mode:", ["By Breakdown","By Trade"])

    # Tabs
    tab_risk, tab_pos, tab_trades, tab_settings = st.tabs([
        "📊 Risk Attribution",
        "📂 Input Positions",
        "🛠 Trade Definitions",
        "⚙️ Settings"
    ])

    # --- Input Positions Tab ---
    with tab_pos:
        st.header("🔄 Input Positions")

        instruments_data = pd.DataFrame({
            "Ticker": [
                "GACGB2 Index","GACGB10 Index","TUAFWD Comdty","FVAFWD Comdty","TYAFWD Comdty",
                "UXYAFWD Comdty","WNAFWD Comdty","DUAFWD Comdty","OEAFWD Comdty","RXAFWD Comdty",
                "GAFWD Comdty","IKAFWD Comdty","CNAFWD Comdty","JBAFWD Comdty","CCSWNI1 Curncy",
                "ADSW2 Curncy","CDSO2 Curncy","USSW2 Curncy","EUSA2 Curncy","BPSWS2 BGN Curncy",
                "NDSWAP2 BGN Curncy","I39302Y Index","MPSW2B BGN Curncy","MPSWF2B Curncy",
                "SAFR1I2 BGN Curncy","CKSW2 BGN Curncy","PZSW2 BGN Curncy","KWSWNI2 BGN Curncy",
                "CCSWNI2 CMPN Curncy","ADSW5 Curncy","CDSO5 Curncy","USSW5 Curncy","EUSA5 Curncy",
                "BPSWS5 BGN Curncy","NDSWAP5 BGN Curncy","I39305Y Index","MPSW5E Curncy","MPSWF5E Curncy",
                "SASW5 Curncy","CKSW5 Curncy","PZSW5 Curncy","KWSWNI5 Curncy","CCSWNI5 Curncy",
                "JYSO5 Curncy","ADSW10 Curncy","CDSW10 Curncy","USSW10 Curncy","EUSA10 Curncy",
                "BPSWS10 BGN Curncy","NDSWAP10 BGN Curncy","ADSW30 Curncy","CDSW30 Curncy","USSW30 Curncy",
                "EUSA30 Curncy","BPSWS30 BGN Curncy","NDSWAP30 BGN Curncy","JYSO30 Curncy","MPSW10J BGN Curncy",
                "MPSWF10J BGN Curncy","SASW10 Curncy","CKSW10 BGN Curncy","PZSW10 BGN Curncy","KWSWNI10 BGN Curncy",
                "CCSWNI10 Curncy","BPSWIT10 Curncy","SKSW10 Curncy","NKSW10 Curncy"
            ],
            "Instrument Name": [
                "AU 3Y Future","AU 10Y Future","US 2Y Future","US 5Y Future","US 10Y Future",
                "US 10Y Ultra Future","US 30Y Future","DE 2Y Future","DE 5Y Future","DE 10Y Future",
                "UK 10Y Future","IT 10Y Future","CA 10Y Future","JP 10Y Future","CH 1Y Swap",
                "AU 2Y Swap","CA 2Y Swap","US 2Y Swap","DE 2Y Swap","UK 2Y Swap","NZ 2Y Swap",
                "BR 2Y Swap","MX 2Y Swap","MX 2Y Swap OIS","SA 2Y Swap","CZ 2Y Swap","PO 2Y Swap",
                "SK 2Y Swap","CH 2Y Swap","AU 5Y Swap","CA 5Y Swap","US 5Y Swap","DE 5Y Swap",
                "UK 5Y Swap","NZ 5Y Swap","BR 5Y Swap","MX 5Y Swap","MX 5Y Swap OIS","SA 5Y Swap",
                "CZ 5Y Swap","PO 5Y Swap","SK 5Y Swap","CH 5Y Swap","JP 5Y Swap","AU 10Y Swap",
                "CA 10Y Swap","US 10Y Swap","DE 10Y Swap","UK 10Y Swap","NZ 10Y Swap","AU 30Y Swap",
                "CA 30Y Swap","US 30Y Swap","DE 30Y Swap","UK 30Y Swap","NZ 30Y Swap","JP 30Y Swap",
                "MX 10Y Swap","MX 10Y Swap OIS","SA 10Y Swap","CZ 10Y Swap","PO 10Y Swap","SK 10Y Swap",
                "CH 10Y Swap","UK 10Y Swap Inf","SW 10Y Swap","NK 10Y Swap"
            ],
            "Portfolio": (
                ["DM"]*14 + ["EM"]*1 + ["DM"]*6 + ["EM"]*8 +
                ["DM"]*6 + ["EM"]*8 + ["DM"]*14 + ["EM"]*7 + ["DM"]*3
            )
        })

        dm_insts = instruments_data[instruments_data['Portfolio']=="DM"]['Instrument Name'].tolist()
        em_insts = instruments_data[instruments_data['Portfolio']=="EM"]['Instrument Name'].tolist()

        default_dm = pd.DataFrame({
            "Instrument": dm_insts,
            "Outright": [0.0]*len(dm_insts),
            "Curve":    [0.0]*len(dm_insts),
            "Spread":   [0.0]*len(dm_insts)
        })
        default_em = pd.DataFrame({
            "Instrument": em_insts,
            "Outright": [0.0]*len(em_insts),
            "Curve":    [0.0]*len(em_insts),
            "Spread":   [0.0]*len(em_insts)
        })

        st.subheader("📈 DM Portfolio Positions")
        gb_dm = GridOptionsBuilder.from_dataframe(default_dm)
        gb_dm.configure_default_column(editable=True, resizable=True)
        dm_resp = AgGrid(
            default_dm,
            gridOptions=gb_dm.build(),
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            height=400,
            key="dm_positions"
        )
        pos_dm = pd.DataFrame(dm_resp['data'])

        st.subheader("🌍 EM Portfolio Positions")
        gb_em = GridOptionsBuilder.from_dataframe(default_em)
        gb_em.configure_default_column(editable=True, resizable=True)
        em_resp = AgGrid(
            default_em,
            gridOptions=gb_em.build(),
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            height=400,
            key="em_positions"
        )
        pos_em = pd.DataFrame(em_resp['data'])

    # --- Trade Definitions Tab ---
    with tab_trades:
        st.header("🛠 Trade Definitions")

        if 'trade_defs' not in st.session_state:
            st.session_state.trade_defs = pd.DataFrame(
                columns=["Trade Name","Instrument","Position Type","Position"]
            )

        df_for_grid = st.session_state.trade_defs.reset_index(drop=True)
        df_for_grid['row_id'] = df_for_grid.index

        gb = GridOptionsBuilder.from_dataframe(df_for_grid)
        gb.configure_default_column(editable=True, resizable=True)
        gb.configure_column(
            "Instrument", cellEditor="agSelectCellEditor",
            cellEditorParams={"values":available_instruments}
        )
        gb.configure_column(
            "Position Type", cellEditor="agSelectCellEditor",
            cellEditorParams={"values":["Outright","Curve","Spread"]}
        )
        grid_response = AgGrid(
            df_for_grid,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            height=300,
            key="trade_defs_grid"
        )

        col_add, col_del, col_save = st.columns(3)
        with col_add:
            if st.button("➕ Add blank trade row"):
                new = pd.DataFrame([{
                    "Trade Name":"", "Instrument":available_instruments[0],
                    "Position Type":"Outright","Position":0.0
                }])
                st.session_state.trade_defs = pd.concat(
                    [st.session_state.trade_defs, new], ignore_index=True
                )
                st.experimental_rerun()
        with col_del:
            selected = grid_response.get("selected_rows",[])
            if st.button("❌ Delete selected rows") and selected:
                ids = [r["row_id"] for r in selected]
                st.session_state.trade_defs = st.session_state.trade_defs.drop(ids).reset_index(drop=True)
                st.experimental_rerun()
        with col_save:
            if st.button("💾 Save Trades"):
                df_new = pd.DataFrame(grid_response["data"]).drop(columns="row_id",errors="ignore")
                st.session_state.trade_defs = df_new.reset_index(drop=True)
                st.success("Trades updated.")

    # --- Settings Tab ---
    with tab_settings:
        st.header("⚙️ Settings")
        vol_opts = {
            '1 month (~21d)':21,'3 months (~63d)':63,'6 months (~126d)':126,
            '1 year (252d)':252,'3 years (~756d)':756,'5 years (~1260d)':1260
        }
        vol_label = st.selectbox("Volatility lookback:", list(vol_opts.keys()), index=3)
        vol_lookback = vol_opts[vol_label]

        var_opts = {'1 year (252d)':252,'5 years (~1260d)':1260}
        var_label = st.selectbox("VaR lookback:", list(var_opts.keys()), index=1)
        var_lookback = var_opts[var_label]

    # --- Risk Attribution Tab ---
    with tab_risk:
        st.header("📊 Risk Attribution Results")
        if st.button("🚀 Run Risk Attribution"):
            # Preprocess
            df = adjust_time_zones(raw_df, instrument_country)
            dc = calculate_daily_changes_in_bps(df)
            dc = fallback_mx_ois_data(dc)
            vols = calculate_volatilities(dc, vol_lookback)
            cov = calculate_covariance_matrix(dc, vol_lookback)

            # Build legs_df
            pos_dm['Portfolio']="DM"; pos_em['Portfolio']="EM"
            all_pos = pd.concat([pos_dm,pos_em],ignore_index=True)
            legs=[]
            for _,r in all_pos.iterrows():
                for pt in ["Outright","Curve","Spread"]:
                    v=r[pt]
                    if v!=0 and not pd.isna(v):
                        legs.append({
                            "Instrument":r["Instrument"],
                            "Position Type":pt,
                            "Position":v,
                            "Portfolio":r["Portfolio"]
                        })
            legs_df = pd.DataFrame(legs)
            if legs_df.empty:
                st.warning("No active positions."); return
            vec = legs_df.set_index(["Instrument","Position Type"])["Position"]

            insts = vec.index.get_level_values("Instrument").unique()
            missing = [i for i in insts if i not in cov.index]
            if missing:
                drop=[(i,pt) for i in missing for pt in ["Outright","Curve","Spread"]]
                vec=vec.drop(labels=drop,errors="ignore")
                insts=vec.index.get_level_values("Instrument").unique()
            cov_sub=cov.loc[insts,insts]
            order=vec.index.get_level_values("Instrument").to_numpy()
            cov_mat=pd.DataFrame(cov_sub.loc[order,order].values,
                                 index=vec.index,columns=vec.index)

            pvar=np.dot(vec.values,cov_mat.values.dot(vec.values))
            pvol=np.sqrt(pvar) if pvar>0 else np.nan

            marg=cov_mat.dot(vec)
            cvar=vec*marg
            cvol=cvar/pvol
            legs_df["Contr to Vol (bps)"]=cvol.values
            legs_df["Country"]=legs_df["Instrument"].apply(guess_country)

            # VaR & cVaR
            pr_var=dc.tail(var_lookback)
            pos_by_inst=vec.groupby("Instrument").sum()
            common=pos_by_inst.index.intersection(pr_var.columns)
            VaR95=VaR99=cVaR95=cVaR99=np.nan
            if not common.empty:
                pr2=pr_var[common]; pb=pos_by_inst.loc[common]
                port_ret=pr2.dot(pb)
                VaR95,VaR99=-np.percentile(port_ret,[5,1])
                if (port_ret<=-VaR95).any(): cVaR95=-port_ret[port_ret<=-VaR95].mean()
                if (port_ret<=-VaR99).any(): cVaR99=-port_ret[port_ret<=-VaR99].mean()

            inst_c95={}; inst_c99={}
            if not np.isnan(VaR95):
                for inst,pos in pos_by_inst.items():
                    loss=-pr2[inst]*pos
                    m95=loss<=-VaR95; m99=loss<=-VaR99
                    inst_c95[inst]=loss[m95].mean() if m95.any() else np.nan
                    inst_c99[inst]=loss[m99].mean() if m99.any() else np.nan

            individual_cvar_95={}; individual_cvar_99={}
            for inst,pos in pos_by_inst.items():
                if inst in pr2:
                    ls=-pr2[inst]*pos
                    if not ls.empty:
                        v95=np.percentile(ls,95); v99=np.percentile(ls,99)
                        individual_cvar_95[inst]=ls[ls>=v95].mean() if (ls>=v95).any() else np.nan
                        individual_cvar_99[inst]=ls[ls>=v99].mean() if (ls>=v99).any() else np.nan

            # Waterfalls & bar
            vol_inst=legs_df.groupby("Instrument")["Contr to Vol (bps)"].sum().reset_index()
            vol_inst['abs']=vol_inst["Contr to Vol (bps)"].abs()
            vol_inst=vol_inst.sort_values("abs",ascending=False)
            fig_vol_inst=create_waterfall_chart(
                vol_inst["Instrument"].tolist(),
                vol_inst["Contr to Vol (bps)"].tolist(),
                pvol,"Volatility Contributions by Instrument",False
            )
            cb=legs_df.groupby(["Country","Position Type"])["Contr to Vol (bps)"].sum().reset_index()
            cb["Group"]=cb["Country"]+" - "+cb["Position Type"]
            cb['abs']=cb["Contr to Vol (bps)"].abs()
            cb=cb.sort_values("abs",ascending=False)
            fig_vol_group=create_waterfall_chart(
                cb["Group"].tolist(),
                cb["Contr to Vol (bps)"].tolist(),
                pvol,"Volatility Contributions by Country & Bucket",False
            )
            fig_cvar95=create_waterfall_chart(
                list(inst_c95.keys()),list(inst_c95.values()),cVaR95,
                "cVaR (95%) Contributions by Instrument"
            )
            fig_cvar99=create_waterfall_chart(
                list(inst_c99.keys()),list(inst_c99.values()),cVaR99,
                "cVaR (99%) Contributions by Instrument"
            )
            cdf_m=pd.DataFrame({
                "Instrument":list(individual_cvar_95.keys()),
                "cVaR 95":[individual_cvar_95[k] for k in individual_cvar_95],
                "cVaR 99":[individual_cvar_99[k] for k in individual_cvar_99]
            }).melt(id_vars="Instrument",var_name="Confidence Level",value_name="cVaR")
            fig_instrument_cvar=px.bar(
                cdf_m,x="Instrument",y="cVaR",
                color="Confidence Level",barmode="group",
                title="Standalone Instrument cVaR (95% & 99%)"
            )

            # --- PRESENTATION IN ORIGINAL ORDER ---

            st.subheader("Risk Attribution by Instrument (Volatility)")
            st.plotly_chart(fig_vol_inst, use_container_width=True)

            st.subheader("Risk Attribution by Country & Bucket (Volatility)")
            st.plotly_chart(fig_vol_group, use_container_width=True)
            st.dataframe(cb[["Country","Position Type","Contr to Vol (bps)"]])

            st.subheader("cVaR (95%) Contributions by Instrument")
            st.plotly_chart(fig_cvar95, use_container_width=True)

            st.subheader("cVaR (99%) Contributions by Instrument")
            st.plotly_chart(fig_cvar99, use_container_width=True)

            st.subheader("Standalone Instrument cVaR (95% & 99%)")
            st.plotly_chart(fig_instrument_cvar, use_container_width=True)

            c1,c2,c3,c4=st.columns(4)
            c1.metric("📊 Total Portfolio Volatility",fmt_val(pvol))
            c2.metric("📉 Daily VaR (95%)",fmt_val(VaR95))
            c3.metric("📉 Daily VaR (99%)",fmt_val(VaR99))
            c4.metric("📈 Daily cVaR (95%)",fmt_val(cVaR95))

            st.subheader("📈 Value at Risk (VaR) and Conditional VaR (cVaR)")
            st.write(f"**Daily VaR at 95%:** {fmt_val(VaR95)}")
            st.write(f"**Daily cVaR at 95%:** {fmt_val(cVaR95)}")
            st.write(f"**Daily VaR at 99%:** {fmt_val(VaR99)}")
            st.write(f"**Daily cVaR at 99%:** {fmt_val(cVaR99)}")

            st.subheader("📉 Beta to US 10yr Rates (Daily Basis)")
            # … insert your beta logic here …

            st.subheader("📄 Detailed Risk Contributions by Instrument")
            AgGrid(
                legs_df[[
                    "Instrument","Position Type","Position","Contr to Vol (bps)","Country","Portfolio"
                ]].round(4),
                gridOptions=GridOptionsBuilder.from_dataframe(
                    legs_df[[
                        "Instrument","Position Type","Position","Contr to Vol (bps)","Country","Portfolio"
                    ]].round(4)
                ).build(),
                fit_columns_on_grid_load=True,
                height=400,
                key="detailed_risk_grid"
            )
            csv=legs_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV",data=csv,file_name="risk_contributions.csv",mime="text/csv")

if __name__ == "__main__":
    main()
