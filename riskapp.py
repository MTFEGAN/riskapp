import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go  # Needed for waterfall charts
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="📈 BMIX Portfolio Risk Attribution", layout="wide")

# Helper: simple formatting for metrics
def fmt_val(x):
    return f"{x:.2f} bps" if (not np.isnan(x) and not np.isinf(x)) else "N/A"

# Load historical data from Excel
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
        # Remove weekends
        df = df[~df.index.dayofweek.isin([5, 6])]
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return pd.DataFrame()

# Adjust time zones by lagging non-local markets
def adjust_time_zones(df, instrument_country):
    non_lag_countries = ['JP', 'AU', 'SK', 'CH']
    instrument_countries = pd.Series(
        [instrument_country.get(instr, 'Other') for instr in df.columns], index=df.columns
    )
    instruments_to_lag = instrument_countries[~instrument_countries.isin(non_lag_countries)].index.tolist()
    adjusted_df = df.copy()
    if instruments_to_lag:
        adjusted_df[instruments_to_lag] = adjusted_df[instruments_to_lag].shift(1)
    return adjusted_df.dropna()

# Calculate daily changes in bps
def calculate_daily_changes_in_bps(df):
    return df.diff().dropna() * 100

# Fallback for MX OIS missing data
def fallback_mx_ois_data(daily_changes):
    ois_map = {
        'MX 2Y Swap OIS': 'MX 2Y Swap',
        'MX 5Y Swap OIS': 'MX 5Y Swap',
        'MX 10Y Swap OIS': 'MX 10Y Swap'
    }
    for ois_col, non_ois_col in ois_map.items():
        if ois_col in daily_changes.columns and non_ois_col in daily_changes.columns:
            daily_changes[ois_col] = daily_changes[ois_col].fillna(daily_changes[non_ois_col])
    return daily_changes

# Volatility and covariance
def calculate_volatilities(daily_changes, lookback_days):
    if daily_changes.empty:
        return pd.Series(dtype=float)
    recent = daily_changes.tail(lookback_days)
    if recent.empty:
        return pd.Series(dtype=float)
    return recent.std() * np.sqrt(252)

def calculate_covariance_matrix(daily_changes, lookback_days):
    if daily_changes.empty:
        return pd.DataFrame()
    recent = daily_changes.tail(lookback_days)
    if recent.empty:
        return pd.DataFrame()
    return recent.cov() * 252

# Compute beta
def compute_beta(x_returns, y_returns, lookback_days):
    if x_returns.empty or y_returns.empty:
        return np.nan
    common = x_returns.index.intersection(y_returns.index)
    if common.empty:
        return np.nan
    x = x_returns.loc[common].tail(lookback_days)
    y = y_returns.loc[common].tail(lookback_days)
    if x.empty or y.empty or x.std()==0 or y.std()==0:
        return np.nan
    cov = np.cov(x, y)[0,1]
    var_y = np.var(y)
    return cov/var_y if var_y!=0 else np.nan

# Guess country code
def guess_country_from_instrument_name(name):
    country_codes = ['AU','US','DE','UK','IT','CA','JP','CH','BR','MX','SA','CZ','PO','SK','NZ','SW','NK']
    for code in country_codes:
        if code in name:
            return code
    return 'Other'

# Waterfall chart helper
def create_waterfall_chart(labels, values, total, title, include_diversification=False):
    if include_diversification:
        diversification = total - sum(values)
        measures = ['relative']*len(values) + ['relative','total']
        x = labels + ['Diversification Impact','Total']
        y = values + [diversification, total]
        text = [f"{v:.2f}" for v in values] + [f"{diversification:.2f}", f"{total:.2f}"]
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

# Comprehensive instrument-country mapping
instrument_country = {
    "AU 3Y Future": "AU", "AU 10Y Future": "AU", "US 2Y Future": "US", "US 5Y Future": "US",
    "US 10Y Future": "US", "US 10Y Ultra Future": "US", "US 30Y Future": "US", "DE 2Y Future": "DE",
    "DE 5Y Future": "DE", "DE 10Y Future": "DE", "UK 10Y Future": "UK", "IT 10Y Future": "IT",
    "CA 10Y Future": "CA", "JP 10Y Future": "JP", "CH 1Y Swap": "CH", "AU 2Y Swap": "AU",
    "CA 2Y Swap": "CA", "US 2Y Swap": "US", "DE 2Y Swap": "DE", "UK 2Y Swap": "UK",
    "NZ 2Y Swap": "NZ", "BR 2Y Swap": "BR", "MX 2Y Swap": "MX", "MX 2Y Swap OIS": "MX",
    "SA 2Y Swap": "SA", "CZ 2Y Swap": "CZ", "PO 2Y Swap": "PO", "SK 2Y Swap": "SK",
    "CH 2Y Swap": "CH", "AU 5Y Swap": "AU", "CA 5Y Swap": "CA", "US 5Y Swap": "US",
    "DE 5Y Swap": "DE", "UK 5Y Swap": "UK", "NZ 5Y Swap": "NZ", "BR 5Y Swap": "BR",
    "MX 5Y Swap": "MX", "MX 5Y Swap OIS": "MX", "SA 5Y Swap": "SA", "CZ 5Y Swap": "CZ",
    "PO 5Y Swap": "PO", "SK 5Y Swap": "SK", "CH 5Y Swap": "CH", "JP 5Y Swap": "JP",
    "AU 10Y Swap": "AU", "CA 10Y Swap": "CA", "US 10Y Swap": "US", "DE 10Y Swap": "DE",
    "UK 10Y Swap": "UK", "NZ 10Y Swap": "NZ", "AU 30Y Swap": "AU", "CA 30Y Swap": "CA",
    "US 30Y Swap": "US", "DE 30Y Swap": "DE", "UK 30Y Swap": "UK", "NZ 30Y Swap": "NZ",
    "JP 30Y Swap": "JP", "MX 10Y Swap": "MX", "MX 10Y Swap OIS": "MX", "SA 10Y Swap": "SA",
    "CZ 10Y Swap": "CZ", "PO 10Y Swap": "PO", "SK 10Y Swap": "SK", "CH 10Y Swap": "CH",
    "UK 10Y Swap Inf": "UK", "SW 10Y Swap": "SW", "NK 10Y Swap": "NK"
}

def main():
    st.title('📈 BMIX Portfolio Risk Attribution')
    st.write("App initialized successfully.")

    # Prepare instrument list and default positions
    instruments_data = pd.DataFrame({
        "Ticker": [
            "GACGB2 Index", "GACGB10 Index", "TUAFWD Comdty", "FVAFWD Comdty",
            "TYAFWD Comdty", "UXYAFWD Comdty", "WNAFWD Comdty", "DUAFWD Comdty",
            "OEAFWD Comdty", "RXAFWD Comdty", "GAFWD Comdty", "IKAFWD Comdty",
            "CNAFWD Comdty", "JBAFWD Comdty", "CCSWNI1 Curncy", "ADSW2 Curncy",
            "CDSO2 Curncy", "USSW2 Curncy", "EUSA2 Curncy", "BPSWS2 BGN Curncy",
            "NDSWAP2 BGN Curncy", "I39302Y Index", "MPSW2B BGN Curncy", "MPSWF2B Curncy",
            "SAFR1I2 BGN Curncy", "CKSW2 BGN Curncy", "PZSW2 BGN Curncy",
            "KWSWNI2 BGN Curncy", "CCSWNI2 CMPN Curncy", "ADSW5 Curncy", "CDSO5 Curncy",
            "USSW5 Curncy", "EUSA5 Curncy", "BPSWS5 BGN Curncy", "NDSWAP5 BGN Curncy",
            "I39305Y Index", "MPSW5E Curncy", "MPSWF5E Curncy", "SASW5 Curncy",
            "CKSW5 Curncy", "PZSW5 Curncy", "KWSWNI5 Curncy", "CCSWNI5 Curncy",
            "JYSO5 Curncy", "ADSW10 Curncy", "CDSO10 Curncy", "USSW10 Curncy",
            "EUSA10 Curncy", "BPSWS10 BGN Curncy", "NDSWAP10 BGN Curncy",
            "ADSW30 Curncy", "CDSO30 Curncy", "USSW30 Curncy", "EUSA30 Curncy",
            "BPSWS30 BGN Curncy", "NDSWAP30 BGN Curncy", "JYSO30 Curncy",
            "MPSW10J BGN Curncy", "MPSWF10J BGN Curncy", "SASW10 Curncy",
            "CKSW10 BGN Curncy", "PZSW10 BGN Curncy", "KWSWNI10 BGN Curncy",
            "CCSWNI10 Curncy", "BPSWIT10 Curncy", "SKSW10 Curncy", "NKSW10 Curncy"
        ],
        "Instrument Name": [
            "AU 3Y Future", "AU 10Y Future", "US 2Y Future", "US 5Y Future",
            "US 10Y Future", "US 10Y Ultra Future", "US 30Y Future", "DE 2Y Future",
            "DE 5Y Future", "DE 10Y Future", "UK 10Y Future", "IT 10Y Future",
            "CA 10Y Future", "JP 10Y Future", "CH 1Y Swap", "AU 2Y Swap",
            "CA 2Y Swap", "US 2Y Swap", "DE 2Y Swap", "UK 2Y Swap", "NZ 2Y Swap",
            "BR 2Y Swap", "MX 2Y Swap", "MX 2Y Swap OIS", "SA 2Y Swap", "CZ 2Y Swap",
            "PO 2Y Swap", "SK 2Y Swap", "CH 2Y Swap", "AU 5Y Swap", "CA 5Y Swap",
            "US 5Y Swap", "DE 5Y Swap", "UK 5Y Swap", "NZ 5Y Swap", "BR 5Y Swap",
            "MX 5Y Swap", "MX 5Y Swap OIS", "SA 5Y Swap", "CZ 5Y Swap", "PO 5Y Swap",
            "SK 5Y Swap", "CH 5Y Swap", "JP 5Y Swap", "AU 10Y Swap", "CA 10Y Swap",
            "US 10Y Swap", "DE 10Y Swap", "UK 10Y Swap", "NZ 10Y Swap", "AU 30Y Swap",
            "CA 30Y Swap", "US 30Y Swap", "DE 30Y Swap", "UK 30Y Swap", "NZ 30Y Swap",
            "JP 30Y Swap", "MX 10Y Swap", "MX 10Y Swap OIS", "SA 10Y Swap", "CZ 10Y Swap",
            "PO 10Y Swap", "SK 10Y Swap", "CH 10Y Swap", "UK 10Y Swap Inf", "SW 10Y Swap",
            "NK 10Y Swap"
        ],
        "Portfolio": (
            ["DM"]*14 + ["EM"]*1 + ["DM"]*6 + ["EM"]*8 +
            ["DM"]*6 + ["EM"]*8 + ["DM"]*14 + ["EM"]*7 + ["DM"]*3
        )
    })

    dm_instruments = instruments_data[instruments_data['Portfolio']=='DM']['Instrument Name'].tolist()
    em_instruments = instruments_data[instruments_data['Portfolio']=='EM']['Instrument Name'].tolist()
    default_positions_dm = pd.DataFrame({
        'Instrument': dm_instruments,
        'Outright': [0.0]*len(dm_instruments),
        'Curve':    [0.0]*len(dm_instruments),
        'Spread':   [0.0]*len(dm_instruments)
    })
    default_positions_em = pd.DataFrame({
        'Instrument': em_instruments,
        'Outright': [0.0]*len(em_instruments),
        'Curve':    [0.0]*len(em_instruments),
        'Spread':   [0.0]*len(em_instruments)
    })

    # Sidebar: Sensitivity selection and file check
    st.sidebar.header("🔍 Sensitivity Rate Configuration")
    excel_file = 'historical_data.xlsx'
    if not os.path.exists(excel_file):
        st.sidebar.error(f"❌ '{excel_file}' not found.")
        st.stop()
    raw_df = load_historical_data(excel_file)
    if raw_df.empty:
        st.error("No data loaded from Excel.")
        st.stop()
    available_columns = raw_df.columns.tolist()
    default_index = available_columns.index('US 10Y Future') if 'US 10Y Future' in available_columns else 0
    sensitivity_rate = st.sidebar.selectbox(
        'Select sensitivity instrument:',
        options=available_columns,
        index=default_index
    )

    # Create tabs
    tabs = st.tabs([
        "📊 Risk Attribution",
        "📂 Input Positions",
        "🛠 Trade Definitions",
        "⚙️ Settings"
    ])

    # Input Positions tab
    with tabs[1]:
        st.header("🔄 Input Positions")
        st.subheader("📈 DM Portfolio Positions")
        gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
        gb_dm.configure_default_column(editable=True, resizable=True)
        dm_opts = gb_dm.build()
        dm_resp = AgGrid(
            default_positions_dm,
            gridOptions=dm_opts,
            height=600, width='100%',
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True
        )
        positions_data_dm = dm_resp['data']

        st.subheader("🌍 EM Portfolio Positions")
        gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
        gb_em.configure_default_column(editable=True, resizable=True)
        em_opts = gb_em.build()
        em_resp = AgGrid(
            default_positions_em,
            gridOptions=em_opts,
            height=600, width='100%',
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True
        )
        positions_data_em = em_resp['data']

    # Trade Definitions tab
    with tabs[2]:
        st.header("🛠 Trade Definitions")
        # Pre-populate 4 blank rows so you can immediately start typing
        default_trades = pd.DataFrame([
            {'Trade Name':'', 'Instrument':'', 'Position Type':'', 'Position':np.nan}
            for _ in range(4)
        ])
        gb_tr = GridOptionsBuilder.from_dataframe(default_trades)
        gb_tr.configure_default_column(editable=True, resizable=True)
        tr_opts = gb_tr.build()
        tr_resp = AgGrid(
            default_trades,
            gridOptions=tr_opts,
            height=400, width='100%',
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True
        )
        trade_defs = pd.DataFrame(tr_resp['data'])

    # Settings tab
    with tabs[3]:
        st.header("⚙️ Configuration Settings")
        volatility_period_options = {
            '📅 1 month (~21 days)': 21,
            '📆 3 months (~63 days)': 63,
            '📅 6 months (~126 days)': 126,
            '🗓️ 1 year (252 days)': 252,
            '📅 3 years (756 days)': 756,
            '📆 5 years (1260 days)': 1260
        }
        volatility_period = st.selectbox(
            'Volatility lookback:',
            list(volatility_period_options.keys()),
            index=3
        )
        volatility_lookback_days = volatility_period_options[volatility_period]

        var_period_options = {
            '🗓️ 1 year (252 days)': 252,
            '📆 5 years (~1260 days)': 1260
        }
        var_period = st.selectbox(
            'VaR lookback:',
            list(var_period_options.keys()),
            index=1
        )
        var_lookback_days = var_period_options[var_period]

    # Risk Attribution tab
    with tabs[0]:
        st.header("📊 Risk Attribution Results")
        if st.button('🚀 Run Risk Attribution'):
            # --- Load & preprocess data ---
            df = load_historical_data(excel_file)
            if df.empty:
                st.error("No data loaded. Check Excel file.")
                st.stop()
            df = adjust_time_zones(df, instrument_country)
            daily_changes = calculate_daily_changes_in_bps(df)
            daily_changes = fallback_mx_ois_data(daily_changes)

            volatilities = calculate_volatilities(daily_changes, volatility_lookback_days)
            covariance_matrix = calculate_covariance_matrix(daily_changes, volatility_lookback_days)
            beta_data_window = daily_changes.tail(volatility_lookback_days)

            # Build positions vector
            pos_dm = pd.DataFrame(positions_data_dm).astype({
                'Outright': float, 'Curve': float, 'Spread': float
            })
            pos_dm['Portfolio'] = 'DM'
            pos_em = pd.DataFrame(positions_data_em).astype({
                'Outright': float, 'Curve': float, 'Spread': float
            })
            pos_em['Portfolio'] = 'EM'
            positions_data = pd.concat([pos_dm, pos_em], ignore_index=True)

            positions_list = []
            for _, row in positions_data.iterrows():
                for pt in ['Outright', 'Curve', 'Spread']:
                    val = row[pt]
                    if val != 0 and not pd.isna(val):
                        positions_list.append({
                            'Instrument': row['Instrument'],
                            'Position Type': pt,
                            'Position': val,
                            'Portfolio': row['Portfolio']
                        })
            expanded_positions_data = pd.DataFrame(positions_list)
            if expanded_positions_data.empty:
                st.warning("No active positions entered.")
                st.stop()
            expanded_positions_vector = expanded_positions_data.set_index(
                ['Instrument', 'Position Type']
            )['Position']

            # Covariance submatrix
            instrs = expanded_positions_vector.index.get_level_values('Instrument').unique()
            missing_instruments = [
                i for i in instrs if i not in covariance_matrix.index
            ]
            if missing_instruments:
                expanded_positions_vector = expanded_positions_vector.drop(
                    labels=[(i,pt) for i in missing_instruments for pt in ['Outright','Curve','Spread']],
                    errors='ignore'
                )
                instrs = expanded_positions_vector.index.get_level_values('Instrument').unique()
            cov_sub = covariance_matrix.loc[instrs, instrs]
            order = expanded_positions_vector.index.get_level_values('Instrument').to_numpy()
            cov_vals = cov_sub.loc[order, order].values
            expanded_cov_matrix = pd.DataFrame(
                cov_vals, index=expanded_positions_vector.index, columns=expanded_positions_vector.index
            )

            # Portfolio variance & volatility
            port_var = np.dot(
                expanded_positions_vector.values,
                np.dot(expanded_cov_matrix.values, expanded_positions_vector.values)
            )
            portfolio_volatility = np.sqrt(port_var) if port_var > 0 else np.nan

            # Standalone & marginal contributions
            insts = expanded_positions_vector.index.get_level_values('Instrument')
            exp_vols = pd.Series(insts).map(volatilities.to_dict())
            exp_vols.index = expanded_positions_vector.index
            standalone_volatilities = expanded_positions_vector.abs() * exp_vols
            marg = expanded_cov_matrix.dot(expanded_positions_vector)
            contrib_var = expanded_positions_vector * marg
            contrib_vol = contrib_var / portfolio_volatility
            pct_contrib = contrib_var / port_var * 100

            # Build risk contributions DataFrame
            rc = expanded_positions_data.copy()
            rc['Position Stand-alone Volatility'] = standalone_volatilities.values
            rc['Contribution to Volatility (bps)'] = contrib_vol.values
            rc['Percent Contribution (%)'] = pct_contrib.values
            rc['Instrument Volatility per 1Y Duration (bps)'] = exp_vols.values
            rc['Country'] = rc['Instrument'].apply(guess_country_from_instrument_name)
            risk_contributions_formatted = rc[[
                'Instrument','Position Type','Position',
                'Position Stand-alone Volatility',
                'Instrument Volatility per 1Y Duration (bps)',
                'Contribution to Volatility (bps)',
                'Percent Contribution (%)','Country','Portfolio'
            ]].round(2)

            # VaR & cVaR
            VaR_95=VaR_99=cVaR_95=cVaR_99=np.nan
            pr_var = daily_changes.tail(var_lookback_days)
            pos_for_var = expanded_positions_vector.groupby('Instrument').sum()
            avail = pos_for_var.index.intersection(pr_var.columns)
            if not avail.empty:
                pr_var = pr_var[avail]
                pos_for_var = pos_for_var.loc[avail]
                port_ret = pr_var.dot(pos_for_var)
                VaR_95 = -np.percentile(port_ret,5)
                VaR_99 = -np.percentile(port_ret,1)
                if (port_ret <= -VaR_95).any():
                    cVaR_95 = -port_ret[port_ret <= -VaR_95].mean()
                if (port_ret <= -VaR_99).any():
                    cVaR_99 = -port_ret[port_ret <= -VaR_99].mean()

            # Instrument-level cVaR contributions
            instrument_contrib_95 = {}
            instrument_contrib_99 = {}
            for instr, pos in pos_for_var.items():
                loss = -pr_var[instr] * pos
                mask95 = loss <= -VaR_95
                mask99 = loss <= -VaR_99
                instrument_contrib_95[instr] = loss[mask95].mean() if mask95.any() else np.nan
                instrument_contrib_99[instr] = loss[mask99].mean() if mask99.any() else np.nan

            # Enforce trade assignment
            td = trade_defs.dropna(
                subset=['Trade Name','Instrument','Position Type','Position']
            ).copy()
            td['Position'] = td['Position'].astype(float)
            pos_keys = set(expanded_positions_data.apply(
                lambda r:(r['Instrument'],r['Position Type'],r['Position']), axis=1
            ))
            trade_keys = set(td.apply(
                lambda r:(r['Instrument'],r['Position Type'],r['Position']), axis=1
            ))
            missing = pos_keys - trade_keys
            if missing:
                st.error(f"❌ The following positions are not assigned to trades: {missing}")
                st.stop()

            # Trade-level volatility contributions
            trade_vol_df = (
                risk_contributions_formatted
                .merge(td[['Trade Name','Instrument','Position Type','Position']],
                       on=['Instrument','Position Type','Position'], how='left')
                .groupby('Trade Name')['Contribution to Volatility (bps)']
                .sum().reset_index()
            )
            fig_trade_vol = create_waterfall_chart(
                trade_vol_df['Trade Name'].tolist(),
                trade_vol_df['Contribution to Volatility (bps)'].tolist(),
                portfolio_volatility,
                "Volatility Contributions by Trade"
            )

            # Trade-level cVaR contributions
            instr_trade_map = td.set_index(
                ['Instrument','Position Type','Position']
            )['Trade Name'].to_dict()
            trade_c95 = {}
            trade_c99 = {}
            for instr, contrib in instrument_contrib_95.items():
                for key,val in instr_trade_map.items():
                    if key[0] == instr:
                        trade_c95[val] = trade_c95.get(val, 0) + contrib
            for instr, contrib in instrument_contrib_99.items():
                for key,val in instr_trade_map.items():
                    if key[0] == instr:
                        trade_c99[val] = trade_c99.get(val, 0) + contrib
            fig_trade_c95 = create_waterfall_chart(
                list(trade_c95.keys()), list(trade_c95.values()), cVaR_95,
                "cVaR (95%) Contributions by Trade"
            )
            fig_trade_c99 = create_waterfall_chart(
                list(trade_c99.keys()), list(trade_c99.values()), cVaR_99,
                "cVaR (99%) Contributions by Trade"
            )

            # Display trade-level charts
            st.subheader("Risk Attribution by Trade (Volatility)")
            st.plotly_chart(fig_trade_vol, use_container_width=True)
            st.subheader("cVaR (95%) Contributions by Trade")
            st.plotly_chart(fig_trade_c95, use_container_width=True)
            st.subheader("cVaR (99%) Contributions by Trade")
            st.plotly_chart(fig_trade_c99, use_container_width=True)

            # Instrument-level charts & tables (original)
            fig_vol_inst = create_waterfall_chart(
                risk_contributions_formatted['Instrument'].tolist(),
                risk_contributions_formatted['Contribution to Volatility (bps)'].tolist(),
                portfolio_volatility,
                "Volatility Contributions by Instrument"
            )
            st.subheader("Risk Attribution by Instrument (Volatility)")
            st.plotly_chart(fig_vol_inst, use_container_width=True)

            country_bucket = risk_contributions_formatted.groupby(
                ['Country','Position Type']
            )['Contribution to Volatility (bps)'].sum().reset_index()
            if not country_bucket.empty:
                country_bucket['Group'] = country_bucket['Country'] + " - " + country_bucket['Position Type']
                country_bucket['abs'] = country_bucket['Contribution to Volatility (bps)'].abs()
                country_bucket = country_bucket.sort_values(by='abs', ascending=False)
                fig_vol_group = create_waterfall_chart(
                    country_bucket['Group'].tolist(),
                    country_bucket['Contribution to Volatility (bps)'].tolist(),
                    portfolio_volatility,
                    "Volatility Contributions by Country & Bucket"
                )
                st.subheader("Risk Attribution by Country & Bucket (Volatility)")
                st.plotly_chart(fig_vol_group, use_container_width=True)
                st.dataframe(country_bucket[['Country','Position Type','Contribution to Volatility (bps)']])
            else:
                st.subheader("Risk Attribution by Country & Bucket (Volatility)")
                st.write("No country/bucket data to display.")

            fig_cvar95 = create_waterfall_chart(
                list(instrument_contrib_95.keys()),
                list(instrument_contrib_95.values()),
                cVaR_95,
                "cVaR (95%) Contributions by Instrument"
            )
            st.subheader("cVaR (95%) Contributions by Instrument")
            st.plotly_chart(fig_cvar95, use_container_width=True)

            fig_cvar99 = create_waterfall_chart(
                list(instrument_contrib_99.keys()),
                list(instrument_contrib_99.values()),
                cVaR_99,
                "cVaR (99%) Contributions by Instrument"
            )
            st.subheader("cVaR (99%) Contributions by Instrument")
            st.plotly_chart(fig_cvar99, use_container_width=True)

            cvar_instrument_df = pd.DataFrame({
                "Instrument": list(instrument_contrib_95.keys()),
                "cVaR 95": [instrument_contrib_95[i] for i in instrument_contrib_95],
                "cVaR 99": [instrument_contrib_99[i] for i in instrument_contrib_99]
            })
            cvar_instrument_df_melt = cvar_instrument_df.melt(
                id_vars=["Instrument"],
                value_vars=["cVaR 95","cVaR 99"],
                var_name="Confidence Level",
                value_name="cVaR"
            )
            cvar_instrument_df_melt["cVaR"] = cvar_instrument_df_melt["cVaR"].abs()
            fig_instrument_cvar = px.bar(
                cvar_instrument_df_melt, x="Instrument", y="cVaR",
                color="Confidence Level", barmode="group",
                title="Standalone Instrument cVaR (95% & 99%)"
            )
            st.subheader("Standalone Instrument cVaR (95% & 99%)")
            st.plotly_chart(fig_instrument_cvar, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📊 Total Portfolio Volatility", fmt_val(portfolio_volatility))
            m2.metric("📉 Daily VaR (95%)", fmt_val(VaR_95))
            m3.metric("📉 Daily VaR (99%)", fmt_val(VaR_99))
            m4.metric("📈 Daily cVaR (95%)", fmt_val(cVaR_95))

            st.subheader("📈 Value at Risk (VaR) and Conditional VaR (cVaR)")
            st.write(f"**Daily VaR at 95%:** {fmt_val(VaR_95)}")
            st.write(f"**Daily cVaR at 95%:** {fmt_val(cVaR_95)}")
            st.write(f"**Daily VaR at 99%:** {fmt_val(VaR_99)}")
            st.write(f"**Daily cVaR at 99%:** {fmt_val(cVaR_99)}")

            # Beta calculations
            portfolio_beta = np.nan
            portfolio_r2 = np.nan
            instrument_betas = {}
            if sensitivity_rate in beta_data_window.columns:
                us_returns = beta_data_window[sensitivity_rate]
                port_returns = beta_data_window[pos_for_var.index].dot(pos_for_var)
                portfolio_beta = compute_beta(port_returns, us_returns, volatility_lookback_days)
                common_idx = port_returns.index.intersection(us_returns.index)
                if len(common_idx)>1:
                    corr = np.corrcoef(port_returns.loc[common_idx], us_returns.loc[common_idx])[0,1]
                    portfolio_r2 = corr**2
                for instr in pos_for_var.index:
                    instr_ret = beta_data_window[instr]
                    instr_beta = compute_beta(instr_ret, us_returns, volatility_lookback_days)
                    if not np.isnan(instr_beta):
                        instrument_betas[instr] = (float(pos_for_var[instr]), instr_beta)
            st.subheader("📉 Beta to US 10yr Rates (Daily Basis)")
            if not np.isnan(portfolio_beta):
                st.write(f"**Portfolio Beta to {sensitivity_rate} (Daily):** {portfolio_beta:.4f}")
                st.write(f"**Portfolio R² with {sensitivity_rate} (Daily):** {portfolio_r2:.4f}")
                if instrument_betas:
                    beta_df = pd.DataFrame([
                        {'Instrument':k, 'Position':v[0], 'Beta':v[1]} 
                        for k,v in instrument_betas.items()
                    ])
                    beta_df['Position'] = beta_df['Position'].round(2)
                    beta_df['Beta'] = beta_df['Beta'].round(4)
                    st.dataframe(beta_df)
                    st.markdown("*Footnote:* If the US 10-year yield moves by 1bp in a day, the portfolio changes by approximately Beta × 1bp in that day.")
                else:
                    st.write("No individual instrument betas to display.")
            else:
                st.write("No portfolio beta computed. Check data and positions.")

            # Detailed table & download
            if not risk_contributions_formatted.empty:
                st.subheader("📄 Detailed Risk Contributions by Instrument")
                gb_rc = GridOptionsBuilder.from_dataframe(risk_contributions_formatted)
                gb_rc.configure_default_column(editable=False, resizable=True)
                rc_opts = gb_rc.build()
                AgGrid(
                    risk_contributions_formatted,
                    gridOptions=rc_opts,
                    height=400, width='100%',
                    enable_enterprise_modules=False,
                    fit_columns_on_grid_load=True
                )
                csv = risk_contributions_formatted.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Risk Contributions as CSV",
                    data=csv,
                    file_name='risk_contributions.csv',
                    mime='text/csv',
                )

if __name__ == '__main__':
    main()
