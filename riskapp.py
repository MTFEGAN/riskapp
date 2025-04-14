import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go  # Needed for waterfall charts
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="📈 BMIX Portfolio Risk Attribution", layout="wide")

def load_historical_data(excel_file):
    """
    Load historical yield data from the specified Excel file.
    Combine all sheets and ensure the index is datetime.
    Remove weekends from the data.
    """
    try:
        excel = pd.ExcelFile(excel_file)
        df_list = []
        for sheet in excel.sheet_names:
            df_sheet = excel.parse(sheet_name=sheet, index_col='date', parse_dates=True)
            df_list.append(df_sheet)
        df = pd.concat(df_list, axis=1)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.dayofweek.isin([5, 6])]
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return pd.DataFrame()

def adjust_time_zones(df, instrument_country):
    """
    Adjust time zones by shifting certain instruments by one business day.
    Non-lag countries: JP, AU, SK, CH (no shift).
    All other instruments are shifted forward by one day to lag data.
    """
    non_lag_countries = ['JP', 'AU', 'SK', 'CH']
    instrument_countries = pd.Series([instrument_country.get(instr, 'Other') for instr in df.columns], index=df.columns)
    instruments_to_lag = instrument_countries[~instrument_countries.isin(non_lag_countries)].index.tolist()

    adjusted_df = df.copy()
    if instruments_to_lag:
        # Shift forward by 1 day to lag data
        adjusted_df[instruments_to_lag] = adjusted_df[instruments_to_lag].shift(1)
    adjusted_df = adjusted_df.dropna()
    return adjusted_df

def calculate_daily_changes_in_bps(df):
    """
    Calculate daily changes in yields and convert to bps.
    If yields are in percentage points (e.g. 2.50 = 2.50%),
    a 0.01 change in yield = 1bp, so multiply differences by 100.
    """
    daily_changes = df.diff().dropna() * 100
    return daily_changes

def fallback_mx_ois_data(daily_changes):
    """
    If there is no data for 'MX 2Y Swap OIS', 'MX 5Y Swap OIS', 'MX 10Y Swap OIS',
    fallback to 'MX 2Y Swap', 'MX 5Y Swap', and 'MX 10Y Swap' respectively.
    """
    ois_map = {
        'MX 2Y Swap OIS': 'MX 2Y Swap',
        'MX 5Y Swap OIS': 'MX 5Y Swap',
        'MX 10Y Swap OIS': 'MX 10Y Swap'
    }

    for ois_col, non_ois_col in ois_map.items():
        if ois_col in daily_changes.columns and non_ois_col in daily_changes.columns:
            daily_changes[ois_col] = daily_changes[ois_col].fillna(daily_changes[non_ois_col])

    return daily_changes

def calculate_volatilities(daily_changes, lookback_days):
    """
    Calculate annualized volatility over the specified lookback period.
    Annualize daily volatility: std * sqrt(252).
    """
    if daily_changes.empty:
        return pd.Series(dtype=float)
    recent_returns = daily_changes.tail(lookback_days)
    if recent_returns.empty:
        return pd.Series(dtype=float)
    volatilities = recent_returns.std() * np.sqrt(252)
    return volatilities

def calculate_covariance_matrix(daily_changes, lookback_days):
    """
    Calculate annualized covariance matrix over the specified lookback period.
    Annualize daily covariance: cov * 252.
    """
    if daily_changes.empty:
        return pd.DataFrame()
    recent_returns = daily_changes.tail(lookback_days)
    if recent_returns.empty:
        return pd.DataFrame()
    covariance_matrix = recent_returns.cov() * 252
    return covariance_matrix

def compute_beta(x_returns, y_returns, lookback_days):
    """
    Compute beta of x_returns relative to y_returns using the specified lookback_days window.
    Beta = Cov(x, y) / Var(y).
    """
    if x_returns.empty or y_returns.empty:
        return np.nan
    common_dates = x_returns.index.intersection(y_returns.index)
    if common_dates.empty:
        return np.nan

    x = x_returns.loc[common_dates].tail(lookback_days)
    y = y_returns.loc[common_dates].tail(lookback_days)

    if x.empty or y.empty:
        return np.nan
    if x.std() == 0 or y.std() == 0:
        return np.nan

    cov = np.cov(x, y)[0, 1]
    var_y = np.var(y)
    if var_y == 0:
        return np.nan
    return cov / var_y

def guess_country_from_instrument_name(name):
    """
    Guess the country of an instrument from its name.
    """
    country_codes = {
        'AU': 'AU', 'US': 'US', 'DE': 'DE', 'UK': 'UK', 'IT': 'IT',
        'CA': 'CA', 'JP': 'JP', 'CH': 'CH', 'BR': 'BR', 'MX': 'MX',
        'SA': 'SA', 'CZ': 'CZ', 'PO': 'PO', 'SK': 'SK', 'NZ': 'NZ', 'SW': 'SW', 'NK': 'NK'
    }
    for code in country_codes:
        if code in name:
            return country_codes[code]
    return 'Other'

# Comprehensive instrument-country mapping
instrument_country = {
    "AU 3Y Future": "AU",
    "AU 10Y Future": "AU",
    "US 2Y Future": "US",
    "US 5Y Future": "US",
    "US 10Y Future": "US",
    "US 10Y Ultra Future": "US",
    "US 30Y Future": "US",
    "DE 2Y Future": "DE",
    "DE 5Y Future": "DE",
    "DE 10Y Future": "DE",
    "UK 10Y Future": "UK",
    "IT 10Y Future": "IT",
    "CA 10Y Future": "CA",
    "JP 10Y Future": "JP",
    "CH 1Y Swap": "CH",
    "AU 2Y Swap": "AU",
    "CA 2Y Swap": "CA",
    "US 2Y Swap": "US",
    "DE 2Y Swap": "DE",
    "UK 2Y Swap": "UK",
    "NZ 2Y Swap": "NZ",
    "BR 2Y Swap": "BR",
    "MX 2Y Swap": "MX",
    "MX 2Y Swap OIS": "MX",
    "SA 2Y Swap": "SA",
    "CZ 2Y Swap": "CZ",
    "PO 2Y Swap": "PO",
    "SK 2Y Swap": "SK",
    "CH 2Y Swap": "CH",
    "AU 5Y Swap": "AU",
    "CA 5Y Swap": "CA",
    "US 5Y Swap": "US",
    "DE 5Y Swap": "DE",
    "UK 5Y Swap": "UK",
    "NZ 5Y Swap": "NZ",
    "BR 5Y Swap": "BR",
    "MX 5Y Swap": "MX",
    "MX 5Y Swap OIS": "MX",
    "SA 5Y Swap": "SA",
    "CZ 5Y Swap": "CZ",
    "PO 5Y Swap": "PO",
    "SK 5Y Swap": "SK",
    "CH 5Y Swap": "CH",
    "JP 5Y Swap": "JP",
    "AU 10Y Swap": "AU",
    "CA 10Y Swap": "CA",
    "US 10Y Swap": "US",
    "DE 10Y Swap": "DE",
    "UK 10Y Swap": "UK",
    "NZ 10Y Swap": "NZ",
    "AU 30Y Swap": "AU",
    "CA 30Y Swap": "CA",
    "US 30Y Swap": "US",
    "DE 30Y Swap": "DE",
    "UK 30Y Swap": "UK",
    "NZ 30Y Swap": "NZ",
    "JP 30Y Swap": "JP",
    "MX 10Y Swap": "MX",
    "MX 10Y Swap OIS": "MX",
    "SA 10Y Swap": "SA",
    "CZ 10Y Swap": "CZ",
    "PO 10Y Swap": "PO",
    "SK 10Y Swap": "SK",
    "CH 10Y Swap": "CH",
    "UK 10Y Swap Inf": "UK",
    "SW 10Y Swap": "SW",
    "NK 10Y Swap": "NK"
}

# Utility function: create a waterfall chart.
def create_waterfall_chart(labels, values, total, title, include_diversification=False):
    """
    Build a waterfall chart using plotly.graph_objects.
    If include_diversification is True, then an extra bar labeled 'Diversification Impact'
    is added before the final Total, where:
         diversification = total - sum(values)
    """
    if include_diversification:
        diversification = total - sum(values)
        measures = ["relative"] * len(values) + ["relative", "total"]
        x = labels + ["Diversification Impact", "Total"]
        text = [f"{v:.2f}" for v in values] + [f"{diversification:.2f}", f"{total:.2f}"]
        y = values + [diversification, total]
    else:
        measures = ["relative"] * len(values) + ["total"]
        x = labels + ["Total"]
        text = [f"{v:.2f}" for v in values] + [f"{total:.2f}"]
        y = values + [total]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x,
        text=text,
        y=y,
        connector={"line": {"color": "gray"}}
    ))
    fig.update_layout(
        title=title,
        waterfallgroupgap=0.3
    )
    return fig

def main():
    st.title('📈 BMIX Portfolio Risk Attribution')
    st.write("App initialized successfully.")

    instruments_data = pd.DataFrame({
        "Ticker": [
            "GACGB2 Index","GACGB10 Index","TUAFWD Comdty","FVAFWD Comdty","TYAFWD Comdty","UXYAFWD Comdty",
            "WNAFWD Comdty","DUAFWD Comdty","OEAFWD Comdty","RXAFWD Comdty","GAFWD Comdty","IKAFWD Comdty",
            "CNAFWD Comdty","JBAFWD Comdty","CCSWNI1 Curncy","ADSW2 Curncy","CDSO2 Curncy","USSW2 Curncy",
            "EUSA2 Curncy","BPSWS2 BGN Curncy","NDSWAP2 BGN Curncy","I39302Y Index","MPSW2B BGN Curncy",
            "MPSWF2B Curncy","SAFR1I2 BGN Curncy","CKSW2 BGN Curncy","PZSW2 BGN Curncy","KWSWNI2 BGN Curncy",
            "CCSWNI2 CMPN Curncy","ADSW5 Curncy","CDSO5 Curncy","USSW5 Curncy","EUSA5 Curncy","BPSWS5 BGN Curncy",
            "NDSWAP5 BGN Curncy","I39305Y Index","MPSW5E Curncy","MPSWF5E Curncy","SASW5 Curncy","CKSW5 Curncy",
            "PZSW5 Curncy","KWSWNI5 Curncy","CCSWNI5 Curncy","JYSO5 Curncy","ADSW10 Curncy","CDSW10 Curncy",
            "USSW10 Curncy","EUSA10 Curncy","BPSWS10 BGN Curncy","NDSWAP10 BGN Curncy","ADSW30 Curncy",
            "CDSW30 Curncy","USSW30 Curncy","EUSA30 Curncy","BPSWS30 BGN Curncy","NDSWAP30 BGN Curncy",
            "JYSO30 Curncy","MPSW10J BGN Curncy","MPSWF10J BGN Curncy","SASW10 Curncy","CKSW10 BGN Curncy",
            "PZSW10 BGN Curncy","KWSWNI10 BGN Curncy","CCSWNI10 Curncy","BPSWIT10 Curncy", "SKSW10 Curncy", "NKSW10 Curncy"
        ],
        "Instrument Name": [
            "AU 3Y Future","AU 10Y Future","US 2Y Future","US 5Y Future","US 10Y Future","US 10Y Ultra Future",
            "US 30Y Future","DE 2Y Future","DE 5Y Future","DE 10Y Future","UK 10Y Future","IT 10Y Future",
            "CA 10Y Future","JP 10Y Future","CH 1Y Swap","AU 2Y Swap","CA 2Y Swap","US 2Y Swap","DE 2Y Swap",
            "UK 2Y Swap","NZ 2Y Swap","BR 2Y Swap","MX 2Y Swap","MX 2Y Swap OIS","SA 2Y Swap","CZ 2Y Swap",
            "PO 2Y Swap","SK 2Y Swap","CH 2Y Swap","AU 5Y Swap","CA 5Y Swap","US 5Y Swap","DE 5Y Swap","UK 5Y Swap",
            "NZ 5Y Swap","BR 5Y Swap","MX 5Y Swap","MX 5Y Swap OIS","SA 5Y Swap","CZ 5Y Swap","PO 5Y Swap",
            "SK 5Y Swap","CH 5Y Swap","JP 5Y Swap","AU 10Y Swap","CA 10Y Swap","US 10Y Swap","DE 10Y Swap",
            "UK 10Y Swap","NZ 10Y Swap","AU 30Y Swap","CA 30Y Swap","US 30Y Swap","DE 30Y Swap","UK 30Y Swap",
            "NZ 30Y Swap","JP 30Y Swap","MX 10Y Swap","MX 10Y Swap OIS","SA 10Y Swap","CZ 10Y Swap","PO 10Y Swap",
            "SK 10Y Swap","CH 10Y Swap","UK 10Y Swap Inf", "SW 10Y Swap", "NK 10Y Swap"
        ],
        "Portfolio": [
            "DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","EM","DM","DM","DM",
            "DM","DM","DM","EM","EM","EM","EM","EM","EM","EM","EM","DM","DM","DM","DM","DM","DM","EM",
            "EM","EM","EM","EM","EM","EM","EM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM",
            "DM","DM","DM","EM","EM","EM","EM","EM","EM","EM","DM", "DM", "DM"
        ]
    })

    dm_instruments = instruments_data[instruments_data['Portfolio'] == 'DM']['Instrument Name'].tolist()
    em_instruments = instruments_data[instruments_data['Portfolio'] == 'EM']['Instrument Name'].tolist()

    default_positions_dm = pd.DataFrame({
        'Instrument': dm_instruments,
        'Outright': [0.0]*len(dm_instruments),
        'Curve': [0.0]*len(dm_instruments),
        'Spread': [0.0]*len(dm_instruments),
    })

    default_positions_em = pd.DataFrame({
        'Instrument': em_instruments,
        'Outright': [0.0]*len(em_instruments),
        'Curve': [0.0]*len(em_instruments),
        'Spread': [0.0]*len(em_instruments),
    })

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
    default_index = 0
    if 'US 10Y Future' in available_columns:
        default_index = available_columns.index('US 10Y Future')
    sensitivity_rate = st.sidebar.selectbox(
        'Select sensitivity instrument:',
        options=available_columns,
        index=default_index
    )

    tabs = st.tabs(["📊 Risk Attribution", "📂 Input Positions", "⚙️ Settings"])

    with tabs[1]:
        st.header("🔄 Input Positions")

        st.subheader('📈 DM Portfolio Positions')
        gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
        gb_dm.configure_default_column(editable=True, resizable=True)
        dm_options = gb_dm.build()
        dm_response = AgGrid(
            default_positions_dm,
            gridOptions=dm_options,
            height=600,
            width='100%',
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True
        )
        positions_data_dm = dm_response['data']

        if len(em_instruments) > 0:
            st.subheader('🌍 EM Portfolio Positions')
            gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
            gb_em.configure_default_column(editable=True, resizable=True)
            em_options = gb_em.build()
            em_response = AgGrid(
                default_positions_em,
                gridOptions=em_options,
                height=600,
                width='100%',
                enable_enterprise_modules=False,
                fit_columns_on_grid_load=True
            )
            positions_data_em = em_response['data']
        else:
            positions_data_em = pd.DataFrame(columns=['Instrument', 'Outright', 'Curve', 'Spread'])

    with tabs[2]:
        st.header("⚙️ Configuration Settings")
        volatility_period_options = {
            '📅 1 month (~21 days)': 21,
            '📆 3 months (~63 days)': 63,
            '📅 6 months (~126 days)': 126,
            '🗓️ 1 year (252 days)': 252,
            '📅 3 years (756 days)': 756,
            '📆 5 years (1260 days)': 1260
        }
        volatility_period = st.selectbox('Volatility lookback:', list(volatility_period_options.keys()), index=3)
        volatility_lookback_days = volatility_period_options[volatility_period]

        var_period_options = {
            '🗓️ 1 year (252 days)': 252,
            '📆 5 years (~1260 days)': 1260
        }
        var_period = st.selectbox('VaR lookback:', list(var_period_options.keys()), index=1)
        var_lookback_days = var_period_options[var_period]

    with tabs[0]:
        st.header("📊 Risk Attribution Results")
        if st.button('🚀 Run Risk Attribution'):
            with st.spinner('Calculating risk attribution...'):
                df = load_historical_data(excel_file)
                if df.empty:
                    st.error("No data loaded. Check Excel file.")
                    st.stop()

                df = adjust_time_zones(df, instrument_country)
                daily_changes = calculate_daily_changes_in_bps(df)
                daily_changes = fallback_mx_ois_data(daily_changes)

                if daily_changes.empty:
                    st.warning("No daily changes computed.")
                    st.stop()

                volatilities = calculate_volatilities(daily_changes, volatility_lookback_days)
                covariance_matrix = calculate_covariance_matrix(daily_changes, volatility_lookback_days)
                beta_data_window = daily_changes.tail(volatility_lookback_days)

                positions_data_dm = pd.DataFrame(positions_data_dm).astype({'Outright': float, 'Curve': float, 'Spread': float})
                if not positions_data_em.empty:
                    positions_data_em = pd.DataFrame(positions_data_em).astype({'Outright': float, 'Curve': float, 'Spread': float})
                else:
                    positions_data_em = pd.DataFrame(columns=['Instrument', 'Outright', 'Curve', 'Spread'])

                positions_data_dm['Portfolio'] = 'DM'
                if not positions_data_em.empty:
                    positions_data_em['Portfolio'] = 'EM'
                positions_data = pd.concat([positions_data_dm, positions_data_em], ignore_index=True)

                positions_list = []
                for _, row in positions_data.iterrows():
                    instrument = row['Instrument']
                    portfolio = row['Portfolio']
                    for pos_type in ['Outright', 'Curve', 'Spread']:
                        pos_val = row[pos_type]
                        if pos_val != 0:
                            positions_list.append({
                                'Instrument': instrument,
                                'Position Type': pos_type,
                                'Position': pos_val,
                                'Portfolio': portfolio
                            })

                expanded_positions_data = pd.DataFrame(positions_list)
                if expanded_positions_data.empty:
                    st.warning("No active positions entered.")
                    st.stop()

                expanded_positions_vector = expanded_positions_data.set_index(['Instrument', 'Position Type'])['Position']
                if covariance_matrix.empty:
                    st.warning("Covariance matrix empty.")
                    st.stop()

                instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()
                missing_instruments = [instr for instr in instruments if instr not in covariance_matrix.index]
                if missing_instruments:
                    drop_labels = [(instr, pt) for instr in missing_instruments for pt in ['Outright', 'Curve', 'Spread']]
                    expanded_positions_vector = expanded_positions_vector.drop(labels=drop_labels, errors='ignore')
                    if expanded_positions_vector.empty:
                        st.warning("All instruments missing from covariance.")
                        st.stop()
                    instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()

                valid_instruments = [instr for instr in instruments if instr in covariance_matrix.index]
                if not valid_instruments:
                    st.warning("No valid instruments after filtering.")
                    st.stop()

                # Build the base submatrix for valid instruments
                covariance_submatrix = covariance_matrix.loc[valid_instruments, valid_instruments]

                # Build the expanded covariance matrix for each (Instrument, Position Type)
                expanded_cov_matrix = pd.DataFrame(
                    data=0.0,
                    index=expanded_positions_vector.index,
                    columns=expanded_positions_vector.index
                )
                for pos1 in expanded_positions_vector.index:
                    instr1 = pos1[0]  # Instrument part of (Instrument, Position Type)
                    for pos2 in expanded_positions_vector.index:
                        instr2 = pos2[0]
                        val = 0.0
                        if (instr1 in covariance_submatrix.index) and (instr2 in covariance_submatrix.columns):
                            val = covariance_submatrix.loc[instr1, instr2]
                        expanded_cov_matrix.loc[pos1, pos2] = val

                expanded_cov_matrix = expanded_cov_matrix.astype(float)

                portfolio_variance = np.dot(expanded_positions_vector.values,
                                            np.dot(expanded_cov_matrix.values, expanded_positions_vector.values))
                if np.isnan(portfolio_variance) or portfolio_variance <= 0:
                    st.warning("Portfolio variance invalid.")
                    st.stop()
                portfolio_volatility = np.sqrt(portfolio_variance)
                if np.isnan(portfolio_volatility):
                    st.warning("Volatility is NaN.")
                    st.stop()

                expanded_volatilities = expanded_positions_vector.index.get_level_values('Instrument').map(volatilities).to_series(index=expanded_positions_vector.index)
                standalone_volatilities = expanded_positions_vector.abs() * expanded_volatilities
                marginal_contributions = expanded_cov_matrix.dot(expanded_positions_vector)
                contribution_to_variance = expanded_positions_vector * marginal_contributions
                # Euler decomposition for volatility:
                contribution_to_volatility = contribution_to_variance / portfolio_volatility
                percent_contribution = (contribution_to_variance / portfolio_variance) * 100

                risk_contributions = expanded_positions_data.copy()
                risk_contributions = risk_contributions.set_index(['Instrument', 'Position Type']).reindex(expanded_positions_vector.index).reset_index()
                risk_contributions['Position Stand-alone Volatility'] = standalone_volatilities.values
                risk_contributions['Contribution to Volatility (bps)'] = contribution_to_volatility.values
                risk_contributions['Percent Contribution (%)'] = percent_contribution.values
                risk_contributions['Instrument Volatility per 1Y Duration (bps)'] = expanded_volatilities.values

                risk_contributions_formatted = risk_contributions[
                    ['Instrument', 'Position Type', 'Position', 'Position Stand-alone Volatility',
                     'Instrument Volatility per 1Y Duration (bps)',
                     'Contribution to Volatility (bps)', 'Percent Contribution (%)', 'Portfolio']
                ]
                risk_contributions_formatted = risk_contributions_formatted[
                    (risk_contributions_formatted['Position'].notna()) & (risk_contributions_formatted['Position'] != 0)
                ]
                numeric_cols = [
                    'Position', 'Position Stand-alone Volatility',
                    'Instrument Volatility per 1Y Duration (bps)',
                    'Contribution to Volatility (bps)', 'Percent Contribution (%)'
                ]
                risk_contributions_formatted[numeric_cols] = risk_contributions_formatted[numeric_cols].round(2)

                def fmt_val(x):
                    return f"{x:.2f} bps" if (not np.isnan(x) and not np.isinf(x)) else "N/A"

                # VaR/cVaR calculations using portfolio returns
                VaR_95, VaR_99, cVaR_95, cVaR_99 = (np.nan, np.nan, np.nan, np.nan)
                price_returns_var = daily_changes.tail(var_lookback_days)
                positions_per_instrument = expanded_positions_vector.groupby('Instrument').sum()
                if not price_returns_var.empty:
                    available_instruments_var = positions_per_instrument.index.intersection(price_returns_var.columns)
                    if not available_instruments_var.empty:
                        positions_for_var = positions_per_instrument.loc[available_instruments_var]
                        price_returns_var = price_returns_var[available_instruments_var]
                        if not price_returns_var.empty:
                            portfolio_returns_var = price_returns_var.dot(positions_for_var)
                            if not portfolio_returns_var.empty:
                                # Compute portfolio VaR and cVaR (losses are positive)
                                VaR_95 = -np.percentile(portfolio_returns_var, 5)
                                VaR_99 = -np.percentile(portfolio_returns_var, 1)
                                cVaR_95 = -portfolio_returns_var[portfolio_returns_var <= -VaR_95].mean() if (portfolio_returns_var <= -VaR_95).any() else np.nan
                                cVaR_99 = -portfolio_returns_var[portfolio_returns_var <= -VaR_99].mean() if (portfolio_returns_var <= -VaR_99).any() else np.nan

                # --- Compute instrument contributions to portfolio cVaR ---
                # Instead of computing standalone instrument cVaRs, we now look at the days when the portfolio reaches its cVaR level.
                # For each instrument, we compute the loss series as: loss = - (position × instrument return)
                # and then average that loss over the extreme days.
                if not portfolio_returns_var.empty:
                    extreme_mask_95 = portfolio_returns_var <= -VaR_95
                    extreme_mask_99 = portfolio_returns_var <= -VaR_99
                    instrument_contrib_95 = {}
                    instrument_contrib_99 = {}
                    for instr in positions_for_var.index:
                        if instr in price_returns_var.columns:
                            pos = positions_for_var[instr]
                            loss_series = - price_returns_var[instr] * pos  # loss series for instrument
                            contrib_95 = loss_series[extreme_mask_95].mean() if extreme_mask_95.any() else np.nan
                            contrib_99 = loss_series[extreme_mask_99].mean() if extreme_mask_99.any() else np.nan
                            instrument_contrib_95[instr] = contrib_95
                            instrument_contrib_99[instr] = contrib_99

                # Sort instrument contributions for waterfall charts
                cvar95_items = sorted(instrument_contrib_95.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
                cvar95_labels = [item[0] for item in cvar95_items]
                cvar95_values = [item[1] for item in cvar95_items]
                # Check if there is any reconciliation gap due to rounding; ideally sum should equal portfolio cVaR.
                diff_95 = cVaR_95 - sum([v for v in cvar95_values if v is not None])
                use_diversification_95 = abs(diff_95) > 1e-6

                cvar99_items = sorted(instrument_contrib_99.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
                cvar99_labels = [item[0] for item in cvar99_items]
                cvar99_values = [item[1] for item in cvar99_items]
                diff_99 = cVaR_99 - sum([v for v in cvar99_values if v is not None])
                use_diversification_99 = abs(diff_99) > 1e-6

                # Compute portfolio beta using beta_data_window
                portfolio_beta = np.nan
                portfolio_r2 = np.nan
                instrument_betas = {}
                if (sensitivity_rate in beta_data_window.columns) and (not beta_data_window.empty) and (not positions_for_var.empty):
                    available_for_beta = positions_for_var.index.intersection(beta_data_window.columns)
                    if not available_for_beta.empty:
                        us10yr_returns = beta_data_window[sensitivity_rate]
                        portfolio_returns_for_beta = beta_data_window[available_for_beta].dot(positions_for_var.loc[available_for_beta])
                        portfolio_beta = compute_beta(portfolio_returns_for_beta, us10yr_returns, volatility_lookback_days)

                        common_dates = portfolio_returns_for_beta.index.intersection(us10yr_returns.index)
                        if len(common_dates) > 1:
                            pvals = portfolio_returns_for_beta.loc[common_dates]
                            uvals = us10yr_returns.loc[common_dates]
                            corr = np.corrcoef(pvals, uvals)[0, 1]
                            portfolio_r2 = corr**2

                        for instr in available_for_beta:
                            pos_val = positions_for_var[instr]
                            if pos_val != 0:
                                instr_return = beta_data_window[instr]
                                instr_beta = compute_beta(instr_return, us10yr_returns, volatility_lookback_days)
                                if not np.isnan(instr_beta):
                                    instrument_betas[instr] = (pos_val, instr_beta)

                risk_contributions_formatted['Country'] = risk_contributions_formatted['Instrument'].apply(guess_country_from_instrument_name)
                country_bucket = risk_contributions_formatted.groupby(['Country', 'Position Type']).agg({
                    'Contribution to Volatility (bps)': 'sum'
                }).reset_index()

                # ---------------------------
                # Create Waterfall Charts
                # ---------------------------
                # Volatility Waterfall: by instrument
                vol_inst = risk_contributions_formatted.groupby("Instrument")["Contribution to Volatility (bps)"].sum().reset_index()
                vol_inst['abs'] = vol_inst["Contribution to Volatility (bps)"].abs()
                vol_inst = vol_inst.sort_values(by="abs", ascending=False)
                vol_inst_labels = vol_inst["Instrument"].tolist()
                vol_inst_values = vol_inst["Contribution to Volatility (bps)"].tolist()
                fig_vol_inst = create_waterfall_chart(vol_inst_labels, vol_inst_values, portfolio_volatility,
                                                      "Volatility Contributions by Instrument", include_diversification=False)

                # Volatility Waterfall: by Country & Bucket
                if not country_bucket.empty:
                    country_bucket["Group"] = country_bucket["Country"] + " - " + country_bucket["Position Type"]
                    country_bucket['abs'] = country_bucket["Contribution to Volatility (bps)"].abs()
                    country_bucket = country_bucket.sort_values(by="abs", ascending=False)
                    vol_group_labels = country_bucket["Group"].tolist()
                    vol_group_values = country_bucket["Contribution to Volatility (bps)"].tolist()
                    fig_vol_group = create_waterfall_chart(vol_group_labels, vol_group_values, portfolio_volatility,
                                                           "Volatility Contributions by Country & Bucket", include_diversification=False)
                else:
                    fig_vol_group = None

                # cVaR Waterfall: by instrument (95% & 99%)
                fig_cvar95 = create_waterfall_chart(cvar95_labels, cvar95_values, cVaR_95,
                                                    "cVaR (95%) Contributions by Instrument", include_diversification=use_diversification_95)
                fig_cvar99 = create_waterfall_chart(cvar99_labels, cvar99_values, cVaR_99,
                                                    "cVaR (99%) Contributions by Instrument", include_diversification=use_diversification_99)

                # ---------------------------
                # Display the charts and metrics
                # ---------------------------
                st.subheader("Risk Attribution by Instrument (Volatility)")
                st.plotly_chart(fig_vol_inst, use_container_width=True)

                st.subheader("Risk Attribution by Country & Bucket (Volatility)")
                if fig_vol_group:
                    st.plotly_chart(fig_vol_group, use_container_width=True)
                    st.write("Aggregated Risk Contributions by Country and Bucket:")
                    st.dataframe(country_bucket)
                else:
                    st.write("No country/bucket data to display.")

                st.subheader("cVaR (95%) Contributions by Instrument")
                st.plotly_chart(fig_cvar95, use_container_width=True)

                st.subheader("cVaR (99%) Contributions by Instrument")
                st.plotly_chart(fig_cvar99, use_container_width=True)

                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                metrics_col1.metric(label="📊 Total Portfolio Volatility", value=fmt_val(portfolio_volatility))
                metrics_col2.metric(label="📉 Daily VaR (95%)", value=fmt_val(VaR_95))
                metrics_col3.metric(label="📉 Daily VaR (99%)", value=fmt_val(VaR_99))
                metrics_col4.metric(label="📈 Daily cVaR (95%)", value=fmt_val(cVaR_95))

                st.subheader('📈 Value at Risk (VaR) and Conditional VaR (cVaR)')
                st.write(f"**Daily VaR at 95%:** {fmt_val(VaR_95)}")
                st.write(f"**Daily cVaR at 95%:** {fmt_val(cVaR_95)}")
                st.write(f"**Daily VaR at 99%:** {fmt_val(VaR_99)}")
                st.write(f"**Daily cVaR at 99%:** {fmt_val(cVaR_99)}")

                st.subheader("📉 Beta to US 10yr Rates (Daily Basis)")
                if not np.isnan(portfolio_beta):
                    st.write(f"**Portfolio Beta to {sensitivity_rate} (Daily):** {portfolio_beta:.4f}")
                    st.write(f"**Portfolio R² with {sensitivity_rate} (Daily):** {portfolio_r2:.4f}")
                    if instrument_betas:
                        st.write("**Instrument Betas to US 10yr Rates (including Position, Daily):**")
                        beta_data = []
                        for instr, (pos_val, b) in instrument_betas.items():
                            beta_data.append({'Instrument': instr, 'Position': pos_val, 'Beta': b})
                        beta_df = pd.DataFrame(beta_data)
                        beta_df['Position'] = beta_df['Position'].round(2)
                        beta_df['Beta'] = beta_df['Beta'].round(4)
                        st.dataframe(beta_df)
                        st.markdown("*Footnote:* If the US 10-year yield moves by 1bp in a day, the portfolio changes by approximately Beta × 1bp in that day.")
                    else:
                        st.write("No individual instrument betas to display.")
                else:
                    st.write("No portfolio beta computed. Check data and positions.")

                if not risk_contributions_formatted.empty:
                    st.subheader('📄 Detailed Risk Contributions by Instrument')
                    gb_risk = GridOptionsBuilder.from_dataframe(risk_contributions_formatted)
                    gb_risk.configure_default_column(editable=False, resizable=True)
                    risk_grid_options = gb_risk.build()

                    AgGrid(
                        risk_contributions_formatted,
                        gridOptions=risk_grid_options,
                        height=400,
                        width='100%',
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
                else:
                    st.write("No detailed contributions table to display.")

if __name__ == '__main__':
    main()

