import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="📈 Fixed Income Portfolio Risk Attribution", layout="wide")

@st.cache_data(show_spinner=False)
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

# Since we now have direct yield series for Aussie yields, no adjustments needed
@st.cache_data(show_spinner=False)
def adjust_time_zones(df, instrument_country):
    # Certain countries are considered "non-lag"
    # Non-lag countries: JP, AU, SK, CH
    # All other instruments are shifted by one business day backward.
    non_lag_countries = ['JP', 'AU', 'SK', 'CH']
    instrument_countries = pd.Series([instrument_country.get(instr, 'Other') for instr in df.columns],
                                     index=df.columns)
    instruments_to_lag = instrument_countries[~instrument_countries.isin(non_lag_countries)].index.tolist()

    adjusted_df = df.copy()
    if instruments_to_lag:
        adjusted_df[instruments_to_lag] = adjusted_df[instruments_to_lag].shift(-1)
    adjusted_df = adjusted_df.dropna()
    return adjusted_df

@st.cache_data(show_spinner=False)
def to_weekly_data(df):
    # Resample to weekly frequency using Friday closes
    # If Friday data is missing, picks the last available day that week
    weekly_df = df.resample('W-FRI').last().dropna(how='all')
    return weekly_df

@st.cache_data(show_spinner=False)
def calculate_weekly_changes_in_bps(weekly_df):
    # Assuming yields are already in percentage points (0.01 = 1%)
    # A change of 0.01 = 1bp, so multiply changes by 100 to convert to bps
    weekly_changes = weekly_df.diff().dropna() * 100
    return weekly_changes

@st.cache_data(show_spinner=False)
def calculate_volatilities(weekly_changes, lookback_weeks):
    if weekly_changes.empty:
        return pd.Series(dtype=float)
    recent_returns = weekly_changes.tail(lookback_weeks)
    if recent_returns.empty:
        return pd.Series(dtype=float)
    # Annualize volatility from weekly returns: std * sqrt(52)
    volatilities = recent_returns.std() * np.sqrt(52)
    return volatilities

@st.cache_data(show_spinner=False)
def calculate_covariance_matrix(weekly_changes, lookback_weeks):
    if weekly_changes.empty:
        return pd.DataFrame()
    recent_returns = weekly_changes.tail(lookback_weeks)
    if recent_returns.empty:
        return pd.DataFrame()
    # Annualize covariance from weekly returns: cov * 52
    covariance_matrix = recent_returns.cov() * 52
    return covariance_matrix

def compute_beta(x_returns, y_returns):
    if x_returns.empty or y_returns.empty:
        return np.nan
    common_dates = x_returns.index.intersection(y_returns.index)
    if common_dates.empty:
        return np.nan
    x = x_returns.loc[common_dates]
    y = y_returns.loc[common_dates]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    cov = np.cov(x, y)[0, 1]
    var_y = np.var(y)
    if var_y == 0:
        return np.nan
    return cov / var_y

def guess_country_from_instrument_name(name):
    country_codes = {
        'AU': 'AU', 'US': 'US', 'DE': 'DE', 'UK': 'UK', 'IT': 'IT',
        'CA': 'CA', 'JP': 'JP', 'CH': 'CH', 'BR': 'BR', 'MX': 'MX',
        'SA': 'SA', 'CZ': 'CZ', 'PO': 'PO', 'SK': 'SK', 'NZ': 'NZ'
    }
    for code in country_codes:
        if code in name:
            return country_codes[code]
    return 'Other'

instrument_country = {
    'GACGB2 Index': 'AU',
    'GACGB10 Index': 'AU',
    # Add other instruments and mappings as needed
    'US 2Y Future': 'US',
    'US 5Y Future': 'US',
    'US 10Y Future': 'US',
}

def main():
    st.title('📈 Fixed Income Portfolio Risk Attribution')
    st.write("App initialized successfully.")

    # Full arrays with all instruments (example shown, adjust as needed)
    instruments_data = pd.DataFrame({
        "Ticker": [
            "GACGB2 Index","GACGB10 Index","TUAFWD Comdty","FVAFWD Comdty","TYAFWD Comdty","UXYAFWD Comdty",
            "WNAFWD Comdty","DUAFWD Comdty","OEAFWD Comdty","RXAFWD Comdty"
        ],
        "Instrument Name": [
            "AU 3Y Future","AU 10Y Future","US 2Y Future","US 5Y Future","US 10Y Future",
            "US 10Y Ultra Future","US 30Y Future","DE 2Y Future","DE 5Y Future","DE 10Y Future"
        ],
        "Portfolio": [
            "DM","DM","DM","DM","DM","DM","DM","DM","DM","DM"
        ]
    })

    # Update as needed: removing old references and ensuring these map correctly
    # For illustration, we keep the same naming but now assume 'AU 3Y Future' and 'AU 10Y Future'
    # refer to GACGB2 Index and GACGB10 Index yields directly.
    dm_instruments = instruments_data[instruments_data['Portfolio'] == 'DM']['Instrument Name'].tolist()
    em_instruments = []  # Update as needed if you have EM instruments

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

        # DM table
        st.subheader('📈 DM Portfolio Positions')
        gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
        gb_dm.configure_default_column(editable=True, resizable=True)
        gb_dm.configure_column('Instrument', editable=False, width=600)
        dm_options = gb_dm.build()
        dm_response = AgGrid(
            default_positions_dm,
            gridOptions=dm_options,
            height=600,
            width='100%',
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=False
        )
        positions_data_dm = dm_response['data']

        # EM table (if any)
        if len(em_instruments) > 0:
            st.subheader('🌍 EM Portfolio Positions')
            gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
            gb_em.configure_default_column(editable=True, resizable=True)
            gb_em.configure_column('Instrument', editable=False, width=600)
            em_options = gb_em.build()
            em_response = AgGrid(
                default_positions_em,
                gridOptions=em_options,
                height=600,
                width='100%',
                enable_enterprise_modules=False,
                fit_columns_on_grid_load=False
            )
            positions_data_em = em_response['data']
        else:
            positions_data_em = pd.DataFrame(columns=['Instrument', 'Outright', 'Curve', 'Spread'])

    with tabs[2]:
        st.header("⚙️ Configuration Settings")
        # Weekly data: 1 year (52 weeks)
        volatility_period_options = {
            '📅 1 month (approx 4-5 weeks)': 5,
            '📆 3 months (~13 weeks)': 13,
            '📅 6 months (~26 weeks)': 26,
            '🗓️ 1 year (52 weeks)': 52,
            '📅 3 years (156 weeks)': 156,
            '📆 5 years (260 weeks)': 260,
            '📅 10 years (520 weeks)': 520
        }
        volatility_period = st.selectbox('Volatility lookback:', list(volatility_period_options.keys()), index=3)
        volatility_lookback_weeks = volatility_period_options[volatility_period]

        var_period_options = {
            '📆 5 years (~260 weeks)': 260,
            '📆 10 years (~520 weeks)': 520,
            '📆 15 years (~780 weeks)': 780
        }
        var_period = st.selectbox('VaR lookback:', list(var_period_options.keys()), index=1)
        var_lookback_weeks = var_period_options[var_period]

    with tabs[0]:
        st.header("📊 Risk Attribution Results")

        if st.button('🚀 Run Risk Attribution'):
            with st.spinner('Calculating risk attribution...'):
                df = load_historical_data(excel_file)
                if df.empty:
                    st.error("No data loaded. Check Excel file.")
                    st.stop()

                df = adjust_time_zones(df, instrument_country)

                # Convert daily data to weekly closes
                weekly_df = to_weekly_data(df)
                if weekly_df.empty:
                    st.warning("No weekly data available.")
                    st.stop()

                # Calculate weekly changes in bps
                weekly_changes = calculate_weekly_changes_in_bps(weekly_df)
                if weekly_changes.empty:
                    st.warning("No weekly changes computed.")
                    st.stop()

                volatilities = calculate_volatilities(weekly_changes, volatility_lookback_weeks)
                covariance_matrix = calculate_covariance_matrix(weekly_changes, var_lookback_weeks)

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

                covariance_submatrix = covariance_matrix.loc[valid_instruments, valid_instruments]
                expanded_cov_matrix = pd.DataFrame(index=expanded_positions_vector.index, columns=expanded_positions_vector.index)
                for pos1 in expanded_positions_vector.index:
                    instr1 = pos1[0]
                    if instr1 in covariance_submatrix.index:
                        expanded_cov_matrix.loc[pos1, :] = covariance_submatrix.loc[instr1, valid_instruments].values
                    else:
                        expanded_cov_matrix.loc[pos1, :] = 0.0
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
                contribution_to_volatility = contribution_to_variance / portfolio_volatility
                percent_contribution = (contribution_to_volatility / portfolio_volatility) * 100

                risk_contributions = expanded_positions_data.copy()
                risk_contributions = risk_contributions.set_index(['Instrument', 'Position Type']).reindex(expanded_positions_vector.index).reset_index()
                risk_contributions['Position Stand-alone Volatility'] = standalone_volatilities.values
                risk_contributions['Contribution to Volatility (bps)'] = contribution_to_volatility.values
                risk_contributions['Percent Contribution (%)'] = percent_contribution.values
                risk_contributions['Instrument Volatility per 1Y Duration (bps)'] = expanded_volatilities.values

                risk_contributions_formatted = risk_contributions[
                    ['Instrument', 'Position Type', 'Position', 'Position Stand-alone Volatility',
                     'Instrument Volatility per 1Y Duration (bps)', 'Contribution to Volatility (bps)', 'Percent Contribution (%)', 'Portfolio']
                ]
                risk_contributions_formatted = risk_contributions_formatted[
                    (risk_contributions_formatted['Position'].notna()) & (risk_contributions_formatted['Position'] != 0)
                ]
                numeric_cols = ['Position', 'Position Stand-alone Volatility', 'Instrument Volatility per 1Y Duration (bps)', 'Contribution to Volatility (bps)', 'Percent Contribution (%)']
                risk_contributions_formatted[numeric_cols] = risk_contributions_formatted[numeric_cols].round(2)

                def fmt_val(x):
                    return f"{x:.2f} bps" if (not np.isnan(x) and not np.isinf(x)) else "N/A"

                # VaR/cVaR calculations on weekly returns
                VaR_95, VaR_99, cVaR_95, cVaR_99 = (np.nan, np.nan, np.nan, np.nan)
                price_returns_var = weekly_changes.tail(var_lookback_weeks)
                positions_per_instrument = expanded_positions_vector.groupby('Instrument').sum()
                if not price_returns_var.empty:
                    available_instruments_var = positions_per_instrument.index.intersection(price_returns_var.columns)
                    if not available_instruments_var.empty:
                        positions_for_var = positions_per_instrument.loc[available_instruments_var]
                        price_returns_var = price_returns_var[available_instruments_var]
                        if not price_returns_var.empty:
                            portfolio_returns_var = price_returns_var.dot(positions_for_var)
                            if not portfolio_returns_var.empty:
                                VaR_95 = -np.percentile(portfolio_returns_var, 5)
                                VaR_99 = -np.percentile(portfolio_returns_var, 1)
                                cVaR_95 = -portfolio_returns_var[portfolio_returns_var <= -VaR_95].mean() if (portfolio_returns_var <= -VaR_95).any() else np.nan
                                cVaR_99 = -portfolio_returns_var[portfolio_returns_var <= -VaR_99].mean() if (portfolio_returns_var <= -VaR_99).any() else np.nan

                # Compute Beta with weekly returns
                portfolio_beta = np.nan
                instrument_betas = {}
                if (sensitivity_rate in weekly_changes.columns) and (not weekly_changes.empty) and (not positions_per_instrument.empty):
                    us10yr_returns = weekly_changes[sensitivity_rate]  # weekly returns in bps
                    portfolio_returns_for_beta = weekly_changes[positions_per_instrument.index].dot(positions_per_instrument)
                    portfolio_beta = compute_beta(portfolio_returns_for_beta, us10yr_returns)

                    for instr in positions_per_instrument.index:
                        pos_val = positions_per_instrument[instr]
                        if (pos_val != 0) and (instr in weekly_changes.columns):
                            instr_return = weekly_changes[instr] * pos_val
                            instr_beta = compute_beta(instr_return, us10yr_returns)
                            if not np.isnan(instr_beta):
                                instrument_betas[instr] = (pos_val, instr_beta)

                risk_contributions_formatted['Country'] = risk_contributions_formatted['Instrument'].apply(guess_country_from_instrument_name)
                country_bucket = risk_contributions_formatted.groupby(['Country', 'Position Type']).agg({
                    'Contribution to Volatility (bps)': 'sum'
                }).reset_index()

                st.subheader("Risk Attribution by Instrument")
                if not risk_contributions_formatted.empty:
                    fig_instrument_pie = px.pie(
                        risk_contributions_formatted,
                        names='Instrument',
                        values='Percent Contribution (%)',
                        title='Risk Contributions by Instrument',
                        hole=0.4
                    )
                    fig_instrument_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_instrument_pie, use_container_width=True)
                else:
                    st.warning("No risk contributions to display.")

                st.subheader("Risk Attribution by Country and Bucket (Outright, Curve, Spread)")
                if not country_bucket.empty:
                    fig_country_bucket = px.bar(
                        country_bucket,
                        x='Country',
                        y='Contribution to Volatility (bps)',
                        color='Position Type',
                        title='Risk Contributions by Country and Bucket',
                        barmode='stack'
                    )
                    st.plotly_chart(fig_country_bucket, use_container_width=True)
                    st.write("Aggregated Risk Contributions by Country and Bucket:")
                    st.dataframe(country_bucket)
                else:
                    st.write("No country/bucket data to display.")

                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                metrics_col1.metric(label="📊 Total Portfolio Volatility", value=fmt_val(portfolio_volatility))
                metrics_col2.metric(label="📉 Weekly VaR (95%)", value=fmt_val(VaR_95))
                metrics_col3.metric(label="📉 Weekly VaR (99%)", value=fmt_val(VaR_99))
                metrics_col4.metric(label="📈 Weekly cVaR (95%)", value=fmt_val(cVaR_95))

                st.subheader('📈 Value at Risk (VaR) and Conditional VaR (cVaR)')
                st.write(f"**Weekly VaR at 95%:** {fmt_val(VaR_95)}")
                st.write(f"**Weekly cVaR at 95%:** {fmt_val(cVaR_95)}")
                st.write(f"**Weekly VaR at 99%:** {fmt_val(VaR_99)}")
                st.write(f"**Weekly cVaR at 99%:** {fmt_val(cVaR_99)}")

                st.subheader("📉 Beta to US 10yr Rates (Weekly Basis)")
                if not np.isnan(portfolio_beta):
                    st.write(f"**Portfolio Beta to {sensitivity_rate} (Weekly):** {portfolio_beta:.4f}")
                    if instrument_betas:
                        st.write("**Instrument Betas to US 10yr Rates (including Position, Weekly):**")
                        beta_data = []
                        for instr, (pos_val, b) in instrument_betas.items():
                            beta_data.append({'Instrument': instr, 'Position': pos_val, 'Beta': b})
                        beta_df = pd.DataFrame(beta_data)
                        beta_df['Position'] = beta_df['Position'].round(2)
                        beta_df['Beta'] = beta_df['Beta'].round(4)
                        st.dataframe(beta_df)

                        st.markdown("*Footnote:* If the US 10-year yield moves by 1bp in a week, the portfolio changes approximately Beta × 1bp in that week.")
                    else:
                        st.write("No individual instrument betas to display.")
                else:
                    st.write("No portfolio beta computed. Check data and positions.")

                if not risk_contributions_formatted.empty:
                    st.subheader('📄 Detailed Risk Contributions by Instrument')
                    gb_risk = GridOptionsBuilder.from_dataframe(risk_contributions_formatted)
                    gb_risk.configure_default_column(editable=False, resizable=True)
                    gb_risk.configure_column('Instrument', width=300)
                    risk_grid_options = gb_risk.build()

                    AgGrid(
                        risk_contributions_formatted,
                        gridOptions=risk_grid_options,
                        height=400,
                        width='100%',
                        enable_enterprise_modules=False,
                        fit_columns_on_grid_load=False
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






















