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

@st.cache_data(show_spinner=False)
def process_yields(df):
    if 'AU 3Y Future' in df.columns:
        df['AU 3Y Future'] = 100 - df['AU 3Y Future']
    if 'AU 10Y Future' in df.columns:
        df['AU 10Y Future'] = 100 - df['AU 10Y Future']
    return df

@st.cache_data(show_spinner=False)
def calculate_returns(df):
    if df.empty:
        return pd.DataFrame()
    returns = df.diff().dropna()
    price_returns = returns * -1
    return price_returns

instrument_country = {
    'AU 3Y Future': 'AU',
    'AU 10Y Future': 'AU',
    'US 2Y Future': 'US',
    'US 5Y Future': 'US',
    'US 10Y Future': 'US',
}

@st.cache_data(show_spinner=False)
def adjust_time_zones(price_returns, instrument_country):
    non_lag_countries = ['JP', 'AU', 'SK', 'CH']
    instrument_countries = pd.Series([instrument_country.get(instr, 'Other') for instr in price_returns.columns],
                                     index=price_returns.columns)
    instruments_to_lag = instrument_countries[~instrument_countries.isin(non_lag_countries)].index.tolist()
    adjusted_price_returns = price_returns.copy()
    if instruments_to_lag:
        adjusted_price_returns[instruments_to_lag] = adjusted_price_returns[instruments_to_lag].shift(-1)
    adjusted_price_returns = adjusted_price_returns.dropna()
    return adjusted_price_returns

@st.cache_data(show_spinner=False)
def calculate_volatilities(adjusted_price_returns, lookback_days):
    if adjusted_price_returns.empty:
        return pd.Series(dtype=float)
    price_returns_vol = adjusted_price_returns.tail(lookback_days)
    if price_returns_vol.empty:
        return pd.Series(dtype=float)
    volatilities = price_returns_vol.std() * np.sqrt(252) * 100
    return volatilities

@st.cache_data(show_spinner=False)
def calculate_covariance_matrix(adjusted_price_returns, lookback_days):
    if adjusted_price_returns.empty:
        return pd.DataFrame()
    price_returns_cov = adjusted_price_returns.tail(lookback_days)
    if price_returns_cov.empty:
        return pd.DataFrame()
    covariance_matrix = price_returns_cov.cov() * 252 * 10000
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

def main():
    st.title('📈 Fixed Income Portfolio Risk Attribution')
    st.write("App initialized successfully.")

    instruments_data = pd.DataFrame({
        "Ticker": [
            "YM1 Comdty","XM1 Comdty","TUAFWD Comdty","FVAFWD Comdty","TYAFWD Comdty","UXYAFWD Comdty","WNAFWD Comdty","DUAFWD Comdty","OEAFWD Comdty","RXAFWD Comdty","GAFWD Comdty","IKAFWD Comdty","CNAFWD Comdty","JBAFWD Comdty","CCSWNI1 Curncy","ADSW2 Curncy","CDSO2 Curncy","USSW2 Curncy","EUSA2 Curncy","BPSWS2 BGN Curncy","NDSWAP2 BGN Curncy","I39302Y Index","MPSW2B BGN Curncy","MPSWF2B Curncy","SAFR1I2 BGN Curncy","CKSW2 BGN Curncy","PZSW2 BGN Curncy","KWSWNI2 BGN Curncy","CCSWNI2 CMPN Curncy","ADSW5 Curncy","CDSO5 Curncy","USSW5 Curncy","EUSA5 Curncy","BPSWS5 BGN Curncy","NDSWAP5 BGN Curncy","I39305Y Index","MPSW5E Curncy","MPSWF5E Curncy","SASW5 Curncy","CKSW5 Curncy","PZSW5 Curncy","KWSWNI5 Curncy","CCSWNI5 Curncy","JYSO5 Curncy","ADSW10 Curncy","CDSW10 Curncy","USSW10 Curncy","EUSA10 Curncy","BPSWS10 BGN Curncy","NDSWAP10 BGN Curncy","ADSW30 Curncy","CDSW30 Curncy","USSW30 Curncy","EUSA30 Curncy","BPSWS30 BGN Curncy","NDSWAP30 BGN Curncy","JYSO30 Curncy","MPSW10J BGN Curncy","MPSWF10J BGN Curncy","SASW10 Curncy","CKSW10 Curncy","PZSW10 Curncy","KWSWNI10 Curncy","CCSWNI10 Curncy","BPSWIT10 Curncy"
        ],
        "Instrument Name": [
            "AU 3Y Future","AU 10Y Future","US 2Y Future","US 5Y Future","US 10Y Future","US 10Y Ultra Future","US 30Y Future","DE 2Y Future","DE 5Y Future","DE 10Y Future","UK 10Y Future","IT 10Y Future","CA 10Y Future","JP 10Y Future","CH 1Y Swap","AU 2Y Swap","CA 2Y Swap","US 2Y Swap","DE 2Y Swap","UK 2Y Swap","NZ 2Y Swap","BR 2Y Swap","MX 2Y Swap","MX 2Y Swap OIS","SA 2Y Swap","CZ 2Y Swap","PO 2Y Swap","SK 2Y Swap","CH 2Y Swap","AU 5Y Swap","CA 5Y Swap","US 5Y Swap","DE 5Y Swap","UK 5Y Swap","NZ 5Y Swap","BR 5Y Swap","MX 5Y Swap","MX 5Y Swap OIS","SA 5Y Swap","CZ 5Y Swap","PO 5Y Swap","SK 5Y Swap","CH 5Y Swap","JP 5Y Swap","AU 10Y Swap","CA 10Y Swap","US 10Y Swap","DE 10Y Swap","UK 10Y Swap","NZ 10Y Swap","AU 30Y Swap","CA 30Y Swap","US 30Y Swap","DE 30Y Swap","UK 30Y Swap","NZ 30Y Swap","JP 30Y Swap","MX 10Y Swap","MX 10Y Swap OIS","SA 10Y Swap","CZ 10Y Swap","PO 10Y Swap","SK 10Y Swap","CH 10Y Swap","UK 10Y Swap Inf"
        ],
        "Portfolio": [
            "DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","EM","DM","DM","DM","DM","DM","DM","EM","EM","EM","EM","EM","EM","EM","EM","DM","DM","DM","DM","DM","DM","EM","EM","EM","EM","EM","EM","EM","EM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","DM","EM","EM","EM","EM","EM","EM","EM","DM"
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
    if 'US 10Y Future' in available_columns:
        default_index = available_columns.index('US 10Y Future')
    else:
        default_index = 0
    sensitivity_rate = st.sidebar.selectbox(
        'Select sensitivity instrument:',
        options=available_columns,
        index=default_index
    )

    tabs = st.tabs(["📊 Risk Attribution", "📂 Input Positions", "⚙️ Settings"])

    with tabs[1]:
        st.header("🔄 Input Positions")
        st.write("Instrument column is now wider. Columns are editable and resizable.")

        # DM table
        st.subheader('📈 DM Portfolio Positions')
        gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
        gb_dm.configure_default_column(editable=True, resizable=True)
        gb_dm.configure_column('Instrument', editable=False, width=600)
        gb_dm.configure_column('Outright', width=200)
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

        # EM table
        st.subheader('🌍 EM Portfolio Positions')
        gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
        gb_em.configure_default_column(editable=True, resizable=True)
        gb_em.configure_column('Instrument', editable=False, width=600)
        gb_em.configure_column('Outright', width=200)
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

    with tabs[2]:
        st.header("⚙️ Configuration Settings")
        volatility_period_options = {
            '📅 1 month': 21,
            '📆 3 months': 63,
            '📅 6 months': 126,
            '🗓️ 1 year': 252,
            '📅 3 years': 756,
            '📆 5 years': 1260,
            '📅 10 years': 2520
        }
        volatility_period = st.selectbox('Volatility lookback:', list(volatility_period_options.keys()), index=3)
        volatility_lookback_days = volatility_period_options[volatility_period]

        var_period_options = {
            '📆 5 years': 1260,
            '📆 10 years': 2520,
            '📆 15 years': 3780
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

                df = process_yields(df)
                price_returns = calculate_returns(df)
                if price_returns.empty:
                    st.warning("No returns computed.")
                    st.stop()

                adjusted_price_returns = adjust_time_zones(price_returns, instrument_country)
                if adjusted_price_returns.empty:
                    st.warning("No data after time zone adjustment.")
                    st.stop()

                volatilities = calculate_volatilities(adjusted_price_returns, volatility_lookback_days)
                covariance_matrix = calculate_covariance_matrix(adjusted_price_returns, var_lookback_days)

                positions_data_dm = pd.DataFrame(positions_data_dm).astype({'Outright': float, 'Curve': float, 'Spread': float})
                positions_data_em = pd.DataFrame(positions_data_em).astype({'Outright': float, 'Curve': float, 'Spread': float})
                positions_data_dm['Portfolio'] = 'DM'
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
                    st.warning("No active positions entered. Please enter positions and try again.")
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
                    risk_contributions_formatted['Position'].notna() & (risk_contributions_formatted['Position'] != 0)
                ]
                numeric_cols = ['Position', 'Position Stand-alone Volatility', 'Instrument Volatility per 1Y Duration (bps)', 'Contribution to Volatility (bps)', 'Percent Contribution (%)']
                risk_contributions_formatted[numeric_cols] = risk_contributions_formatted[numeric_cols].round(2)

                price_returns_var = adjusted_price_returns.tail(var_lookback_days)
                def fmt_val(x):
                    return f"{x:.2f} bps" if (not np.isnan(x) and not np.isinf(x)) else "N/A"

                VaR_95_daily, VaR_99_daily, cVaR_95_daily, cVaR_99_daily = (np.nan, np.nan, np.nan, np.nan)
                positions_per_instrument = expanded_positions_vector.groupby('Instrument').sum()
                if not price_returns_var.empty:
                    available_instruments_var = positions_per_instrument.index.intersection(price_returns_var.columns)
                    if not available_instruments_var.empty:
                        positions_per_instrument = positions_per_instrument.loc[available_instruments_var]
                        price_returns_var = price_returns_var[available_instruments_var]
                        if not price_returns_var.empty:
                            portfolio_returns = price_returns_var.dot(positions_per_instrument) * 100
                            if not portfolio_returns.empty:
                                VaR_95_daily = -np.percentile(portfolio_returns, 5)
                                VaR_99_daily = -np.percentile(portfolio_returns, 1)
                                cVaR_95_daily = -portfolio_returns[portfolio_returns <= -VaR_95_daily].mean() if (portfolio_returns <= -VaR_95_daily).any() else np.nan
                                cVaR_99_daily = -portfolio_returns[portfolio_returns <= -VaR_99_daily].mean() if (portfolio_returns <= -VaR_99_daily).any() else np.nan

                # Compute Betas
                portfolio_beta = np.nan
                instrument_betas = {}
                if (sensitivity_rate in price_returns_var.columns) and (not price_returns_var.empty) and (not positions_per_instrument.empty):
                    portfolio_returns_for_beta = price_returns_var.dot(positions_per_instrument) * 100
                    us10yr_returns = price_returns_var[sensitivity_rate] * 100
                    portfolio_beta = compute_beta(portfolio_returns_for_beta, us10yr_returns)
                    for instr in positions_per_instrument.index:
                        pos_val = positions_per_instrument[instr]
                        if pos_val != 0:
                            instr_return = price_returns_var[instr]*pos_val*100
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
                metrics_col2.metric(label="📉 Daily VaR (95%)", value=fmt_val(VaR_95_daily))
                metrics_col3.metric(label="📉 Daily VaR (99%)", value=fmt_val(VaR_99_daily))
                metrics_col4.metric(label="📈 Daily cVaR (95%)", value=fmt_val(cVaR_95_daily))

                st.subheader('📈 Value at Risk (VaR) and Conditional VaR (cVaR)')
                st.write(f"**Daily VaR at 95%:** {fmt_val(VaR_95_daily)}")
                st.write(f"**Daily cVaR at 95%:** {fmt_val(cVaR_95_daily)}")
                st.write(f"**Daily VaR at 99%:** {fmt_val(VaR_99_daily)}")
                st.write(f"**Daily cVaR at 99%:** {fmt_val(cVaR_99_daily)}")

                # US Beta Section
                st.subheader("📉 Beta to US 10yr Rates")
                if not np.isnan(portfolio_beta):
                    st.write(f"**Portfolio Beta to {sensitivity_rate}:** {portfolio_beta:.4f}")
                    if instrument_betas:
                        st.write("**Instrument Betas to US 10yr Rates (including Position):**")
                        beta_data = []
                        for instr, (pos_val, b) in instrument_betas.items():
                            beta_data.append({'Instrument': instr, 'Position': pos_val, 'Beta': b})
                        beta_df = pd.DataFrame(beta_data)
                        beta_df['Position'] = beta_df['Position'].round(2)
                        beta_df['Beta'] = beta_df['Beta'].round(4)
                        st.dataframe(beta_df)

                        # Footnote on interpretation
                        st.markdown("*Footnote:* If the US 10-year yield moves by 1bp, the portfolio performance changes by Beta × 1bp. For example, if Beta is 0.5 and US 10-year yields fall by 1bp, the portfolio is expected to rise by approximately 0.5bps.")
                    else:
                        st.write("No individual instrument betas to display.")
                else:
                    st.write("No portfolio beta computed. Check US 10Y Future data and positions.")

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
















