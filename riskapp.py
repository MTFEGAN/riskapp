import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

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

        # Drop weekends
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

@st.cache_data(show_spinner=False)
def adjust_time_zones(price_returns, instrument_country):
    if price_returns.empty:
        return price_returns
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
    volatilities = price_returns_vol.std() * np.sqrt(252) * 100  # annualized in bps
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

def main():
    st.set_page_config(page_title="📈 Fixed Income Portfolio Risk Attribution", layout="wide")
    st.title('📈 Fixed Income Portfolio Risk Attribution')

    st.write("App initialized successfully.")

    # Minimal instrument mapping (expand as needed)
    instruments_data = pd.DataFrame({
        'Ticker': [
            'YM1 Comdty', 'XM1 Comdty', 'TUAFWD Comdty', 'FVAFWD Comdty',
            'TYAFWD Comdty', 'UXYAFWD Comdty', 'WNAFWD Comdty', 'DUAFWD Comdty',
            'OEAFWD Comdty', 'RXAFWD Comdty', 'GAFWD Comdty', 'IKAFWD Comdty',
            'CNAFWD Comdty', 'JBAFWD Comdty', 'CCSWNI1 Curncy', 'ADSW2 Curncy',
            'CDSO2 Curncy', 'USSW2 Curncy', 'EUSA2 Curncy', 'BPSWS2 BGN Curncy',
            'NDSWAP2 BGN Curncy', 'I39302Y Index', 'MPSW2B BGN Curncy',
            'MPSWF2B Curncy', 'SAFR1I2 BGN Curncy', 'CKSW2 BGN Curncy',
            'PZSW2 BGN Curncy', 'KWSWNI2 BGN Curncy', 'CCSWNI2 CMPN Curncy',
            'ADSW5 Curncy', 'CDSO5 Curncy', 'USSW5 Curncy', 'EUSA5 Curncy',
            'BPSWS5 BGN Curncy', 'NDSWAP5 BGN Curncy', 'I39305Y Index',
            'MPSW5E Curncy', 'MPSWF5E Curncy', 'SASW5 Curncy', 'CKSW5 Curncy',
            'PZSW5 Curncy', 'KWSWNI5 Curncy', 'CCSWNI5 Curncy', 'JYSO5 Curncy',
            'ADSW10 Curncy', 'CDSO10 Curncy', 'USSW10 Curncy', 'EUSA10 Curncy',
            'BPSWS10 BGN Curncy', 'NDSWAP10 BGN Curncy', 'ADSW30 Curncy',
            'CDSW30 Curncy', 'USSW30 Curncy', 'EUSA30 Curncy', 'BPSWS30 BGN Curncy',
            'NDSWAP30 BGN Curncy', 'JYSO30 Curncy', 'MPSW10J BGN Curncy',
            'MPSWF10J BGN Curncy', 'SASW10 Curncy', 'CKSW10 Curncy',
            'PZSW10 Curncy', 'KWSWNI10 Curncy', 'CCSWNI10 Curncy', 'BPSWIT10 Curncy'
        ],
        'Instrument Name': [
            'AU 3Y Future', 'AU 10Y Future', 'US 2Y Future', 'US 5Y Future',
            'US 10Y Future', 'US 10Y Ultra Future', 'US 30Y Future', 'DE 2Y Future',
            'DE 5Y Future', 'DE 10Y Future', 'UK 10Y Future', 'IT 10Y Future',
            'CA 10Y Future', 'JP 10Y Future', 'CH 1Y Swap', 'AU 2Y Swap',
            'CA 2Y Swap', 'US 2Y Swap', 'DE 2Y Swap', 'UK 2Y Swap',
            'NZ 2Y Swap', 'BR 2Y Swap', 'MX 2Y Swap', 'MX 2Y Swap OIS',
            'SA 2Y Swap', 'CZ 2Y Swap', 'PO 2Y Swap', 'SK 2Y Swap',
            'CH 2Y Swap', 'AU 5Y Swap', 'CA 5Y Swap', 'US 5Y Swap', 'DE 5Y Swap',
            'UK 5Y Swap', 'NZ 5Y Swap', 'BR 5Y Swap', 'MX 5Y Swap',
            'MX 5Y Swap OIS', 'SA 5Y Swap', 'CZ 5Y Swap', 'PO 5Y Swap',
            'SK 5Y Swap', 'CH 5Y Swap', 'JP 5Y Swap', 'AU 10Y Swap',
            'CA 10Y Swap', 'US 10Y Swap', 'DE 10Y Swap', 'UK 10Y Swap',
            'NZ 10Y Swap', 'AU 30Y Swap', 'CA 30Y Swap', 'US 30Y Swap',
            'DE 30Y Swap', 'UK 30Y Swap', 'NZ 30Y Swap', 'JP 30Y Swap',
            'MX 10Y Swap', 'MX 10Y Swap OIS', 'SA 10Y Swap', 'CZ 10Y Swap',
            'PO 10Y Swap', 'SK 10Y Swap', 'CH 10Y Swap', 'UK 10Y Swap Inf'
        ],
        'Portfolio': [
            'DM', 'DM', 'DM', 'DM', 'DM', 'DM', 'DM', 'DM',
            'DM', 'DM', 'DM', 'DM', 'DM', 'DM', 'EM', 'DM',
            'DM', 'DM', 'DM', 'DM', 'DM', 'EM', 'EM', 'EM',
            'EM', 'EM', 'EM', 'EM', 'EM', 'DM', 'DM', 'DM',
            'DM', 'DM', 'DM', 'EM', 'EM', 'EM', 'EM', 'EM',
            'EM', 'EM', 'EM', 'DM', 'DM', 'DM', 'DM', 'DM',
            'DM', 'DM', 'DM', 'DM', 'DM', 'DM', 'DM', 'DM',
            'DM', 'EM', 'EM', 'EM', 'EM', 'EM', 'EM', 'EM',
            'DM'
        ]
    })

    st.sidebar.header("🔍 Sensitivity Rate Configuration")
    excel_file = 'historical_data.xlsx'

    if not os.path.exists(excel_file):
        st.sidebar.error(f"❌ '{excel_file}' not found.")
        st.stop()

    raw_df = load_historical_data(excel_file)
    if raw_df.empty:
        st.error("No data loaded from Excel. Check the file.")
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

    instrument_portfolio = dict(zip(instruments_data['Instrument Name'], instruments_data['Portfolio']))
    # Map some instruments to their countries (expand as needed)
    instrument_country = {
        'AU 3Y Future': 'AU',
        'AU 10Y Future': 'AU',
        'US 2Y Future': 'US',
        'US 5Y Future': 'US',
        'US 10Y Future': 'US'
    }

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

    cell_style_jscode = JsCode("""
    function(params) {
        if (params.value > 0) {
            const intensity = Math.min(Math.abs(params.value) / 10, 1);
            const green = 255;
            const red_blue = 255 * (1 - intensity);
            return {'backgroundColor': 'rgb(' + red_blue + ',' + green + ',' + red_blue + ')'};
        } else if (params.value < 0) {
            const intensity = Math.min(Math.abs(params.value) / 10, 1);
            const red = 255;
            const green_blue = 255 * (1 - intensity);
            return {'backgroundColor': 'rgb(' + red + ',' + green_blue + ',' + green_blue + ')'};
        }
    };
    """)

    tabs = st.tabs(["📊 Risk Attribution", "📂 Input Positions", "⚙️ Settings"])

    with tabs[1]:
        st.header("🔄 Input Positions")
        st.write("Enter DM and EM positions below.")
        dm_col, em_col = st.columns(2)

        with dm_col:
            st.subheader('📈 DM Portfolio Positions')
            gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
            gb_dm.configure_columns(['Outright', 'Curve', 'Spread'], editable=True, cellStyle=cell_style_jscode)
            gb_dm.configure_column('Instrument', editable=False)
            gb_dm.configure_pagination(enabled=True, paginationPageSize=20)
            grid_options_dm = gb_dm.build()
            grid_response_dm = AgGrid(
                default_positions_dm,
                gridOptions=grid_options_dm,
                height=400,
                width='100%',
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )
            positions_data_dm = grid_response_dm['data']

        with em_col:
            st.subheader('🌍 EM Portfolio Positions')
            gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
            gb_em.configure_columns(['Outright', 'Curve', 'Spread'], editable=True, cellStyle=cell_style_jscode)
            gb_em.configure_column('Instrument', editable=False)
            gb_em.configure_pagination(enabled=True, paginationPageSize=20)
            grid_options_em = gb_em.build()
            grid_response_em = AgGrid(
                default_positions_em,
                gridOptions=grid_options_em,
                height=400,
                width='100%',
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )
            positions_data_em = grid_response_em['data']

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
                    st.error("No data loaded. Please check Excel file.")
                    st.stop()

                df = process_yields(df)
                price_returns = calculate_returns(df)
                if price_returns.empty:
                    st.warning("No returns available. Check data.")
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

                # Expand positions
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
                    st.warning("No active positions. Enter positions and try again.")
                    st.stop()

                expanded_positions_vector = expanded_positions_data.set_index(['Instrument', 'Position Type'])['Position']

                # Ensure sensitivity_rate included
                if sensitivity_rate not in expanded_positions_vector.index.get_level_values('Instrument'):
                    zero_position = pd.Series(
                        0.0,
                        index=pd.MultiIndex.from_tuples([(sensitivity_rate, 'Outright')], names=['Instrument', 'Position Type'])
                    )
                    expanded_positions_vector = pd.concat([expanded_positions_vector, zero_position])

                if expanded_positions_vector.empty:
                    st.warning("No positions after processing. Check inputs.")
                    st.stop()

                expanded_index = expanded_positions_vector.index
                if covariance_matrix.empty:
                    st.warning("Covariance matrix is empty. No overlap in instruments.")
                    st.stop()

                instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()
                missing_instruments = [instr for instr in instruments if instr not in covariance_matrix.index]
                if missing_instruments:
                    st.warning(f"Missing instruments in covariance: {missing_instruments}")
                    # Drop missing instruments
                    drop_labels = [(instr, pt) for instr in missing_instruments for pt in ['Outright', 'Curve', 'Spread']]
                    expanded_positions_vector = expanded_positions_vector.drop(labels=drop_labels, errors='ignore')
                    if expanded_positions_vector.empty:
                        st.warning("All instruments missing from covariance. Cannot proceed.")
                        st.stop()
                    instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()

                # Re-check covariance matrix intersection
                valid_instruments = [instr for instr in instruments if instr in covariance_matrix.index]
                if not valid_instruments:
                    st.warning("No valid instruments after filtering. Cannot proceed.")
                    st.stop()

                covariance_submatrix = covariance_matrix.loc[valid_instruments, valid_instruments]

                # Build expanded covariance matrix
                expanded_cov_matrix = pd.DataFrame(index=expanded_positions_vector.index, columns=expanded_positions_vector.index)
                for pos1 in expanded_positions_vector.index:
                    instr1 = pos1[0]
                    if instr1 in covariance_submatrix.index:
                        expanded_cov_matrix.loc[pos1, :] = covariance_submatrix.loc[instr1, valid_instruments].values
                    else:
                        expanded_cov_matrix.loc[pos1, :] = 0.0

                expanded_cov_matrix = expanded_cov_matrix.astype(float)

                # Compute portfolio variance and volatility
                if expanded_cov_matrix.empty:
                    st.warning("Expanded covariance matrix is empty.")
                    st.stop()

                portfolio_variance = np.dot(expanded_positions_vector.values,
                                            np.dot(expanded_cov_matrix.values, expanded_positions_vector.values))
                if np.isnan(portfolio_variance) or np.isinf(portfolio_variance):
                    st.warning("Invalid portfolio variance. Check data and positions.")
                    st.stop()

                # If variance is zero, no volatility
                if portfolio_variance <= 0:
                    st.warning("Portfolio variance is zero or negative. No meaningful risk calculation possible.")
                    st.stop()

                portfolio_volatility = np.sqrt(portfolio_variance)
                if np.isnan(portfolio_volatility):
                    st.warning("Portfolio volatility is NaN. Check data.")
                    st.stop()

                # Compute risk contributions
                instrument_volatilities = volatilities
                expanded_volatilities = expanded_positions_vector.index.get_level_values('Instrument').map(instrument_volatilities)
                expanded_volatilities = pd.Series(expanded_volatilities.values, index=expanded_positions_vector.index)
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

                # Filter out any zero positions again just to be safe
                risk_contributions_formatted = risk_contributions.copy()
                risk_contributions_formatted = risk_contributions_formatted[
                    ['Instrument', 'Position Type', 'Position', 'Position Stand-alone Volatility',
                     'Instrument Volatility per 1Y Duration (bps)', 'Contribution to Volatility (bps)', 'Percent Contribution (%)', 'Portfolio']
                ]
                risk_contributions_formatted = risk_contributions_formatted[
                    risk_contributions_formatted['Position'].notna() & (risk_contributions_formatted['Position'] != 0)
                ]

                # If still empty, warn
                if risk_contributions_formatted.empty:
                    st.warning("No non-zero positions to show in risk contributions.")
                    st.stop()

                # VaR and cVaR computations
                price_returns_var = adjusted_price_returns.tail(var_lookback_days)
                if price_returns_var.empty:
                    st.warning("No data for VaR computations.")
                    VaR_95_daily = np.nan
                    VaR_99_daily = np.nan
                    cVaR_95_daily = np.nan
                    cVaR_99_daily = np.nan
                else:
                    positions_per_instrument = expanded_positions_vector.groupby('Instrument').sum()
                    available_instruments_var = positions_per_instrument.index.intersection(price_returns_var.columns)
                    if available_instruments_var.empty:
                        st.warning("No instruments overlap for VaR computations.")
                        VaR_95_daily = np.nan
                        VaR_99_daily = np.nan
                        cVaR_95_daily = np.nan
                        cVaR_99_daily = np.nan
                    else:
                        positions_per_instrument = positions_per_instrument.loc[available_instruments_var]
                        price_returns_var = price_returns_var[available_instruments_var]
                        if price_returns_var.empty:
                            st.warning("No data after aligning instruments for VaR.")
                            VaR_95_daily = np.nan
                            VaR_99_daily = np.nan
                            cVaR_95_daily = np.nan
                            cVaR_99_daily = np.nan
                        else:
                            portfolio_returns = price_returns_var.dot(positions_per_instrument) * 100
                            if portfolio_returns.empty:
                                st.warning("No portfolio returns for VaR.")
                                VaR_95_daily = np.nan
                                VaR_99_daily = np.nan
                                cVaR_95_daily = np.nan
                                cVaR_99_daily = np.nan
                            else:
                                VaR_95_daily = -np.percentile(portfolio_returns, 5)
                                VaR_99_daily = -np.percentile(portfolio_returns, 1)
                                cVaR_95_daily = -portfolio_returns[portfolio_returns <= -VaR_95_daily].mean() if (portfolio_returns <= -VaR_95_daily).any() else np.nan
                                cVaR_99_daily = -portfolio_returns[portfolio_returns <= -VaR_99_daily].mean() if (portfolio_returns <= -VaR_99_daily).any() else np.nan

                # Plot risk contributions
                fig_risk_contributions = px.bar(
                    risk_contributions_formatted,
                    x='Instrument',
                    y='Percent Contribution (%)',
                    color='Portfolio',
                    title='Risk Contributions by Instrument',
                    hover_data=['Position', 'Contribution to Volatility (bps)'],
                    height=500
                )
                st.plotly_chart(fig_risk_contributions, use_container_width=True)

                risk_contributions_by_portfolio = risk_contributions_formatted.groupby('Portfolio').agg({
                    'Contribution to Volatility (bps)': 'sum'
                })
                if not risk_contributions_by_portfolio.empty and portfolio_volatility > 0:
                    risk_contributions_by_portfolio['Percent Contribution (%)'] = (risk_contributions_by_portfolio['Contribution to Volatility (bps)'] / portfolio_volatility) * 100
                    fig_portfolio_risk = px.pie(
                        risk_contributions_by_portfolio.reset_index(),
                        names='Portfolio',
                        values='Contribution to Volatility (bps)',
                        title='Risk Contributions by Portfolio',
                        hole=0.4
                    )
                    fig_portfolio_risk.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_portfolio_risk, use_container_width=True)
                else:
                    st.warning("No portfolio-level contributions available.")

                # Display metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                if np.isnan(portfolio_volatility):
                    portfolio_volatility_str = "N/A"
                else:
                    portfolio_volatility_str = f"{portfolio_volatility:.2f} bps"

                var_95_str = f"{VaR_95_daily:.2f} bps" if not np.isnan(VaR_95_daily) else "N/A"
                var_99_str = f"{VaR_99_daily:.2f} bps" if not np.isnan(VaR_99_daily) else "N/A"
                cvar_95_str = f"{cVaR_95_daily:.2f} bps" if not np.isnan(cVaR_95_daily) else "N/A"
                cvar_99_str = f"{cVaR_99_daily:.2f} bps" if not np.isnan(cVaR_99_daily) else "N/A"

                metrics_col1.metric(label="📊 Total Portfolio Volatility", value=portfolio_volatility_str)
                metrics_col2.metric(label="📉 Daily VaR (95%)", value=var_95_str)
                metrics_col3.metric(label="📉 Daily VaR (99%)", value=var_99_str)
                metrics_col4.metric(label="📈 Daily cVaR (95%)", value=cvar_95_str)

                st.subheader('📈 Value at Risk (VaR) and Conditional VaR (cVaR)')
                st.write(f"**Daily VaR at 95%:** {var_95_str}")
                st.write(f"**Daily cVaR at 95%:** {cvar_95_str}")
                st.write(f"**Daily VaR at 99%:** {var_99_str}")
                st.write(f"**Daily cVaR at 99%:** {cvar_99_str}")

                # Sensitivity Analysis without trendline
                if sensitivity_rate in available_columns:
                    # Align data for sensitivity
                    price_returns_var2 = adjusted_price_returns.tail(var_lookback_days)
                    if not price_returns_var2.empty and not positions_per_instrument.empty:
                        common_instruments = positions_per_instrument.index.intersection(price_returns_var2.columns)
                        if not common_instruments.empty:
                            price_returns_var2 = price_returns_var2[common_instruments]
                            portfolio_returns_for_sens = price_returns_var2.dot(positions_per_instrument.loc[common_instruments]) * 100
                            if (sensitivity_rate in price_returns_var2.columns) and (not portfolio_returns_for_sens.empty):
                                sensitivity_returns = price_returns_var2[sensitivity_rate] * 100
                                common_dates = portfolio_returns_for_sens.index.intersection(sensitivity_returns.index)
                                portfolio_returns_aligned = portfolio_returns_for_sens.loc[common_dates]
                                sensitivity_returns_aligned = sensitivity_returns.loc[common_dates]

                                if portfolio_returns_aligned.empty or sensitivity_returns_aligned.empty:
                                    st.warning("Insufficient data for sensitivity analysis.")
                                else:
                                    # Just scatter plot without OLS
                                    fig_sensitivity = px.scatter(
                                        x=sensitivity_returns_aligned,
                                        y=portfolio_returns_aligned,
                                        labels={
                                            'x': f'{sensitivity_rate} Returns (bps)',
                                            'y': 'Portfolio Returns (bps)'
                                        },
                                        title=f'Relationship between Portfolio Returns and {sensitivity_rate} Returns',
                                        height=500
                                    )
                                    st.plotly_chart(fig_sensitivity, use_container_width=True)
                            else:
                                st.write("No overlapping data for sensitivity rate in VaR dataset.")
                        else:
                            st.write("No common instruments for sensitivity analysis.")
                    else:
                        st.write("Not enough data for sensitivity analysis.")
                else:
                    st.warning(f"'{sensitivity_rate}' not found in data.")

                st.subheader('📄 Detailed Risk Contributions by Instrument')
                if not risk_contributions_formatted.empty:
                    gb_risk = GridOptionsBuilder.from_dataframe(risk_contributions_formatted)
                    gb_risk.configure_pagination(enabled=True, paginationPageSize=20)
                    gb_risk.configure_side_bar()
                    gb_risk.configure_default_column(editable=False, groupable=True)
                    grid_options_risk = gb_risk.build()

                    AgGrid(
                        risk_contributions_formatted,
                        gridOptions=grid_options_risk,
                        height=400,
                        width='100%',
                        allow_unsafe_jscode=True,
                        enable_enterprise_modules=False
                    )

                    csv = risk_contributions_formatted.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Risk Contributions as CSV",
                        data=csv,
                        file_name='risk_contributions.csv',
                        mime='text/csv',
                    )
                else:
                    st.write("No risk contributions to display.")

if __name__ == '__main__':
    main()






