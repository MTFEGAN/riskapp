import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import OrderedDict  # Ensure OrderedDict is imported

# Import AgGrid for advanced data grid features
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Caching the data loading and processing functions to improve performance
@st.cache_data
def load_historical_data(excel_file, sensitivity_rate):
    try:
        # Load Excel file
        excel = pd.ExcelFile(excel_file)
        # Find the sheet that contains the sensitivity rate column
        sheet_with_sensitivity = None
        for sheet in excel.sheet_names:
            df_sheet = excel.parse(sheet_name=sheet)
            # Clean column names by stripping spaces and making lowercase for matching
            df_sheet.columns = df_sheet.columns.str.strip().str.lower()
            if sensitivity_rate.lower() in df_sheet.columns:
                sheet_with_sensitivity = sheet
                # Rename columns back to original for consistency
                df_sheet.columns = excel.parse(sheet_name=sheet).columns
                break
        if sheet_with_sensitivity is None:
            st.warning(f"'{sensitivity_rate}' data not available for sensitivity analysis.")
            st.write("**Available Sheets:**")
            st.write(excel.sheet_names)
            return None
        # Read the sheet with the sensitivity rate
        df = pd.read_excel(excel_file, sheet_name=sheet_with_sensitivity, index_col='date', parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_data
def process_yields(df):
    # Adjust yields for AU 3Y and 10Y futures if they exist
    if 'AU 3Y Future' in df.columns:
        df['AU 3Y Future'] = 100 - df['AU 3Y Future']
    if 'AU 10Y Future' in df.columns:
        df['AU 10Y Future'] = 100 - df['AU 10Y Future']
    return df

@st.cache_data
def calculate_returns(df):
    # Calculate daily yield changes (returns)
    returns = df.diff().dropna()
    # Correct the sign of returns to reflect price changes
    # Since bond prices move inversely to yields, we multiply by -1
    price_returns = returns * -1
    return price_returns

@st.cache_data
def adjust_time_zones(price_returns, instrument_country):
    # Adjust returns for time zone differences
    non_lag_countries = ['JP', 'AU', 'SK', 'CH']
    instrument_countries = pd.Series([instrument_country.get(instr, 'Other') for instr in price_returns.columns], index=price_returns.columns)

    instruments_to_lag = instrument_countries[~instrument_countries.isin(non_lag_countries)].index.tolist()
    instruments_not_to_lag = instrument_countries[instrument_countries.isin(non_lag_countries)].index.tolist()

    adjusted_price_returns = price_returns.copy()

    if instruments_to_lag:
        adjusted_price_returns[instruments_to_lag] = adjusted_price_returns[instruments_to_lag].shift(-1)

    # Drop rows with NaN values resulting from the shift
    adjusted_price_returns = adjusted_price_returns.dropna()
    return adjusted_price_returns

@st.cache_data
def calculate_volatilities(adjusted_price_returns, lookback_days):
    # Use the selected lookback period for volatility calculations
    price_returns_vol = adjusted_price_returns.tail(lookback_days)
    # Calculate annualized volatilities in basis points
    volatilities = price_returns_vol.std() * np.sqrt(252) * 100  # Annualized volatility in bps
    return volatilities

@st.cache_data
def calculate_covariance_matrix(adjusted_price_returns, lookback_days):
    # Use the selected lookback period for covariance calculations
    price_returns_cov = adjusted_price_returns.tail(lookback_days)
    # Calculate the covariance matrix (annualized) in bps^2
    covariance_matrix = price_returns_cov.cov() * 252 * 10000  # Multiply by 100^2 to convert to bps^2
    return covariance_matrix

def main():
    # Set page configuration to use the full width
    st.set_page_config(page_title="Fixed Income Portfolio Risk Attribution", layout="wide")
    st.title('Fixed Income Portfolio Risk Attribution')

    # Step 1: Define the instruments and their portfolios

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

    # Define the sensitivity rate column
    sensitivity_rate = 'US 10Y Future'  # Change this if your column is named differently

    # Create tickers_data mapping Ticker to Instrument Name
    tickers_data = OrderedDict(zip(instruments_data['Ticker'], instruments_data['Instrument Name']))

    # Create a mapping from Instrument Name to Portfolio
    instrument_portfolio = dict(zip(instruments_data['Instrument Name'], instruments_data['Portfolio']))

    # Create a mapping from Instrument Name to Country
    instrument_country = {
        'AU 3Y Future': 'AU',
        'AU 10Y Future': 'AU',
        'US 2Y Future': 'US',
        'US 5Y Future': 'US',
        'US 10Y Future': 'US',
        'US 10Y Ultra Future': 'US',
        'US 30Y Future': 'US',
        'DE 2Y Future': 'DE',
        'DE 5Y Future': 'DE',
        'DE 10Y Future': 'DE',
        'UK 10Y Future': 'UK',
        'IT 10Y Future': 'IT',
        'CA 10Y Future': 'CA',
        'JP 10Y Future': 'JP',
        'CH 1Y Swap': 'CH',
        'AU 2Y Swap': 'AU',
        'CA 2Y Swap': 'CA',
        'US 2Y Swap': 'US',
        'DE 2Y Swap': 'DE',
        'UK 2Y Swap': 'UK',
        'NZ 2Y Swap': 'NZ',
        'BR 2Y Swap': 'BR',
        'MX 2Y Swap': 'MX',
        'MX 2Y Swap OIS': 'MX',
        'SA 2Y Swap': 'SA',
        'CZ 2Y Swap': 'CZ',
        'PO 2Y Swap': 'PL',  # Poland
        'SK 2Y Swap': 'SK',
        'CH 2Y Swap': 'CH',
        'AU 5Y Swap': 'AU',
        'CA 5Y Swap': 'CA',
        'US 5Y Swap': 'US',
        'DE 5Y Swap': 'DE',
        'UK 5Y Swap': 'UK',
        'NZ 5Y Swap': 'NZ',
        'BR 5Y Swap': 'BR',
        'MX 5Y Swap': 'MX',
        'MX 5Y Swap OIS': 'MX',
        'SA 5Y Swap': 'SA',
        'CZ 5Y Swap': 'CZ',
        'PO 5Y Swap': 'PL',
        'SK 5Y Swap': 'SK',
        'CH 5Y Swap': 'CH',
        'JP 5Y Swap': 'JP',
        'AU 10Y Swap': 'AU',
        'CA 10Y Swap': 'CA',
        'US 10Y Swap': 'US',
        'DE 10Y Swap': 'DE',
        'UK 10Y Swap': 'UK',
        'NZ 10Y Swap': 'NZ',
        'AU 30Y Swap': 'AU',
        'CA 30Y Swap': 'CA',
        'US 30Y Swap': 'US',
        'DE 30Y Swap': 'DE',
        'UK 30Y Swap': 'UK',
        'NZ 30Y Swap': 'NZ',
        'JP 30Y Swap': 'JP',
        'MX 10Y Swap': 'MX',
        'MX 10Y Swap OIS': 'MX',
        'SA 10Y Swap': 'SA',
        'CZ 10Y Swap': 'CZ',
        'PO 10Y Swap': 'PL',
        'SK 10Y Swap': 'SK',
        'CH 10Y Swap': 'CH',
        'UK 10Y Swap Inf': 'UK',
        'CCSWNI1 Curncy': 'CH',
        'CCSWNI2 CMPN Curncy': 'CH',
        'CCSWNI5 Curncy': 'CH',
        'CCSWNI10 Curncy': 'CH',
        'I39302Y Index': 'BR',
        'I39305Y Index': 'BR',
        'MPSW2B BGN Curncy': 'BR',
        'MPSWF2B Curncy': 'BR',
        'MPSW5E Curncy': 'MX',
        'MPSWF5E Curncy': 'MX',
        'MPSW10J BGN Curncy': 'JP',
        'MPSWF10J BGN Curncy': 'JP',
        'SASW5 Curncy': 'SA',
        'SASW10 Curncy': 'SA',
        'CKSW5 Curncy': 'CZ',
        'CKSW10 Curncy': 'CZ',
        'PZSW5 Curncy': 'PL',
        'PZSW10 Curncy': 'PL',
        'NDSWAP2 BGN Curncy': 'NZ',
        'NDSWAP5 BGN Curncy': 'NZ',
        'NDSWAP10 BGN Curncy': 'NZ',
        'NDSWAP30 BGN Curncy': 'NZ',
        'BPSWS2 BGN Curncy': 'UK',
        'BPSWS5 BGN Curncy': 'UK',
        'BPSWS10 BGN Curncy': 'UK',
        'BPSWS30 BGN Curncy': 'UK',
        'BPSWIT10 Curncy': 'UK',
        'ADSW2 Curncy': 'AU',
        'ADSW5 Curncy': 'AU',
        'ADSW10 Curncy': 'AU',
        'ADSW30 Curncy': 'AU',
        'CDSO2 Curncy': 'CA',
        'CDSO5 Curncy': 'CA',
        'CDSO10 Curncy': 'CA',
        'CDSW30 Curncy': 'CA',
        'USSW2 Curncy': 'US',
        'USSW5 Curncy': 'US',
        'USSW10 Curncy': 'US',
        'USSW30 Curncy': 'US',
        'EUSA2 Curncy': 'EU',
        'EUSA5 Curncy': 'EU',
        'EUSA10 Curncy': 'EU',
        'EUSA30 Curncy': 'EU',
        'JYSO5 Curncy': 'JP',
        'JYSO30 Curncy': 'JP',
    }

    # Add 'Country' column to instruments_data
    instruments_data['Country'] = instruments_data['Instrument Name'].map(instrument_country)

    # Separate DM and EM instruments
    dm_instruments = instruments_data[instruments_data['Portfolio'] == 'DM']['Instrument Name'].tolist()
    em_instruments = instruments_data[instruments_data['Portfolio'] == 'EM']['Instrument Name'].tolist()

    # Step 2: Create positions DataFrames with default values for DM and EM
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

    # Define JavaScript code for conditional formatting
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.value > 0) {
            const intensity = Math.min(Math.abs(params.value) / 10, 1);
            const green = 255;
            const red_blue = 255 * (1 - intensity);
            return {
                'backgroundColor': 'rgb(' + red_blue + ',' + green + ',' + red_blue + ')'
            }
        } else if (params.value < 0) {
            const intensity = Math.min(Math.abs(params.value) / 10, 1);
            const red = 255;
            const green_blue = 255 * (1 - intensity);
            return {
                'backgroundColor': 'rgb(' + red + ',' + green_blue + ',' + green_blue + ')'
            }
        }
    };
    """)

    st.subheader('Input Positions for DM Portfolio')
    st.write('Enter your positions for the Developed Markets (DM) portfolio below:')
    gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
    gb_dm.configure_columns(['Outright', 'Curve', 'Spread'], editable=True, cellStyle=cell_style_jscode)
    gb_dm.configure_column('Instrument', editable=False)
    gb_dm.configure_pagination(enabled=True, paginationPageSize=20)  # Enable pagination
    grid_options_dm = gb_dm.build()
    grid_response_dm = AgGrid(
        default_positions_dm,
        gridOptions=grid_options_dm,
        height=300,
        width='100%',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        reload_data=True
    )
    positions_data_dm = grid_response_dm['data']

    st.subheader('Input Positions for EM Portfolio')
    st.write('Enter your positions for the Emerging Markets (EM) portfolio below:')
    gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
    gb_em.configure_columns(['Outright', 'Curve', 'Spread'], editable=True, cellStyle=cell_style_jscode)
    gb_em.configure_column('Instrument', editable=False)
    gb_em.configure_pagination(enabled=True, paginationPageSize=20)  # Enable pagination
    grid_options_em = gb_em.build()
    grid_response_em = AgGrid(
        default_positions_em,
        gridOptions=grid_options_em,
        height=300,
        width='100%',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        reload_data=True
    )
    positions_data_em = grid_response_em['data']

    # Dropdown for volatility lookback period
    st.subheader('Select Lookback Periods')
    volatility_period_options = {
        '1 month': 21,
        '3 months': 63,
        '6 months': 126,
        '1 year': 252,
        '3 years': 756,
        '5 years': 1260,
        '10 years': 2520
    }
    volatility_period = st.selectbox('Select lookback period for volatility calculations:', list(volatility_period_options.keys()), index=3)
    volatility_lookback_days = volatility_period_options[volatility_period]

    # Dropdown for VaR calculation period
    var_period_options = {
        '5 years': 1260,
        '10 years': 2520,
        '15 years': 3780
    }
    var_period = st.selectbox('Select lookback period for VaR calculations:', list(var_period_options.keys()), index=1)
    var_lookback_days = var_period_options[var_period]

    # Step 3: Run Risk Attribution when button is clicked
    if st.button('Run Risk Attribution'):
        st.write('Calculating risk attribution...')
        st.write('This may take a few moments.')

        # Step 4: Load Historical Data from the Excel File
        excel_file = 'historical_data.xlsx'
        if not os.path.exists(excel_file):
            st.error(f"Excel file '{excel_file}' not found. Please ensure it is in the app directory.")
            return

        # Load and process historical data
        df = load_historical_data(excel_file, sensitivity_rate)
        if df is None:
            return  # Error message already displayed

        df = process_yields(df)
        price_returns = calculate_returns(df)
        adjusted_price_returns = adjust_time_zones(price_returns, instrument_country)

        # Check if adjusted_price_returns is empty
        if adjusted_price_returns.empty:
            st.warning("No data available after adjusting for time zones.")
            return

        # Calculate volatilities and covariance matrix
        volatilities = calculate_volatilities(adjusted_price_returns, volatility_lookback_days)
        covariance_matrix = calculate_covariance_matrix(adjusted_price_returns, volatility_lookback_days)

        # Step 5: Process User Input Positions

        # Convert positions_data to the required format
        positions_data_dm = pd.DataFrame(positions_data_dm).astype({'Outright': float, 'Curve': float, 'Spread': float})
        positions_data_em = pd.DataFrame(positions_data_em).astype({'Outright': float, 'Curve': float, 'Spread': float})

        # Combine DM and EM positions
        positions_data_dm['Portfolio'] = 'DM'
        positions_data_em['Portfolio'] = 'EM'
        positions_data = pd.concat([positions_data_dm, positions_data_em], ignore_index=True)

        # Expand positions to have separate entries for 'Outright', 'Curve', 'Spread'
        positions_list = []
        for idx, row in positions_data.iterrows():
            instrument = row['Instrument']
            portfolio = row['Portfolio']
            for position_type in ['Outright', 'Curve', 'Spread']:
                position_value = row[position_type]
                if position_value != 0:
                    positions_list.append({
                        'Instrument': instrument,
                        'Position Type': position_type,
                        'Position': position_value,
                        'Portfolio': portfolio
                    })

        expanded_positions_data = pd.DataFrame(positions_list)

        if expanded_positions_data.empty:
            st.warning("No positions entered. Please enter positions and try again.")
            return

        # Create expanded positions vector
        expanded_positions_vector = expanded_positions_data.set_index(['Instrument', 'Position Type'])['Position']

        # Ensure that the sensitivity rate is included even if the user has no position in it
        if sensitivity_rate not in expanded_positions_vector.index.get_level_values('Instrument'):
            # Add 'US 10Y Future' with zero position
            zero_position = pd.Series(0.0, index=pd.MultiIndex.from_tuples([(sensitivity_rate, 'Outright')]))
            expanded_positions_vector = expanded_positions_vector.append(zero_position)

        # Create an empty DataFrame for the expanded covariance matrix
        expanded_index = expanded_positions_vector.index
        expanded_cov_matrix = pd.DataFrame(index=expanded_index, columns=expanded_index)

        # Populate the expanded covariance matrix (vectorized)
        instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()
        # Extract relevant covariance submatrix
        covariance_submatrix = covariance_matrix.loc[instruments, instruments]

        # Assign covariance values to the expanded covariance matrix
        for pos1 in expanded_index:
            for pos2 in expanded_index:
                instr1 = pos1[0]
                instr2 = pos2[0]
                var_i_j = covariance_submatrix.loc[instr1, instr2]
                expanded_cov_matrix.loc[pos1, pos2] = var_i_j

        expanded_cov_matrix = expanded_cov_matrix.astype(float)

        # Compute Portfolio Volatility
        portfolio_variance = np.dot(expanded_positions_vector.T, np.dot(expanded_cov_matrix, expanded_positions_vector))
        portfolio_volatility = np.sqrt(portfolio_variance)  # Volatility in bps

        # Compute Each Position's Stand-alone Volatility and Contribution

        # Map instrument volatilities to expanded positions
        instrument_volatilities = volatilities
        expanded_volatilities = expanded_positions_vector.index.get_level_values('Instrument').map(instrument_volatilities)
        expanded_volatilities = pd.Series(expanded_volatilities.values, index=expanded_positions_vector.index)

        # Instrument Volatility per 1Y Duration is the instrument's annualized volatility in bps
        instrument_volatilities_per_1y = expanded_volatilities

        # Stand-alone volatilities
        standalone_volatilities = expanded_positions_vector.abs() * expanded_volatilities

        # Marginal contributions to variance
        marginal_contributions = expanded_cov_matrix.dot(expanded_positions_vector)

        # Contribution to variance
        contribution_to_variance = expanded_positions_vector * marginal_contributions

        # Contribution to volatility
        contribution_to_volatility = contribution_to_variance / portfolio_volatility

        # Percentage contribution
        percent_contribution = (contribution_to_volatility / portfolio_volatility) * 100

        # Create a DataFrame for reporting
        risk_contributions = expanded_positions_data.copy()
        # Ensure 'US 10Y Future' is included
        risk_contributions = risk_contributions.append({
            'Instrument': sensitivity_rate,
            'Position Type': 'Outright',
            'Position': 0.0,
            'Portfolio': 'DM'  # Assign to DM or another appropriate portfolio if necessary
        }, ignore_index=True)

        # Merge to align with expanded_volatilities
        risk_contributions = risk_contributions.set_index(['Instrument', 'Position Type']).reindex(expanded_positions_vector.index).reset_index()

        risk_contributions['Position Stand-alone Volatility'] = standalone_volatilities.values
        risk_contributions['Contribution to Volatility (bps)'] = contribution_to_volatility.values
        risk_contributions['Percent Contribution (%)'] = percent_contribution.values
        risk_contributions['Instrument Volatility per 1Y Duration (bps)'] = instrument_volatilities_per_1y.values

        # Format the DataFrame
        risk_contributions_formatted = risk_contributions.copy()
        risk_contributions_formatted['Position'] = risk_contributions_formatted['Position'].astype(float)
        risk_contributions_formatted['Position Stand-alone Volatility'] = risk_contributions_formatted['Position Stand-alone Volatility'].astype(float)
        risk_contributions_formatted['Contribution to Volatility (bps)'] = risk_contributions_formatted['Contribution to Volatility (bps)'].astype(float)
        risk_contributions_formatted['Percent Contribution (%)'] = risk_contributions_formatted['Percent Contribution (%)'].astype(float)
        risk_contributions_formatted['Instrument Volatility per 1Y Duration (bps)'] = risk_contributions_formatted['Instrument Volatility per 1Y Duration (bps)'].astype(float)

        # Rearrange columns
        risk_contributions_formatted = risk_contributions_formatted[
            ['Instrument', 'Position Type', 'Position', 'Position Stand-alone Volatility',
             'Instrument Volatility per 1Y Duration (bps)', 'Contribution to Volatility (bps)', 'Percent Contribution (%)', 'Portfolio']
        ]

        # Add bar charts to 'Percent Contribution (%)' column
        styled_risk_contributions = risk_contributions_formatted.style.bar(
            subset=['Percent Contribution (%)'], color='#d65f5f'
        ).format({
            'Position': '{:,.4f}',
            'Position Stand-alone Volatility': '{:,.2f}',
            'Instrument Volatility per 1Y Duration (bps)': '{:,.2f}',
            'Contribution to Volatility (bps)': '{:,.2f}',
            'Percent Contribution (%)': '{:,.2f}'
        })

        # Render the styled DataFrame in Streamlit
        st.subheader('Risk Contributions by Instrument')
        st.markdown(styled_risk_contributions.to_html(), unsafe_allow_html=True)

        # Compute risk contributions by Portfolio (DM and EM)
        risk_contributions_by_portfolio = risk_contributions.groupby('Portfolio').agg({
            'Contribution to Volatility (bps)': 'sum'
        })
        risk_contributions_by_portfolio['Percent Contribution (%)'] = (risk_contributions_by_portfolio['Contribution to Volatility (bps)'] / portfolio_volatility) * 100

        # Display risk contributions by Portfolio
        st.subheader('Risk Contributions by Portfolio')
        st.write(risk_contributions_by_portfolio)

        # Perform risk attribution for DM and EM portfolios separately
        # For DM Portfolio
        dm_positions_data = expanded_positions_data[expanded_positions_data['Portfolio'] == 'DM']
        if not dm_positions_data.empty:
            dm_positions_vector = dm_positions_data.set_index(['Instrument', 'Position Type'])['Position']
            # Extract relevant covariance submatrix
            dm_cov_submatrix = expanded_cov_matrix.loc[dm_positions_vector.index, dm_positions_vector.index]
            dm_variance = np.dot(dm_positions_vector.T, np.dot(dm_cov_submatrix, dm_positions_vector))
            dm_volatility = np.sqrt(dm_variance)
        else:
            dm_volatility = 0

        # For EM Portfolio
        em_positions_data = expanded_positions_data[expanded_positions_data['Portfolio'] == 'EM']
        if not em_positions_data.empty:
            em_positions_vector = em_positions_data.set_index(['Instrument', 'Position Type'])['Position']
            # Extract relevant covariance submatrix
            em_cov_submatrix = expanded_cov_matrix.loc[em_positions_vector.index, em_positions_vector.index]
            em_variance = np.dot(em_positions_vector.T, np.dot(em_cov_submatrix, em_positions_vector))
            em_volatility = np.sqrt(em_variance)
        else:
            em_volatility = 0

        st.write(f"**DM Portfolio Volatility (Annualized):** {dm_volatility:.2f} bps")
        st.write(f"**EM Portfolio Volatility (Annualized):** {em_volatility:.2f} bps")

        # Compute VaR and cVaR at 95% and 99% Confidence Levels

        # Use the selected lookback period for VaR calculations
        price_returns_var = adjusted_price_returns.tail(var_lookback_days)

        # Aggregate positions per instrument
        positions_per_instrument = expanded_positions_vector.groupby('Instrument').sum()

        # Ensure all instruments in positions_per_instrument are present in price_returns_var
        available_instruments = positions_per_instrument.index.intersection(price_returns_var.columns)
        positions_per_instrument = positions_per_instrument.loc[available_instruments]
        price_returns_var = price_returns_var[available_instruments]

        # Compute portfolio returns
        portfolio_returns = price_returns_var.dot(positions_per_instrument) * 100  # Convert returns to bps

        # Daily VaR
        VaR_95_daily = -np.percentile(portfolio_returns, 5)
        VaR_99_daily = -np.percentile(portfolio_returns, 1)

        # cVaR calculations
        cVaR_95_daily = -portfolio_returns[portfolio_returns <= -VaR_95_daily].mean()
        cVaR_99_daily = -portfolio_returns[portfolio_returns <= -VaR_99_daily].mean()

        # Display VaR and cVaR metrics
        st.subheader('Value at Risk (VaR) and Conditional VaR (cVaR)')
        st.write(f"**Daily VaR at 95% confidence:** {VaR_95_daily:.2f} bps")
        st.write(f"**Daily cVaR at 95% confidence:** {cVaR_95_daily:.2f} bps")
        st.write(f"**Daily VaR at 99% confidence:** {VaR_99_daily:.2f} bps")
        st.write(f"**Daily cVaR at 99% confidence:** {cVaR_99_daily:.2f} bps")

        # Compute Portfolio Sensitivity to US 10Y Rates

        # Use returns of sensitivity_rate as independent variable
        if sensitivity_rate in price_returns_var.columns:
            us_10y_returns = price_returns_var[sensitivity_rate] * 100  # Convert to bps

            # Align the portfolio returns and US 10Y returns
            common_dates = portfolio_returns.index.intersection(us_10y_returns.index)
            portfolio_returns_aligned = portfolio_returns.loc[common_dates]
            us_10y_returns_aligned = us_10y_returns.loc[common_dates]

            # Check if there are overlapping dates
            if portfolio_returns_aligned.empty or us_10y_returns_aligned.empty:
                st.warning("Insufficient overlapping data between portfolio returns and US 10Y Future returns for sensitivity analysis.")
            else:
                # Perform regression to find beta
                covariance = np.cov(portfolio_returns_aligned, us_10y_returns_aligned)[0, 1]
                variance = np.var(us_10y_returns_aligned)
                beta = covariance / variance if variance != 0 else 0

                # Determine sensitivity direction
                if beta > 0:
                    beta_direction = 'positive'
                    yield_movement = 'falling'
                    pl_direction = 'gain'
                elif beta < 0:
                    beta_direction = 'negative'
                    yield_movement = 'rising'
                    pl_direction = 'gain'
                    beta = -beta  # Take absolute value for clarity
                else:
                    beta_direction = 'zero'
                    yield_movement = 'no change'
                    pl_direction = 'no change'

                # Sensitivity to a 1 bps move in US 10Y yields
                sensitivity = beta * 1  # Sensitivity per 1 bps move

                st.subheader('Portfolio Sensitivity to US 10Y Yields')
                st.write(f"**Portfolio Beta to US 10Y Yields:** {beta:.4f} ({beta_direction} beta)")
                st.write(f"The portfolio is expected to {pl_direction} when US 10Y yields are {yield_movement}.")

                # Expected P&L for a 1 bps move in US 10Y yields
                if beta_direction == 'positive':
                    expected_pl = sensitivity
                    st.write(f"**Expected P&L for a 1 bps fall in US 10Y yields:** Gain of {expected_pl:.4f} bps")
                elif beta_direction == 'negative':
                    expected_pl = sensitivity
                    st.write(f"**Expected P&L for a 1 bps rise in US 10Y yields:** Gain of {expected_pl:.4f} bps")
                else:
                    st.write("**Expected P&L for a 1 bps move in US 10Y yields:** No expected change")
        else:
            st.warning(f"'{sensitivity_rate}' data not available for sensitivity analysis.")
            st.write("**Available Columns:**")
            st.write(price_returns_var.columns.tolist())

        # Display Portfolio Volatility
        st.subheader('Total Portfolio Volatility')
        st.write(f"**Total Portfolio Volatility (Annualized):** {portfolio_volatility:.2f} bps")

    else:
        st.info('Click "Run Risk Attribution" to calculate risk metrics.')

if __name__ == '__main__':
    main()