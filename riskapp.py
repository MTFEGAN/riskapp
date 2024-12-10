import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Caching data loading and processing functions to improve performance
@st.cache_data(show_spinner=False)
def load_historical_data(excel_file):
    """
    Load historical data from an Excel file, combining all sheets into a single DataFrame.
    Drops weekend datapoints.
    """
    try:
        # Load Excel file
        excel = pd.ExcelFile(excel_file)
        # Read all sheets and combine them
        df_list = []
        for sheet in excel.sheet_names:
            df_sheet = excel.parse(sheet_name=sheet, index_col='date', parse_dates=True)
            df_list.append(df_sheet)
        df = pd.concat(df_list, axis=1)

        # Ensure the index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # Drop weekends (Saturday=5, Sunday=6)
        df = df[~df.index.dayofweek.isin([5, 6])]

        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_data(show_spinner=False)
def process_yields(df):
    """
    Adjust yields for specific instruments if they exist.
    """
    # Adjust yields for AU 3Y and 10Y futures if they exist
    if 'AU 3Y Future' in df.columns:
        df['AU 3Y Future'] = 100 - df['AU 3Y Future']
    if 'AU 10Y Future' in df.columns:
        df['AU 10Y Future'] = 100 - df['AU 10Y Future']
    return df

@st.cache_data(show_spinner=False)
def calculate_returns(df):
    """
    Calculate daily yield changes (returns) and adjust for price movements.
    """
    # Calculate daily yield changes (returns)
    returns = df.diff().dropna()
    # Correct the sign of returns to reflect price changes (inverse relationship)
    price_returns = returns * -1
    return price_returns

@st.cache_data(show_spinner=False)
def adjust_time_zones(price_returns, instrument_country):
    """
    Adjust returns for time zone differences by shifting certain instruments.
    """
    # Countries that do not require lagging
    non_lag_countries = ['JP', 'AU', 'SK', 'CH']
    # Map instruments to their respective countries
    instrument_countries = pd.Series(
        [instrument_country.get(instr, 'Other') for instr in price_returns.columns],
        index=price_returns.columns
    )

    # Instruments to lag (exclude non-lag countries)
    instruments_to_lag = instrument_countries[~instrument_countries.isin(non_lag_countries)].index.tolist()

    # Create a copy for adjustment
    adjusted_price_returns = price_returns.copy()

    if instruments_to_lag:
        # Shift these instruments by -1 to align with previous day
        adjusted_price_returns[instruments_to_lag] = adjusted_price_returns[instruments_to_lag].shift(-1)

    # Drop rows with NaN values resulting from the shift
    adjusted_price_returns = adjusted_price_returns.dropna()

    return adjusted_price_returns

@st.cache_data(show_spinner=False)
def calculate_volatilities(adjusted_price_returns, lookback_days):
    """
    Calculate annualized volatilities in basis points using the specified lookback period.
    """
    # Select the lookback period
    price_returns_vol = adjusted_price_returns.tail(lookback_days)
    # Calculate annualized volatilities in basis points
    volatilities = price_returns_vol.std() * np.sqrt(252) * 100  # Annualized volatility in bps
    return volatilities

@st.cache_data(show_spinner=False)
def calculate_covariance_matrix(adjusted_price_returns, lookback_days):
    """
    Calculate the annualized covariance matrix in basis points squared using the specified lookback period.
    """
    # Select the lookback period
    price_returns_cov = adjusted_price_returns.tail(lookback_days)
    # Calculate the covariance matrix (annualized) in bps^2
    covariance_matrix = price_returns_cov.cov() * 252 * 10000  # Multiply by 100^2 to convert to bps^2
    return covariance_matrix

def main():
    # Set page configuration to use the wide layout
    st.set_page_config(page_title="📈 Fixed Income Portfolio Risk Attribution", layout="wide")
    st.title('📈 Fixed Income Portfolio Risk Attribution')

    # Debugging statement to confirm app initialization
    st.write("App initialized successfully.")

    # Define a consistent color theme (can be customized)
    primary_color = "#1f77b4"  # Example color

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

    # Sidebar for Sensitivity Rate Configuration
    st.sidebar.header("🔍 Sensitivity Rate Configuration")
    excel_file = 'historical_data.xlsx'

    # Check if the Excel file exists
    if os.path.exists(excel_file):
        raw_df = load_historical_data(excel_file)
        if raw_df is not None:
            available_columns = raw_df.columns.tolist()
            if 'US 10Y Future' in available_columns:
                default_index = available_columns.index('US 10Y Future')
            else:
                default_index = 0  # Default to the first column if 'US 10Y Future' is not present
            sensitivity_rate = st.sidebar.selectbox(
                'Select the rate instrument for sensitivity analysis:',
                options=available_columns,
                index=default_index
            )
        else:
            sensitivity_rate = 'US 10Y Future'  # Default value if data loading failed
    else:
        st.sidebar.error(f"❌ Excel file '{excel_file}' not found. Please ensure it is in the app directory.")
        st.stop()

    # Mapping Ticker to Instrument Name
    tickers_data = OrderedDict(zip(instruments_data['Ticker'], instruments_data['Instrument Name']))

    # Mapping Instrument Name to Portfolio
    instrument_portfolio = dict(zip(instruments_data['Instrument Name'], instruments_data['Portfolio']))

    # Mapping Instrument Name to Country
    # (This should be comprehensive based on your data; adjust as needed)
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

    # Define JavaScript code for conditional formatting in AgGrid
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

    # Create tabs for better organization
    tabs = st.tabs(["📊 Risk Attribution", "📂 Input Positions", "⚙️ Settings"])

    with tabs[1]:
        st.header("🔄 Input Positions")
        st.write("Enter your positions for the Developed Markets (DM) and Emerging Markets (EM) portfolios below.")

        # Use columns to separate DM and EM inputs side by side
        dm_col, em_col = st.columns(2)

        with dm_col:
            st.subheader('📈 DM Portfolio Positions')
            gb_dm = GridOptionsBuilder.from_dataframe(default_positions_dm)
            gb_dm.configure_columns(['Outright', 'Curve', 'Spread'], editable=True, cellStyle=cell_style_jscode)
            gb_dm.configure_column('Instrument', editable=False)
            gb_dm.configure_pagination(enabled=True, paginationPageSize=20)  # Enable pagination
            grid_options_dm = gb_dm.build()
            grid_response_dm = AgGrid(
                default_positions_dm,
                gridOptions=grid_options_dm,
                height=400,
                width='100%',
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
                reload_data=True
            )
            positions_data_dm = grid_response_dm['data']

        with em_col:
            st.subheader('🌍 EM Portfolio Positions')
            gb_em = GridOptionsBuilder.from_dataframe(default_positions_em)
            gb_em.configure_columns(['Outright', 'Curve', 'Spread'], editable=True, cellStyle=cell_style_jscode)
            gb_em.configure_column('Instrument', editable=False)
            gb_em.configure_pagination(enabled=True, paginationPageSize=20)  # Enable pagination
            grid_options_em = gb_em.build()
            grid_response_em = AgGrid(
                default_positions_em,
                gridOptions=grid_options_em,
                height=400,
                width='100%',
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
                reload_data=True
            )
            positions_data_em = grid_response_em['data']

    with tabs[2]:
        st.header("⚙️ Configuration Settings")
        st.write("Adjust the lookback periods for volatility and VaR calculations.")

        # Dropdown for volatility lookback period
        volatility_period_options = {
            '📅 1 month': 21,
            '📆 3 months': 63,
            '📅 6 months': 126,
            '🗓️ 1 year': 252,
            '📅 3 years': 756,
            '📆 5 years': 1260,
            '📅 10 years': 2520
        }
        volatility_period = st.selectbox('Select lookback period for volatility calculations:', list(volatility_period_options.keys()), index=3)
        volatility_lookback_days = volatility_period_options[volatility_period]

        # Dropdown for VaR calculation period
        var_period_options = {
            '📆 5 years': 1260,
            '📆 10 years': 2520,
            '📆 15 years': 3780
        }
        var_period = st.selectbox('Select lookback period for VaR calculations:', list(var_period_options.keys()), index=1)
        var_lookback_days = var_period_options[var_period]

    with tabs[0]:
        st.header("📊 Risk Attribution Results")

        # Step 3: Run Risk Attribution when button is clicked
        if st.button('🚀 Run Risk Attribution'):
            with st.spinner('Calculating risk attribution... This may take a few moments.'):
                # Step 4: Load Historical Data from the Excel File
                if not os.path.exists(excel_file):
                    st.error(f"❌ Excel file '{excel_file}' not found. Please ensure it is in the app directory.")
                    st.stop()

                # Load and process historical data
                df = load_historical_data(excel_file)
                if df is None:
                    st.error("Failed to load data.")
                    st.stop()

                df = process_yields(df)
                price_returns = calculate_returns(df)
                adjusted_price_returns = adjust_time_zones(price_returns, instrument_country)

                # Check if adjusted_price_returns is empty
                if adjusted_price_returns.empty:
                    st.warning("⚠️ No data available after adjusting for time zones.")
                    st.stop()

                # Calculate volatilities and covariance matrix
                volatilities = calculate_volatilities(adjusted_price_returns, volatility_lookback_days)
                covariance_matrix = calculate_covariance_matrix(adjusted_price_returns, var_lookback_days)

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
                    st.warning("⚠️ No positions entered. Please enter positions and try again.")
                    st.stop()

                # Create expanded positions vector
                expanded_positions_vector = expanded_positions_data.set_index(['Instrument', 'Position Type'])['Position']

                # Ensure that the sensitivity rate is included even if the user has no position in it
                if sensitivity_rate not in expanded_positions_vector.index.get_level_values('Instrument'):
                    # Add the sensitivity_rate with zero position
                    zero_position = pd.Series(
                        0.0, 
                        index=pd.MultiIndex.from_tuples([(sensitivity_rate, 'Outright')], names=['Instrument', 'Position Type'])
                    )
                    expanded_positions_vector = pd.concat([expanded_positions_vector, zero_position])

                # Create an empty DataFrame for the expanded covariance matrix
                expanded_index = expanded_positions_vector.index
                expanded_cov_matrix = pd.DataFrame(index=expanded_index, columns=expanded_index)

                # Populate the expanded covariance matrix (vectorized)
                instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()
                
                # Check if all instruments are present in the covariance matrix
                missing_instruments = [instr for instr in instruments if instr not in covariance_matrix.index]
                if missing_instruments:
                    st.warning(f"⚠️ The following instruments are missing from the covariance matrix and will be excluded from calculations: {missing_instruments}")
                    # Remove missing instruments from positions_vector and expanded_cov_matrix
                    # Drop any positions corresponding to missing instruments
                    expanded_positions_vector = expanded_positions_vector.drop(
                        labels=[
                            (instr, pos_type) 
                            for instr in missing_instruments 
                            for pos_type in ['Outright', 'Curve', 'Spread'] 
                            if (instr, pos_type) in expanded_positions_vector.index
                        ], 
                        errors='ignore'
                    )
                    expanded_index = expanded_positions_vector.index
                    expanded_cov_matrix = pd.DataFrame(index=expanded_index, columns=expanded_index)
                    instruments = expanded_positions_vector.index.get_level_values('Instrument').unique()

                # Re-extract covariance_submatrix with available instruments
                covariance_submatrix = covariance_matrix.loc[instruments, instruments]

                # Assign covariance values to the expanded covariance matrix
                for pos1 in expanded_index:
                    instr1 = pos1[0]
                    if instr1 in covariance_submatrix.index:
                        expanded_cov_matrix.loc[pos1, :] = covariance_submatrix.loc[instr1, instruments]
                    else:
                        expanded_cov_matrix.loc[pos1, :] = 0.0  # Assign zero if instrument missing

                expanded_cov_matrix = expanded_cov_matrix.astype(float)

                # Compute Portfolio Volatility
                portfolio_variance = np.dot(expanded_positions_vector.values, np.dot(expanded_cov_matrix.values, expanded_positions_vector.values))
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
                
                # Ensure 'Instrument' and 'Position Type' are set correctly
                if not {'Instrument', 'Position Type'}.issubset(risk_contributions.columns):
                    st.error("⚠️ Positions data is missing 'Instrument' or 'Position Type' columns.")
                    st.stop()

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

                # **Exclude Rows with Zero or NaN Positions**
                risk_contributions_formatted = risk_contributions_formatted[
                    risk_contributions_formatted['Position'].notna() & (risk_contributions_formatted['Position'] != 0)
                ]

                # Compute VaR and cVaR before displaying metrics
                # Use the selected lookback period for VaR calculations
                price_returns_var = adjusted_price_returns.tail(var_lookback_days)

                # Aggregate positions per instrument
                positions_per_instrument = expanded_positions_vector.groupby('Instrument').sum()

                # Ensure all instruments are present in the covariance matrix
                available_instruments_var = positions_per_instrument.index.intersection(price_returns_var.columns)
                positions_per_instrument = positions_per_instrument.loc[available_instruments_var]
                price_returns_var = price_returns_var[available_instruments_var]

                # Compute portfolio returns
                portfolio_returns = price_returns_var.dot(positions_per_instrument) * 100  # Convert returns to bps

                # Compute VaR and cVaR
                VaR_95_daily = -np.percentile(portfolio_returns, 5)
                VaR_99_daily = -np.percentile(portfolio_returns, 1)

                cVaR_95_daily = -portfolio_returns[portfolio_returns <= -VaR_95_daily].mean()
                cVaR_99_daily = -portfolio_returns[portfolio_returns <= -VaR_99_daily].mean()

                # Add bar charts to 'Percent Contribution (%)' column using Plotly
                fig_risk_contributions = px.bar(
                    risk_contributions_formatted,
                    x='Instrument',
                    y='Percent Contribution (%)',
                    color='Portfolio',
                    title='Risk Contributions by Instrument',
                    labels={'Percent Contribution (%)': 'Percent Contribution (%)'},
                    hover_data=['Position', 'Contribution to Volatility (bps)'],
                    height=500
                )
                fig_risk_contributions.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

                # Render the bar chart
                st.plotly_chart(fig_risk_contributions, use_container_width=True)

                # Compute risk contributions by Portfolio (DM and EM)
                risk_contributions_by_portfolio = risk_contributions_formatted.groupby('Portfolio').agg({
                    'Contribution to Volatility (bps)': 'sum'
                })
                risk_contributions_by_portfolio['Percent Contribution (%)'] = (risk_contributions_by_portfolio['Contribution to Volatility (bps)'] / portfolio_volatility) * 100

                # Create a pie chart for risk contributions by portfolio
                fig_portfolio_risk = px.pie(
                    risk_contributions_by_portfolio.reset_index(),
                    names='Portfolio',
                    values='Contribution to Volatility (bps)',
                    title='Risk Contributions by Portfolio',
                    hole=0.4
                )
                fig_portfolio_risk.update_traces(textposition='inside', textinfo='percent+label')
                fig_portfolio_risk.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

                # Render the pie chart
                st.plotly_chart(fig_portfolio_risk, use_container_width=True)

                # Display key metrics using Streamlit's metric component
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

                metrics_col1.metric(label="📊 Total Portfolio Volatility", value=f"{portfolio_volatility:.2f} bps")
                metrics_col2.metric(label="📉 Daily VaR (95%)", value=f"{VaR_95_daily:.2f} bps")
                metrics_col3.metric(label="📉 Daily VaR (99%)", value=f"{VaR_99_daily:.2f} bps")
                metrics_col4.metric(label="📈 Daily cVaR (95%)", value=f"{cVaR_95_daily:.2f} bps")

                # Display VaR and cVaR metrics in detail
                st.subheader('📈 Value at Risk (VaR) and Conditional VaR (cVaR)')
                st.write(f"**Daily VaR at 95% confidence:** {VaR_95_daily:.2f} bps")
                st.write(f"**Daily cVaR at 95% confidence:** {cVaR_95_daily:.2f} bps")
                st.write(f"**Daily VaR at 99% confidence:** {VaR_99_daily:.2f} bps")
                st.write(f"**Daily cVaR at 99% confidence:** {cVaR_99_daily:.2f} bps")

                # Compute Portfolio Sensitivity to Sensitivity Rate

                # Use returns of sensitivity_rate as independent variable
                if sensitivity_rate in price_returns_var.columns:
                    sensitivity_returns = price_returns_var[sensitivity_rate] * 100  # Convert to bps

                    # Align the portfolio returns and sensitivity rate returns
                    common_dates = portfolio_returns.index.intersection(sensitivity_returns.index)
                    portfolio_returns_aligned = portfolio_returns.loc[common_dates]
                    sensitivity_returns_aligned = sensitivity_returns.loc[common_dates]

                    # Check if there are overlapping dates
                    if portfolio_returns_aligned.empty or sensitivity_returns_aligned.empty:
                        st.warning("⚠️ Insufficient overlapping data between portfolio returns and the selected sensitivity rate returns for sensitivity analysis.")
                    else:
                        # Perform regression to find beta
                        covariance = np.cov(portfolio_returns_aligned, sensitivity_returns_aligned)[0, 1]
                        variance = np.var(sensitivity_returns_aligned)
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

                        # Sensitivity to a 1 bps move in sensitivity rate yields
                        sensitivity = beta * 1  # Sensitivity per 1 bps move

                        st.subheader(f'📉 Portfolio Sensitivity to {sensitivity_rate} Yields')
                        st.write(f"**Portfolio Beta to {sensitivity_rate} Yields:** {beta:.4f} ({beta_direction} beta)")
                        st.write(f"The portfolio is expected to {pl_direction} when {sensitivity_rate} yields are {yield_movement}.")

                        # Expected P&L for a 1 bps move in sensitivity rate yields
                        if beta_direction == 'positive':
                            expected_pl = sensitivity
                            st.write(f"**Expected P&L for a 1 bps fall in {sensitivity_rate} yields:** Gain of {expected_pl:.4f} bps")
                        elif beta_direction == 'negative':
                            expected_pl = sensitivity
                            st.write(f"**Expected P&L for a 1 bps rise in {sensitivity_rate} yields:** Gain of {expected_pl:.4f} bps")
                        else:
                            st.write(f"**Expected P&L for a 1 bps move in {sensitivity_rate} yields:** No expected change")

                        # **Visualization: Scatter Plot with Regression Line**
                        fig_sensitivity = px.scatter(
                            x=sensitivity_returns_aligned,
                            y=portfolio_returns_aligned,
                            trendline="ols",
                            labels={
                                'x': f'{sensitivity_rate} Returns (bps)',
                                'y': 'Portfolio Returns (bps)'
                            },
                            title=f'Relationship between Portfolio Returns and {sensitivity_rate} Returns',
                            height=500
                        )
                        fig_sensitivity.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        
                        # Extract trendline results
                        results = px.get_trendline_results(fig_sensitivity)
                        if not results.empty:
                            slope = results.iloc[0]["px_fit_results"].params[1]
                            intercept = results.iloc[0]["px_fit_results"].params[0]
                            fig_sensitivity.add_annotation(
                                x=0.05,
                                y=0.95,
                                xref="paper",
                                yref="paper",
                                text=f"Beta: {beta:.4f}",
                                showarrow=False,
                                font=dict(size=12, color="black")
                            )

                        # Render the scatter plot
                        st.plotly_chart(fig_sensitivity, use_container_width=True)
                else:
                    st.warning(f"⚠️ '{sensitivity_rate}' data not available for sensitivity analysis.")
                    st.write("**Available Columns:**")
                    st.write(price_returns_var.columns.tolist())

                # **Display Risk Contributions Table**
                st.subheader('📄 Detailed Risk Contributions by Instrument')
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

                # **Download Button for Risk Contributions**
                csv = risk_contributions_formatted.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Risk Contributions as CSV",
                    data=csv,
                    file_name='risk_contributions.csv',
                    mime='text/csv',
                )

if __name__ == '__main__':
    main()





