import pandas as pd
import yfinance as yf
import performance_assessment as pa

tickers = {

    'Global Equity': 'VT', # On a changé ca
    'US Core Bonds': 'SCHZ',
    'US Short-Term Bonds': 'IGSB',
    'US Long-Term Bonds': 'TLT',


    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'World Equity': 'VEU',
    'EM Equity': 'IEMG',
    'US IG/HY Bonds': 'IUSB',
    'US Core IG': 'SPAB',
    'REITS': 'SCHH',
    'Gold': 'IAU',
    'Cash': 'BIL',
    'Raw Materials': 'DBC',
    'Green Energy': 'ICLN',
    'Simple Innovation': 'ROBO',
    'Private equity': 'PSP'
}

factor_etf_tickers = {
    'US Value': 'VBR',
    'US Momentum': 'MTUM',
    'US Growth': 'VUG',
    'US Dividend Yield': 'VYM',
    'US Quality': 'QVAL',
    'US Size': 'IJS',
    'US Technology': 'IYW'
}
# Download Asset Class Data

assets_data = pd.DataFrame()

for ticker in tickers:
    tmp_data = yf.download(tickers=tickers.get(ticker), auto_adjust=False)
    assets_data = pd.concat((assets_data, tmp_data['Adj Close']), axis=1, join='outer')

assets_data = assets_data.loc[:'2024', :]
assets_data = assets_data.dropna(how='any')
assets_data.columns = tickers.keys()

# Download US Factor ETFs Data
factor_etf_data = pd.DataFrame()
for ticker in factor_etf_tickers:
    tmp_data = yf.download(tickers=factor_etf_tickers.get(ticker), auto_adjust=False)
    factor_etf_data = pd.concat((factor_etf_data, tmp_data['Adj Close']), axis=1, join='outer')

factor_etf_data = factor_etf_data.loc[:'2024', :]
factor_etf_data = factor_etf_data.dropna(how='any')
factor_etf_data.columns = factor_etf_tickers.keys()

# Align all Timeseries
common_time = assets_data.index.intersection(factor_etf_data.index)
assets_data = assets_data.loc[common_time, :]
factor_etf_data = factor_etf_data.loc[common_time, :]

assets_data_raw = assets_data.copy()
# Renormalize all the series so they start at 100
assets_data = pa.renormalize_prices(assets_data)
factor_etf_data = pa.renormalize_prices(factor_etf_data)
data_excel_file = 'ETF Data Download.xlsx'
with pd.ExcelWriter(data_excel_file, engine='xlsxwriter') as writer:
    assets_data.to_excel(writer, sheet_name='Assets')
    factor_etf_data.to_excel(writer, sheet_name='US Factors')


# Create a Global 60-40 benchmark
bench_columns = ['Global Equity', 'US Core Bonds', 'US Short-Term Bonds', 'US Long-Term Bonds']
bench_component_data = assets_data[bench_columns].copy()
bench_component_data = pa.renormalize_prices(bench_component_data)
custom_60_40_benchmark_fixed_weights = pd.Series([0.6, 0.25, 0.10, 0.05], index=bench_columns)
custom_60_40_benchmark, custom_60_40_benchmark_weights = pa.fix_mix_portfolio_construction(custom_60_40_benchmark_fixed_weights, bench_component_data, rebal_frequency='Q')

# Create a Custom Factor tilted US Equity Portfolio
factor_columns = ['US Value', 'US Quality', 'US Size']
factor_component_data = factor_etf_data[factor_columns].copy()
factor_component_data = pa.renormalize_prices(factor_component_data)
factor_fix_mix_weights = pd.Series([0.3, 0.3, 0.4], index=factor_columns)
custom_us_factor_port, custom_us_factor_port_weights = pa.fix_mix_portfolio_construction(factor_fix_mix_weights, factor_component_data, rebal_frequency='Q')

# Create a Glide Path Global Portfolio mixing various asset classes
final_port_columns = ['World Equity','EM Equity','US IG/HY Bonds','US Core IG','Cash','Gold','Raw Materials','REITS','Green Energy','Simple Innovation','Private equity','Bitcoin','Ethereum']
final_port_comp_data = assets_data[final_port_columns].copy()
final_port_comp_data = pd.concat((custom_us_factor_port.rename('US Factor'), final_port_comp_data), axis=1)
final_port_comp_data = pa.renormalize_prices(final_port_comp_data)
starting_weights = pd.Series( [0.25, 0.05, 0.087, 0.05, 0.05, 0.05, 0.05, 0.087, 0.05, 0.087, 0.087, 0.087,0.0075,0.0075], index=final_port_comp_data.columns)
terminal_weights = pd.Series( [0.22, 0.07, 0.073, 0.07, 0.07, 0.05, 0.07, 0.073, 0.07, 0.073, 0.073, 0.073,0.0075,0.0075] , index=final_port_comp_data.columns)
print(sum(starting_weights))
glide_weights_annual = pa.glide_path_weights(starting_weights, terminal_weights, final_port_comp_data, rebal_frequency='A')
quarterly_data= pa.custom_resample(final_port_comp_data, freq='Q')
glide_weights_quarterly = glide_weights_annual.reindex(quarterly_data.index, method='ffill')
glided_portfolio, glided_portfolio_weights = pa.changing_weights_portfolio_construction(glide_weights_quarterly, final_port_comp_data)
# Compute Performance analysis
combined_portfolio = pd.concat((custom_60_40_benchmark.rename('Benchmark'),final_port_comp_data, glided_portfolio.rename('Final Portfolio')), axis=1)
performance_analysis = pa.compute_performance_stats(combined_portfolio, risk_free_price=assets_data['Cash'])
abs_drawdown_series, _ = pa.compute_drawdown(combined_portfolio)
rel_drawdown_series, _ = pa.compute_drawdown(combined_portfolio.div(combined_portfolio['Benchmark'], axis=0))

output_file = 'Final Output.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    combined_portfolio.to_excel(writer, sheet_name='All Data')
    performance_analysis.to_excel(writer, sheet_name='Performance Analysis')
    abs_drawdown_series.to_excel(writer, sheet_name='Drawdown')
    rel_drawdown_series.to_excel(writer, sheet_name='Relative Drawdown')

# Compute 3-Year Rolling performance analysis
rolling_performances = pa.rolling_performance(combined_portfolio, rolling_window_years=3, risk_free_price=assets_data['Cash'])
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    rolling_performances.to_excel(writer, sheet_name='3Y Rolling Perf')
# Compute Style analysis (Factor Regression)


reg_output = pa.regression_analysis(combined_portfolio['US Factor'].to_frame())
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    reg_output.to_excel(writer, sheet_name='Regression Output')
# Compute Correlation analysis


correlation = combined_portfolio.resample('W-Fri').last().pct_change().corr()
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:

    correlation.to_excel(writer, sheet_name='Correlation')


portfolio_values = final_port_comp_data.copy()
output_file_values = 'Valeurs_Portefeuille_Final.xlsx'

with pd.ExcelWriter(output_file_values, engine='xlsxwriter') as writer:
    portfolio_values.to_excel(writer, sheet_name='Final Portfolio Values')

portfolio_values.to_excel(writer, sheet_name='Final Portfolio Values')



final_port_raw_prices = assets_data_raw[final_port_columns].copy()

output_file_raw = 'Prix_Nodzdzdzfeuille_Final.xlsx'
with pd.ExcelWriter(output_file_raw, engine='xlsxwriter') as writer:
    final_port_raw_prices.to_excel(writer, sheet_name='Final Portfolio Raw Prices')

