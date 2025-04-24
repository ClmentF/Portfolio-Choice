import pandas as pd
import numpy as np
import pandas_datareader.data as web  
from pandas_datareader.famafrench import get_available_datasets
import statsmodels.api as sm


def determine_frequency(ts):
    """
    Determines the frequency of a pandas Series or DataFrame with a DateTimeIndex.

    Parameters:
    ts (pd.Series or pd.DataFrame): Time series data with a DateTimeIndex.

    Returns:
    str: Frequency string (e.g., 'D' for daily, 'H' for hourly, etc.), or None if undetermined.
    """
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame.")
    
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DateTimeIndex.")
    

    num_days = ts.index.diff().min().days
    if num_days == 1:
        inferred_freq = 'D'
    elif (num_days >= 5) & (num_days<= 7):
        inferred_freq = 'W'
    elif (num_days >= 28) & (num_days<= 31):
        inferred_freq = 'M'
    elif (num_days >= 65) & (num_days<= 95):
        inferred_freq = 'Q'
    elif (num_days >= 125) & (num_days<= 190):
        inferred_freq = 'S'
    elif (num_days >= 250) & (num_days<= 366):
        inferred_freq = 'A'
    return inferred_freq



def returns_to_prices(returns, initial_price=100, log_returns=False):
    """
    Converts a series or dataframe of returns into price values, ensuring correct first return handling.

    Parameters:
    - returns (pd.Series or pd.DataFrame): A pandas Series or DataFrame of returns with a DateTimeIndex.
    - initial_price (float or array-like): The starting price(s). Can be a single float or an array-like matching the number of columns.
    - log_returns (bool): If True, assumes returns are log returns. Otherwise, assumes simple returns.

    Returns:
    - pd.Series or pd.DataFrame: A series or dataframe of price values.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame.")
    
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DateTimeIndex.")

    returns = returns.copy()

    # Ensure first return is 0 if it's NaN
    returns.iloc[0] = 0
    
    # If the first return is not 0 or NaN, prepend a 0 return
    if not (returns.iloc[0] == 0).all():
        inferred_freq = determine_frequency(returns)
        if inferred_freq is None:
            raise ValueError("Could not infer frequency from index. Ensure it's a valid time series.")
        
        new_index = returns.index[0] - pd.to_timedelta(1, unit=inferred_freq)
        if isinstance(returns, pd.Series):
            returns = pd.concat([pd.Series([0], index=[new_index]), returns])
        else:  # DataFrame case
            zero_row = pd.DataFrame([[0] * returns.shape[1]], index=[new_index], columns=returns.columns)
            returns = pd.concat([zero_row, returns])

    # Set initial price
    if isinstance(initial_price, (int, float)):
        initial_price = np.full(returns.shape[1] if isinstance(returns, pd.DataFrame) else 1, initial_price)

    # Convert returns to prices
    if log_returns:
        prices = initial_price * np.exp(returns.cumsum())
    else:
        prices = initial_price * (1 + returns).cumprod()

    return prices


def prices_to_returns(prices, log_returns=False):
    """
    Converts a series or dataframe of prices into returns, ensuring the first return is zero.

    Parameters:
    - prices (pd.Series or pd.DataFrame): A pandas Series or DataFrame of price values.
    - log_returns (bool): If True, computes log returns. Otherwise, computes simple returns.

    Returns:
    - pd.Series or pd.DataFrame: A series or dataframe of return values, with the first value set to zero.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame.")

    if log_returns:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    # Ensure the first return is zero
    returns.iloc[0] = 0

    return returns


def renormalize_prices(prices, start_value=100):
    """
    Renormalizes a series or dataframe of prices to start from a specified value.

    Parameters:
    - prices (pd.Series or pd.DataFrame): A pandas Series or DataFrame of price values.
    - start_value (float): The new starting price value (default is 100).

    Returns:
    - pd.Series or pd.DataFrame: Renormalized prices.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame.")

    return (prices / prices.iloc[0]) * start_value


def get_annualization_factor(frequency):
    """
    Returns the annualization factor based on the inferred frequency from pandas.

    Parameters:
    - frequency (str): The inferred frequency from pd.infer_freq (e.g., 'D', 'W', 'M', 'Q', 'S', 'A').

    Returns:
    - int: The corresponding annualization factor.
    """
    frequency = frequency.split('-')[0]
    frequency_map = {
        'D': 252,   # Daily
        'B': 252,   # Business Daily
        'W': 52,    # Weekly
        'M': 12,    # Monthly
        'Q': 4,     # Quarterly
        'S': 2,     # Semi-Annual
        'A': 1,     # Annual
        'Y': 1      # Yearly (Alias for 'A')
    }

    if frequency not in frequency_map:
        raise ValueError(f"Unknown frequency '{frequency}'. Ensure it is a valid `pd.infer_freq` output.")

    return frequency_map[frequency]


def compute_drawdown(prices):
    """
    Computes the drawdown time series and the maximum drawdown for a series or dataframe of prices.

    Parameters:
    - prices (pd.Series or pd.DataFrame): A pandas Series or DataFrame of price values.

    Returns:
    - drawdowns (pd.Series or pd.DataFrame): A series or dataframe of drawdown values.
    - max_drawdown (float): The maximum drawdown value.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame.")
    
    # Ensure we're working with a copy of the data to avoid modifying the original
    prices = prices.copy()

    # Calculate cumulative maximum
    cumulative_max = prices.cummax()

    # Calculate drawdown
    drawdowns = (prices - cumulative_max) / cumulative_max

    # Calculate max drawdown
    max_drawdown = -drawdowns.min()

    return drawdowns, max_drawdown


def compute_performance_stats(prices, risk_free_price=None, benchmark_col=0):
    """
    Computes performance metrics from portfolio price data, including cumulative returns, annualized returns,
    volatility, Sharpe ratio, Sortino ratio, maximum drawdown and the corresponding relative statistics.

    Parameters:
    - prices (pd.Series or pd.DataFrame): Portfolio price data.
    - risk_free_price (pd.Series or pd.DataFrame, optional): Risk-free rate as price time series.
    - benchmark_col (int): The column number of the prices dataframe that is to be used as benchmark for relative performance stats.

    Returns:
    - pd.DataFrame: A dataframe containing the performance metrics for each portfolio.
    """
    
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise ValueError("Input prices must be a pandas Series or DataFrame.")
    
    assert isinstance(benchmark_col, int) and benchmark_col >= 0, "benchmark_col should be a positive integer"
    assert benchmark_col <= prices.shape[1], "benchmark_col should be less than the number of columns in prices"

    # Ensure the input is a copy to avoid modifying original data
    prices = prices.copy()
    if benchmark_col != 0:
        prices = pd.concat((prices.iloc[:, benchmark_col], prices.drop(prices.columns[benchmark_col], axis=1)), axis=1)
    # Calculate returns from prices
    returns = prices_to_returns(prices)

    # Handle the risk-free rate (if provided as price series)
    if risk_free_price is not None:
        if not isinstance(risk_free_price, (pd.Series, pd.DataFrame)):
            raise ValueError("Risk-free rate must be a pandas Series or DataFrame.")
        
        # Check if the index of the risk-free price matches the portfolio prices
        if not risk_free_price.index.equals(returns.index):
            raise ValueError("The index of the risk-free price time series must match the index of the portfolio prices.")
        
        # Calculate risk-free returns from the price series
        risk_free_returns = prices_to_returns(risk_free_price)
        
    else:
        risk_free_returns = pd.Series(0, index=returns.index)  # If no risk-free rate, assume 0% return
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod().iloc[-1] - 1
    cumulative_risk_free_returns = (1 + risk_free_returns).cumprod().iloc[-1] - 1
    cumulative_relative_returns = cumulative_returns - cumulative_returns.iloc[0]

    periods_per_year = get_annualization_factor(determine_frequency(prices))
    start_date = prices.index[0]
    end_date = prices.index[-1]
    num_days = (end_date - start_date).days

    # Annualized returns
    annualized_returns = (1 + cumulative_returns)**(365 / num_days) - 1
    annualised_risk_free_returns = (1 + cumulative_risk_free_returns)**(365 / num_days) - 1

    # Annualized relative returns
    annualized_relative_returns = annualized_returns - annualized_returns.iloc[0]

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Tracking error (annualized)
    excess_returns = returns.sub(returns.iloc[:, 0], axis=0)
    tracking_error = excess_returns.std()* np.sqrt(periods_per_year)

    # Sharpe ratio (adjusted with risk-free returns if risk_free_price is provided)
    sharpe_ratio = annualized_returns.sub(annualised_risk_free_returns) / volatility

    # Information ratio
    information_ratio = annualized_relative_returns / tracking_error

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = annualized_returns.sub(annualised_risk_free_returns) / downside_volatility

    # Maximum drawdown calculation
    _, max_drawdown = compute_drawdown(prices)

    # Maximum relative drawdown calculation
    _, max_relative_drawdown = compute_drawdown(prices.div(prices.iloc[:, 0], axis=0))


    # Compile results into a DataFrame
    performance_metrics = pd.DataFrame({
        'Cumulative Returns': cumulative_returns,
        'Annualized Returns': annualized_returns,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Cumulative Relative Returns': cumulative_relative_returns,
        'Annualized Relative Returns': annualized_relative_returns,
        'Tracking Error': tracking_error,
        'Information Ratio': information_ratio,
        'Maximum Relative Drawdown': max_relative_drawdown
    })

    return performance_metrics.T



def rolling_performance(prices, rolling_window_years, risk_free_price=None, benchmark_col=0):
    """
    Calculate rolling performance statistics over a rolling window in years for multiple portfolios
    against a benchmark.

    Parameters:
    - prices (pd.DataFrame): A dataframe containing portfolio prices, one column as the benchmark.
    - benchmark_col (str): The name of the benchmark column.
    - rolling_window_years (int): The rolling window in years.
    - risk_free_price (pd.Series or None): Risk-free rate as price time series (optional).
    - periods_per_year (int): The number of periods per year (e.g., 252 for daily data).

    Returns:
    - pd.DataFrame: A dataframe containing the rolling performance statistics for each portfolio.
    """
    
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise ValueError("Input prices must be a pandas Series or DataFrame.")
    
    assert isinstance(benchmark_col, int) and benchmark_col >= 0, "benchmark_col should be a positive integer"
    assert benchmark_col <= prices.shape[1], "benchmark_col should be less than the number of columns in prices"


    prices = prices.copy()
    if benchmark_col != 0:
        prices = pd.concat((prices.iloc[:, benchmark_col], prices.drop(prices.columns[benchmark_col], axis=1)), axis=1)
    # Calculate returns for portfolios and benchmark
    returns = prices_to_returns(prices)
    
    # Handle the risk-free rate (if provided as price series)
    if risk_free_price is not None:
        if not isinstance(risk_free_price, (pd.Series, pd.DataFrame)):
            raise ValueError("Risk-free rate must be a pandas Series or DataFrame.")
        
        # Check if the index of the risk-free price matches the portfolio prices
        if not risk_free_price.index.equals(returns.index):
            raise ValueError("The index of the risk-free price time series must match the index of the portfolio prices.")
        
        # Calculate risk-free returns from the price series
        risk_free_returns = prices_to_returns(risk_free_price)
        
    else:
        risk_free_returns = pd.Series(0, index=returns.index)  # If no risk-free rate, assume 0% return    
    periods_per_year = get_annualization_factor(determine_frequency(prices))

    # Define the rolling window size in periods
    window_size = rolling_window_years * periods_per_year
    if isinstance(returns, pd.DataFrame):
        benchmark_returns = returns.iloc[:, 0].copy()
    else:
        benchmark_returns = returns.copy()
    
    # Create a dictionary to hold performance stats
    performance_stats = {}


    # Calculate rolling stats
    rolling_annualized_returns = returns.rolling(window=window_size).apply(
        lambda x: (1 + x).prod() ** (periods_per_year / len(x)) - 1, raw=False
    )
    
    rolling_annualized_riskfree_returns = risk_free_returns.rolling(window=window_size).apply(
        lambda x: (1 + x).prod() ** (periods_per_year / len(x)) - 1, raw=False
    )

    rolling_volatility = returns.rolling(window=window_size).std() * np.sqrt(periods_per_year)
    
    rolling_sharpe_ratio = (rolling_annualized_returns.sub(risk_free_returns, axis=0)) / rolling_volatility
    
    # Relative Returns
    relative_returns = returns.sub(benchmark_returns, axis=0)
    rolling_relative_returns = rolling_annualized_returns.sub(rolling_annualized_returns[rolling_annualized_returns.columns[0]], axis=0)
    
    # Tracking Error
    rolling_tracking_error = relative_returns.rolling(window=window_size).std() * np.sqrt(periods_per_year)
    
    # Information Ratio
    rolling_information_ratio = rolling_relative_returns / rolling_tracking_error
    stats = ['Rolling Annualized Return', 'Rolling Volatility', 'Rolling Sharpe Ratio', 'Rolling Relative Returns', 'Rolling Tracking Error', 'Rolling Information Ratio']
    multi_index_columns = pd.MultiIndex.from_product([stats, prices.columns])
    rolling_perf = pd.DataFrame(data=None, columns = multi_index_columns)

    rolling_perf['Rolling Annualized Return'] = rolling_annualized_returns
    rolling_perf['Rolling Volatility'] = rolling_volatility
    rolling_perf['Rolling Sharpe Ratio'] = rolling_sharpe_ratio
    rolling_perf['Rolling Relative Returns'] = rolling_relative_returns
    rolling_perf['Rolling Tracking Error'] = rolling_tracking_error
    rolling_perf['Rolling Information Ratio'] = rolling_information_ratio

    return rolling_perf



def regression_analysis(prices):
    assert isinstance(prices, pd.DataFrame), 'prices must be a DataFrame'
    field_1 = 'F-F_Research_Data_5_Factors_2x3_daily'
    five_factors = web.DataReader(field_1,'famafrench',start='2000-01-01',end='2024-12-31')
    five_factors_data = five_factors[0]/100
    field_2 = 'F-F_Momentum_Factor_daily'
    mom_factor = web.DataReader(field_2,'famafrench',start='2000-01-01',end='2024-12-31')
    mom_factor_data = mom_factor[0]/100

    five_factors_idx = returns_to_prices(five_factors_data)
    mom_factors_idx = returns_to_prices(mom_factor_data)
    indep_idx = pd.concat((five_factors_idx[['Mkt-RF', 'SMB', 'HML']], mom_factors_idx, five_factors_idx[['RMW', 'CMA']]), axis=1)
    indep_idx = indep_idx.reindex(prices.index).ffill()
    indep_ret = prices_to_returns(indep_idx)
    dep_ret = prices_to_returns(prices)

    indep_ret = sm.add_constant(indep_ret)
    stats = ['Beta', 't-statistic', 'p-value', 'Adjusted R2']
    multi_index_columns = pd.MultiIndex.from_product([prices.columns, stats])
    reg_output = pd.DataFrame(data=np.nan, index=indep_ret.columns, columns=multi_index_columns)
    for col in dep_ret.columns:
        reg_results = sm.OLS(endog=dep_ret[col], exog=indep_ret).fit()
        tmp_opt = pd.concat((reg_results.params, reg_results.tvalues, reg_results.pvalues), axis=1)
        tmp_opt.columns = ['Beta', 't-statistic', 'p-value']
        tmp_opt['Adjusted R2'] = reg_results.rsquared_adj
        reg_output[col] = tmp_opt.copy()
    return reg_output


def custom_resample(df, freq):
    # This is a modified ressample function to include the first data point and 
    # include only the dates available in the original sample
    # In contrast the default resample method always returns 30th or 31st of the month as the index
    # This method is useful for rebalancing our portfolio based on market data when the daily timeseries has no data for weekends and holidays
    assert (isinstance(df , pd.DataFrame) | isinstance(df , pd.Series)) , 'df must be a dataframe or series'
    resampled = df.resample(freq).last()  # Get end-of-period values
    resampled.index = resampled.index.map(lambda d: df.index[df.index <= d][-1]) 
    first_entry = df.iloc[[0]]  # Get the very first row
    resampled = pd.concat([first_entry, resampled]).drop_duplicates().sort_index()  # Combine & sort
    return resampled


def glide_path_weights(starting_weights, terminal_weights, prices, rebal_frequency='M'):
    assert isinstance(starting_weights, pd.Series), 'starting_weights must be a series'
    assert isinstance(terminal_weights, pd.Series), 'terminal_weights must be a series'
    assert isinstance(prices, pd.DataFrame), 'prices must be a dataframe'
    assert all(starting_weights.index == prices.columns), 'the columns of prices and starting_weights must be the same'
    assert all(terminal_weights.index == prices.columns), 'the columns of prices and terminal_weights must be the same'
    assert rebal_frequency in ['M', 'Q', 'S', 'A'], 'rebal_frequency must be in [M, Q, S, A]'
    assert abs(starting_weights.sum() - 1.0) < 1e-7, 'starting_weights must sum to 1'
    assert abs(terminal_weights.sum() - 1.0) < 1e-7, 'terminal_weights must sum to 1'
    tmp_weights = prices.copy()
    tmp_rebal_weights = custom_resample(tmp_weights, rebal_frequency)
    tmp_rebal_weights[:] = 0
    delta = (terminal_weights - starting_weights)/(tmp_rebal_weights.shape[0]-1)
    indices = np.arange(len(tmp_rebal_weights)).reshape(-1, 1)
    cumulative_adjustments = indices * delta.values
    rebal_weights = starting_weights.values + cumulative_adjustments
    glided_weights = pd.DataFrame(rebal_weights, index= tmp_rebal_weights.index, columns=tmp_rebal_weights.columns)
    return glided_weights


def fix_mix_portfolio_construction(fix_weights, prices, rebal_frequency='M'):
    assert isinstance(fix_weights, pd.Series), 'fix_weights must be a series'
    assert isinstance(prices, pd.DataFrame), 'prices must be a dataframe'
    assert all(fix_weights.index == prices.columns), 'the columns of prices and fix_weights must be the same'
    assert rebal_frequency in ['M', 'Q', 'S', 'A'], 'rebal_frequency must be in [M, Q, S, A]'
    assert(fix_weights.sum() == 1), 'fix_weights must sum to 1'
    weights = pd.DataFrame(data=np.nan, index=prices.index, columns=prices.columns)
    rebal_dates = custom_resample(prices, freq=rebal_frequency).index
    returns = prices_to_returns(prices)
    for _i, idx in enumerate(prices.index):
        if idx in rebal_dates:
            weights.loc[idx, :] = fix_weights.values
        else:
            weights.loc[idx, :] = weights.iloc[_i - 1, :] * (1 + returns.loc[idx, :])
            weights.loc[idx, :] = weights.loc[idx, :] / weights.loc[idx, :].sum()
    portfolio_returns = (weights.shift(1)*returns).sum(axis=1)
    portfolio_idx = returns_to_prices(portfolio_returns)
    return (portfolio_idx, weights)


def changing_weights_portfolio_construction(rebal_weights, prices,):
    assert isinstance(rebal_weights, pd.DataFrame), 'rebal_weights must be a DataFrame'
    assert isinstance(prices, pd.DataFrame), 'prices must be a dataframe'
    assert all(rebal_weights.columns == prices.columns), 'the columns of prices and fix_weights must be the same'
    assert all (rebal_weights.index.isin(prices.index)), 'Every index of rebal_weights must be in the price index'
    assert (rebal_weights.index[0] == prices.index[0]), 'The first date must be a rebalancing date'
    assert (rebal_weights.sum(axis=1).sum() == rebal_weights.shape[0]), 'rebal_weights must sume upto 1 for every row'
    weights = pd.DataFrame(data=np.nan, index=prices.index, columns=prices.columns)
    rebal_dates = rebal_weights.index
    returns = prices_to_returns(prices)
    for _i, idx in enumerate(prices.index):
        if idx in rebal_dates:
            weights.loc[idx, :] = rebal_weights.loc[idx, :].values
        else:
            weights.loc[idx, :] = weights.iloc[_i - 1, :] * (1 + returns.loc[idx, :])
            weights.loc[idx, :] = weights.loc[idx, :] / weights.loc[idx, :].sum()
    portfolio_returns = (weights.shift(1)*returns).sum(axis=1)
    portfolio_idx = returns_to_prices(portfolio_returns)
    return (portfolio_idx, weights)


