import datetime
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# from mybaby import fig1, ax1, fig2, ax2, date_range, heatmap_data_bottle, heatmap_data_sleep, time_range
from scipy.signal import find_peaks

MORNING_START = 7
NIGHT_START = 18
SLEEP_ERR = 10 * 60 # 10 mins in seconds
BOTTLE_ERR = 15 # ml


def add_day_of_week(df, date_column):
    """
    Add a day of the week column to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (str): Name of the column containing dates

    Returns:
    pd.DataFrame: The DataFrame with an additional 'day_of_week' column
    """
    df.loc[:,'day_of_week'] = pd.to_datetime(df.loc[:,date_column]).dt.day_name()
    return df


def group_and_sum_by_day(df, date_column, sum_columns, err):
    """
    Group a DataFrame by day and sum specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (str): Name of the column containing dates
    sum_columns (list): List of column names to sum

    Returns:
    pd.DataFrame: A new DataFrame grouped by day with summed columns
    """
    # Ensure the date column is in datetime format
    df.loc[:,date_column] = pd.to_datetime(df.loc[:,date_column])

    functions = ['sum', 'count']
    agg_dict = {}
    for param in sum_columns:
        for func in functions:
            agg_dict[f"{param}_{func}"] = (param, func)

    # Group by date (ignoring time) and sum specified columns
    grouped_df = df.groupby(df[date_column].dt.date)[sum_columns].agg(**agg_dict).reset_index()
    for param in sum_columns:
        grouped_df[f"{param}_err"] = err * grouped_df[f"{param}_count"].pow(1./2.)

    return grouped_df


def group_and_sum_durations_by_day(df, date_columns, err):
    """
    Group a DataFrame by day and sum specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (list): Name of the column containing dates start and end

    Returns:
    pd.DataFrame: A new DataFrame grouped by day with summed columns
    """
    # Ensure the date column is in datetime format
    for date_col in date_columns:
        df.loc[:,date_col] = pd.to_datetime(df.loc[:,date_col])

    df.loc[:,'event_duration'] = (df.loc[:,date_columns[1]] - df.loc[:,date_columns[0]]).dt.total_seconds()

    functions = ['sum', 'count']
    agg_dict = {}
    for func in functions:
        agg_dict[f"event_duration_{func}"] = ("event_duration", func)

    # Group by date (ignoring time) and sum specified columns
    grouped_df = df.groupby(df[date_columns[0]].dt.date)[['event_duration']].agg(**agg_dict).reset_index()
    grouped_df["event_duration_sum_err"] = err * grouped_df["event_duration_count"].pow(1./2.)

    return grouped_df


def group_and_mean_by_day_of_week(df, date_column, mean_columns, err):
    """
    Group a DataFrame by day and mean specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (str): Name of the column containing dates
    sum_columns (list): List of column names to sum

    Returns:
    pd.DataFrame: A new DataFrame grouped by day with summed columns
    """
    # Ensure the date column is in datetime format
    df.loc[:,date_column] = pd.to_datetime(df.loc[:,date_column])
    day_of_week = add_day_of_week(df, date_column)

    functions = ['mean', 'std', 'count']
    agg_dict = {}
    for param in mean_columns:
        for func in functions:
            agg_dict[f"{param}_{func}"] = (param, func)

    # Group by day of week and mean specified columns
    grouped_df = df.groupby(day_of_week['day_of_week'])[mean_columns].agg(**agg_dict).reset_index()

    # error on mean of entries per weekday
    for param in mean_columns:
        grouped_df.loc[:,f"{param}_err"] = err / grouped_df.loc[:,f"{param}_count"].pow(1./2.)
    grouped_df = grouped_df.reindex([3,1,5,6,4,0,2])

    return grouped_df


def group_and_mean_sums_by_day_of_week(df, date_column, mean_columns, err):
    """
    Group a DataFrame by day and sum specified columns, then report mean in day of the week.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (str): Name of the column containing dates
    mean_columns (list): List of column names to mean/sum

    Returns:
    pd.DataFrame: A new DataFrame grouped by day with meaned columns
    """
    # Ensure the date column is in datetime format
    df.loc[:,date_column] = pd.to_datetime(df.loc[:,date_column])

    functions = ['sum', 'count']
    agg_dict = {}
    for param in mean_columns:
        for func in functions:
            agg_dict[f"{param}_{func}"] = (param,func)

    # Group by date (ignoring time) and sum specified columns
    grouped_summed_by_day_df = df.groupby(df[date_column].dt.date)[mean_columns].agg(**agg_dict).reset_index()
    # _sum and _count are totals per day (usually <10)

    grouped_summed_by_day_df = add_day_of_week(grouped_summed_by_day_df, date_column)

    functions = ['mean', 'std', 'count']
    agg_dict = {}
    for param in mean_columns:
        for func in functions:
            agg_dict[f"{param}_sum_{func}"] = (f"{param}_sum",func)
        agg_dict[f"{param}_count_sum"] = (f"{param}_count","sum")

    # Group by day of week and mean specified columns
    grouped_on_sums_df = grouped_summed_by_day_df.groupby('day_of_week').agg(**agg_dict).reset_index()

    # _mean _std and _count are values for same weekday (which have been summed), i.e. mean of weekday sum.
    for param in mean_columns:
        # error = # sum/# week * err
        grouped_on_sums_df.loc[:,f"{param}_err"] = err * ( grouped_on_sums_df.loc[:,f"{param}_count_sum"].pow(1./2.) / grouped_on_sums_df.loc[:,f"{param}_count_sum"] )
    grouped_on_sums_df = grouped_on_sums_df.reindex([3,1,5,6,4,0,2])

    return grouped_on_sums_df


def group_and_sum_by_day_of_week(df, date_column, sum_columns, err=1.0):
    """
    Group a DataFrame by day of week and sum specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (str): Name of the column containing dates
    sum_columns (list): List of column names to sum

    Returns:
    pd.DataFrame: A new DataFrame grouped by day of week with summed columns
    """
    # Ensure the date column is in datetime format
    df.loc[:,date_column] = pd.to_datetime(df.loc[:,date_column])
    day_of_week = add_day_of_week(df, date_column)

    functions = ['sum', 'count']
    agg_dict = {}
    for param in sum_columns:
        for func in functions:
            agg_dict[f"{param}_{func}"] = (param,func)

    # Group by date (ignoring time) and sum specified columns
    grouped_df = df.groupby(day_of_week.loc[:,'day_of_week'])[sum_columns].agg(**agg_dict).reset_index()
    for param in sum_columns:
        # error = err* # sum
        grouped_df.loc[:,f"{param}_err"] = err * grouped_df.loc[:,f"{param}_count"].pow(1./2.)

    grouped_df = grouped_df.reindex([3,1,5,6,4,0,2])

    return grouped_df


def filter_date_range(df, date_column, start_date, end_date):
    """
    Filter the DataFrame for a specific date range.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    date_column (str): Name of the column containing dates
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    pd.DataFrame: The filtered DataFrame
    """
    df.loc[:,date_column] = pd.to_datetime(df.loc[:,date_column])
    mask = (df.loc[:,date_column] >= start_date) & (df.loc[:,date_column] <= end_date)
    return df.loc[mask]


def energy_model(t, E0, A, k, t0):
    return E0 + A * np.heaviside(t - t0, 1) * np.exp(-k * (t - t0))


def categorize_day_night(df, start_time_col, end_time_col):
    """
    Categorize entries as day or night and create daytime and daytime_Time columns.

    Night is defined as 7 PM to 7 AM the next day, and is associated with the second day.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    start_time_col (str): Name of the column containing start times
    end_time_col (str): Name of the column containing end times

    Returns:
    pd.DataFrame: The DataFrame with additional 'daytime' and 'daytime_Time' columns
    """
    # Ensure datetime format
    df.loc[:,start_time_col] = pd.to_datetime(df.loc[:,start_time_col])
    df.loc[:,end_time_col] = pd.to_datetime(df.loc[:,end_time_col])

    # Define night start and end times
    night_start = datetime.time(NIGHT_START, 0)  # 7 PM
    night_end = datetime.time(MORNING_START, 0)    # 7 AM

    def categorize(row):
        start = row[start_time_col]
        end = row[end_time_col]

        # Check if the entry spans midnight
        spans_midnight = start.date() != end.date()

        # Determine if it's night time
        if spans_midnight:
            is_night = True
        elif start.time() >= night_start or end.time() <= night_end:
            is_night = True
        else:
            is_night = False

        # Assign daytime and date
        if is_night:
            return 'night', end.date()
        else:
            return 'day', start.date()

    # Apply the categorization
    df.loc[:,'daytime'], df.loc[:,'daytime_Time'] = zip(*df.loc[:,start_time_col].apply(categorize))

    return df


def categorize_day_night_start_only(df, start_time_col):
    """
    Categorize entries as day or night and create daytime and daytime_Time columns.

    Night is defined as 7 PM to 7 AM the next day, and is associated with the second day.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    start_time_col (str): Name of the column containing start times

    Returns:
    pd.DataFrame: The DataFrame with additional 'daytime' and 'daytime_Time' columns
    """
    # Ensure datetime format
    df.loc[:,start_time_col] = pd.to_datetime(df.loc[:,start_time_col])

    # Define night start and end times
    night_start = datetime.time(18, 0)  # 7 PM
    night_end = datetime.time(7, 0)    # 7 AM

    def categorize(row):
        start = row[start_time_col]
        if night_start <= start.time() or start.time() < night_end:
            daytime = 'night'
            # If it's night and after midnight, use the same date
            # If it's night before midnight, use the next day's date
            daytime_time = start.date() if start.time() < night_end else start.date() + pd.Timedelta(days=1)
        else:
            daytime = 'day'
            daytime_time = start.date()

        return daytime, daytime_time

    # Apply the categorization
    result = df.apply(categorize, axis=1)

    # Assign values to each column separately
    df.loc[:, 'daytime'] = [r[0] for r in result]
    df.loc[:, 'daytime_Time'] = [r[1] for r in result]


    return df


def get_type_data(df, entry_type ):
    return df.query(f"type == '{entry_type}'")


def get_timeperiod_data(df, period ):
    if 'daytime' in df:
        return df.query(f"daytime == '{period}'")
    else:
        return df


def get_plot_kwargs(period, plot_type='scatter',custom_label=None):
    base_kwargs = {
        'day': {'color': 'gold', 'label': 'day', 'ls': 'None', 'alpha': 0.7},
        'night': {'color': 'slateblue', 'label': 'night', 'ls': 'None', 'alpha': 0.7},
        'all': {'color': 'black', 'label': 'all', 'ls': 'None', 'alpha': 0.7}
    }

    if plot_type.lower() != 'histogram':
        for key in base_kwargs:
            base_kwargs[key]['marker'] = 'o'

    if custom_label:
        base_kwargs[period]['label']=custom_label

    return base_kwargs[period]


def get_consecutive_durations_from_two_dfs(df1, df2, date_column='date', duration_column='duration'):
    """
    Returns durations from two DataFrames where df2's date is df1's date + 1 day.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame
    df2 (pd.DataFrame): The second DataFrame
    date_column (str): Name of the column containing dates
    duration_column (str): Name of the column containing durations

    Returns:
    list of tuples: Each tuple contains (df1_date, df1_duration, df2_date, df2_duration)
    """
    # Ensure both date columns are in datetime format
    df1[date_column] = pd.to_datetime(df1[date_column])
    df2[date_column] = pd.to_datetime(df2[date_column])

    # Sort both DataFrames by the date column
    df1 = df1.sort_values(date_column)
    df2 = df2.sort_values(date_column)

    # Create a dictionary mapping dates in df2 to their durations for fast lookup
    df2_date_to_duration = dict(zip(df2[date_column], df2[duration_column]))

    results = []

    # Iterate through each row in df1
    for _, row in df1.iterrows():
        current_date = row[date_column]
        next_date = current_date + pd.Timedelta(days=1)

        # Check if the next_date exists in df2
        if next_date in df2_date_to_duration:
            results.append((
                current_date,
                row[duration_column],
                next_date,
                df2_date_to_duration[next_date]
            ))

    return np.array(results)


def make_year_heatmap(dates):
    # Assuming your input is a pandas Series of datetime objects named 'dates_series'
    # If you don't have this, you can create a sample one like this:
    # dates_series = pd.Series([pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-14'),
    #                           pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-15'),
    #                           pd.Timestamp('2025-05-20'), pd.Timestamp('2025-05-20'), pd.Timestamp('2025-12-31')])


    # Count occurrences of each date
    date_counts = Counter(dates)

    start_date = min(dates).replace(day=1)
    end_date = max(dates)
    total_weeks = (end_date - start_date).days // 7 + 1

    # Determine the range of years in the data
    min_year = min(date.year for date in dates)
    max_year = max(date.year for date in dates)
    num_years = max_year - min_year + 1

    # Create a 2D array to hold the data
    data = np.zeros((7, total_weeks))

    # Fill the data array
    for date, count in date_counts.items():
        week_num = (date - start_date).days // 7
        day_of_week = date.weekday()
        data[day_of_week, week_num] = count

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 4))

    # Create the heatmap
    heatmap = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Set up the y-axis labels (days of the week)
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    current_date = start_date
    for week in range(total_weeks):
        if current_date.day == 1 or week == 0 or current_date.day <=7:
            ax.text(week, -1, current_date.strftime('%b'),
                    ha='center', va='center', fontsize=8)
        if current_date.month == 1 and current_date.day <= 7:
            ax.text(week, -1.5, str(current_date.year),
                    ha='center', va='center', fontsize=10, fontweight='bold')
        current_date += datetime.timedelta(days=7)

    # Add colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Count of occurrences')
    print(total_weeks)

    # Set xlabel
    plt.xlabel("Week #")

    # Set x-axis ticks to integers in increments of 5
    tick_positions = np.arange(0, total_weeks+1, 4)  # Start at 0, go to max(x), step by 5
    ax.set_xticks(tick_positions)

    # Format x-axis labels (optional: ensure they are integers)
    ax.set_xticklabels([str(int(tick)) for tick in tick_positions])

    # Set title and adjust layout
    plt.title('Date Occurrence Heatmap', y=1.2, pad=15)
    plt.tight_layout()

    # Show the plot
    plt.show()
    return ax


def plot_fft_results(hourly_bins, hourly_counts, freq, power_spectrum):
    """
    Plot FFT results.

    Parameters:
    hourly_bins (pd.DatetimeIndex): The time bins used in FFT
    hourly_counts (pd.Series): Amplitude counts in each bin
    freq (np.array): Array of frequencies from FFT
    power_spectrum (np.array): Power spectrum from FFT

    Returns:
    tuple: fig, axes
    """
    # Plot original data
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(hourly_bins, hourly_counts, width=0.8, alpha=0.7, color='b')
    ax1.set_title('Original Time vs Amplitude Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    plt.show()

    # Plot power spectrum
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    positives = freq > 0
    period = (1 / freq)
    ax2.plot(freq[positives], power_spectrum[positives], color='b')
    ax2.set_title('Power Spectrum')
    ax2.set_xlabel('Period (min)')
    ax2.set_ylabel('Power')
    ax2.set_yscale('log')  # Log scale for better visibility of peaks
    ax2.set_xscale('log')
    plt.show()

    return (fig1, ax1), (fig2, ax2)


def plot_time_differences(differences, amplitudes, frequency):
    """
    Plot time differences from reference beat.

    Parameters:
    differences (np.array): Array of time differences
    amplitudes (pd.Series or np.array): Amplitudes corresponding to the times
    frequency (float): Reference frequency used for calculating the differences
    """
    plt.figure(figsize=(12, 6))
    plt.plot(differences / 3600, amplitudes, 'o')
    plt.title(f'Time Differences from Reference Beat (F={frequency:.2f} Hz)')
    plt.ylabel('Amplitude (ml)')
    plt.xlabel('Difference (hour)')
    plt.show()#%%



def analyze_fft(df, time_col='start_time', amplitude_col='amplitude'):
    """
    Perform FFT on amplitude data and find peak frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time and amplitude data
    time_col (str): Name of the column containing time data
    amplitude_col (str): Name of the column containing amplitude data

    Returns:
    tuple: (minute_bins, minute_counts, frequencies, power_spectrum, peak_frequencies)
    """
    # Ensure time data is in datetime format and sort
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)

    # Get min and max timestamps
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    # Create time bins from min to max at minute intervals
    binning_time = 'min'  # Binning set to minutes
    time_bins = pd.date_range(start=min_time.floor(binning_time), end=max_time.ceil(binning_time), freq=binning_time)

    # Count events in each minute bin
    time_counts = pd.Series(0, index=time_bins)
    binned = df.groupby(pd.Grouper(key=time_col, freq=binning_time)).size().reindex(time_bins, fill_value=0)
    time_counts.update(binned)

    # Apply detrending to remove any linear trend in the data
    detrended_counts = time_counts - np.mean(time_counts)

    # Apply Hanning window to reduce spectral leakage
    hanning_window = np.hanning(len(detrended_counts))
    windowed_counts = detrended_counts * hanning_window

    # Perform FFT
    fft_result = np.fft.fft(windowed_counts)
    power_spectrum = np.abs(fft_result[:len(fft_result)//2]) ** 2  # Take positive frequencies only (real part)
    freq = np.fft.fftfreq(len(fft_result), d=60)[:len(fft_result)//2]  # Convert to frequencies in Hz (1/min)

    # Find peaks in the power spectrum
    peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.1)  # Peaks > 10% of the max
    peak_frequencies = freq[peaks]

    return time_bins, time_counts, freq, power_spectrum, peak_frequencies


def plot_heatmap_with_date_range(df, start_date_str='2025-01-01', end_date_str='2025-05-31'):
    # First, get the combined date range across both DataFrames
    df_bottle = get_type_data(df, 'bottle')
    df_sleep = get_type_data(df, 'sleep')

    # Get the combined min and max dates from both DataFrames
    min_date = min(df_bottle['enteredDate'].min().date(), df_sleep['leftStart'].min().date())
    max_date = max((df_bottle['enteredDate'] + pd.Timedelta(15, 'min')).max().date(),
                   df_sleep['leftEnd'].max().date())

    # Create a unified date range for both heatmaps
    date_range = pd.date_range(start=min_date, end=max_date)

    # Create time range for y-axis (24 hours, in 15-minute intervals)
    time_range = pd.date_range(start='00:00', end='23:59', freq='15T').time

    # Create 2D arrays with the same dimensions for both heatmaps
    heatmap_data_bottle = np.zeros((len(time_range), len(date_range)))
    heatmap_data_sleep = np.zeros((len(time_range), len(date_range)))

    # Fill the bottle feeding heatmap data
    for _, row in df_bottle.iterrows():
        start_date = row['enteredDate'].date()
        end_date = (row['enteredDate'] + pd.Timedelta(15, 'min')).date()
        start_time = row['enteredDate'].time()
        end_time = (row['enteredDate'] + pd.Timedelta(15, 'min')).time()

        date_idx_start = np.where(date_range.normalize() == pd.Timestamp(start_date).normalize())[0][0]
        date_idx_end = np.where(date_range.normalize() == pd.Timestamp(end_date).normalize())[0][0]

        start_idx = time_range.searchsorted(start_time)
        end_idx = time_range.searchsorted(end_time)

        if date_idx_start == date_idx_end:
            heatmap_data_bottle[start_idx:end_idx, date_idx_start] = 1
        else:
            heatmap_data_bottle[start_idx:, date_idx_start] = 1
            heatmap_data_bottle[:end_idx, date_idx_end] = 1
            if date_idx_end - date_idx_start > 1:
                heatmap_data_bottle[:, date_idx_start + 1:date_idx_end] = 1

    # Fill the sleep heatmap data
    for _, row in df_sleep.iterrows():
        start_date = row['leftStart'].date()
        end_date = row['leftEnd'].date()
        start_time = row['leftStart'].time()
        end_time = row['leftEnd'].time()

        date_idx_start = np.where(date_range.normalize() == pd.Timestamp(start_date).normalize())[0][0]
        date_idx_end = np.where(date_range.normalize() == pd.Timestamp(end_date).normalize())[0][0]

        start_idx = time_range.searchsorted(start_time)
        end_idx = time_range.searchsorted(end_time)

        if date_idx_start == date_idx_end:
            heatmap_data_sleep[start_idx:end_idx, date_idx_start] = 1
        else:
            heatmap_data_sleep[start_idx:, date_idx_start] = 1
            heatmap_data_sleep[:end_idx, date_idx_end] = 1
            if date_idx_end - date_idx_start > 1:
                heatmap_data_sleep[:, date_idx_start + 1:date_idx_end] = 1
    # Convert string dates to datetime objects
    start_date = pd.Timestamp(start_date_str).date()
    end_date = pd.Timestamp(end_date_str).date()

    # Find the indices for the specified date range
    start_idx = np.where(date_range.date >= start_date)[0]
    end_idx = np.where(date_range.date <= end_date)[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        print(f"Warning: Specified date range {start_date_str} to {end_date_str} is outside the available data range.")
        return

    start_idx = start_idx[0]
    end_idx = end_idx[-1]

    # Extract the portion of the heatmap data for the specified date range
    date_slice = slice(start_idx, end_idx + 1)
    limited_date_range = date_range[date_slice]
    limited_heatmap_bottle = heatmap_data_bottle[:, date_slice]
    limited_heatmap_sleep = heatmap_data_sleep[:, date_slice]

    # TRANSPOSE THE DATA
    limited_heatmap_bottle_T = limited_heatmap_bottle.T
    limited_heatmap_sleep_T = limited_heatmap_sleep.T

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 8))

    # Add vertical bands instead of horizontal
    # We need to convert the hour ranges to indices in the time array
    morning_start = 0  # 0:00
    morning_end = 7 * 4  # 7:00 (4 steps per hour)
    evening_start = 18 * 4  # 18:00
    evening_end = len(time_range)  # 23:59

    ax.axvspan(morning_start, morning_end, facecolor='slateblue', alpha=0.2)  # 0-7 hours
    ax.axvspan(morning_end, evening_start, facecolor='gold', alpha=0.2)  # 7-18 hours
    ax.axvspan(evening_start, evening_end, facecolor='slateblue', alpha=0.2)  # 18-24 hours

    # Create custom colormaps
    # For sleep data - white for 0, blue for 1
    cmap_sleep = ListedColormap(['white', 'blue'])

    # For bottle data - transparent for 0, green for 1
    cmap_bottle = ListedColormap(['none', 'red'])  # 'none' makes the 0 values transparent

    # Plot sleep heatmap first (as the background)
    im_sleep = ax.imshow(limited_heatmap_sleep_T, aspect='auto', cmap=cmap_sleep, interpolation='nearest')

    # Then plot bottle heatmap with transparency for zeros
    im_bottle = ax.imshow(limited_heatmap_bottle_T, aspect='auto', cmap=cmap_bottle, interpolation='nearest')

    # Set y-axis ticks and labels
    num_dates = len(limited_date_range)
    # Adjust tick interval based on the range length
    tick_interval = max(1, num_dates // 10)  # Show at most ~10 date labels
    tick_positions = np.arange(0, num_dates, tick_interval)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(limited_date_range[tick_positions].strftime('%Y-%m-%d'))

    # Set x-axis ticks and labels (every hour)
    hour_indices = np.arange(0, len(time_range), 4)
    ax.set_xticks(hour_indices)
    ax.set_xticklabels([t.strftime('%H:%M') for t in time_range[hour_indices]], rotation=45, ha='right')
    plt.grid(True, color='gray', alpha=0.2, linestyle=':')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Date')
    ax.set_title(f'Bottle Feeding and Sleep Heatmap ({start_date_str} to {end_date_str})')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Bottle Feeding'),
        Patch(facecolor='blue', edgecolor='blue', label='Sleep')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    return ax


def plot_heatmap_with_date_range_transpose(full_df, start_date_str='2024-01-01', end_date_str='2025-05-31'):
    # First, get the combined date range across both DataFrames
    df_bottle = get_type_data(full_df, 'bottle')
    df_sleep = get_type_data(full_df, 'sleep')

    # Get the combined min and max dates from both DataFrames
    min_date = min(df_bottle['enteredDate'].min().date(), df_sleep['leftStart'].min().date())
    max_date = max((df_bottle['enteredDate'] + pd.Timedelta(15, 'min')).max().date(),
                   df_sleep['leftEnd'].max().date())

    # Create a unified date range for both heatmaps
    date_range = pd.date_range(start=min_date, end=max_date)

    # Create time range for y-axis (24 hours, in 15-minute intervals)
    time_range = pd.date_range(start='00:00', end='23:59', freq='15T').time

    # Create 2D arrays with the same dimensions for both heatmaps
    heatmap_data_bottle = np.zeros((len(time_range), len(date_range)))
    heatmap_data_sleep = np.zeros((len(time_range), len(date_range)))

    # Fill the bottle feeding heatmap data
    for _, row in df_bottle.iterrows():
        start_date = row['enteredDate'].date()
        end_date = (row['enteredDate'] + pd.Timedelta(15, 'min')).date()
        start_time = row['enteredDate'].time()
        end_time = (row['enteredDate'] + pd.Timedelta(15, 'min')).time()

        date_idx_start = np.where(date_range.normalize() == pd.Timestamp(start_date).normalize())[0][0]
        date_idx_end = np.where(date_range.normalize() == pd.Timestamp(end_date).normalize())[0][0]

        start_idx = time_range.searchsorted(start_time)
        end_idx = time_range.searchsorted(end_time)

        if date_idx_start == date_idx_end:
            heatmap_data_bottle[start_idx:end_idx, date_idx_start] = 1
        else:
            heatmap_data_bottle[start_idx:, date_idx_start] = 1
            heatmap_data_bottle[:end_idx, date_idx_end] = 1
            if date_idx_end - date_idx_start > 1:
                heatmap_data_bottle[:, date_idx_start + 1:date_idx_end] = 1

    # Fill the sleep heatmap data
    for _, row in df_sleep.iterrows():
        start_date = row['leftStart'].date()
        end_date = row['leftEnd'].date()
        start_time = row['leftStart'].time()
        end_time = row['leftEnd'].time()

        date_idx_start = np.where(date_range.normalize() == pd.Timestamp(start_date).normalize())[0][0]
        date_idx_end = np.where(date_range.normalize() == pd.Timestamp(end_date).normalize())[0][0]

        start_idx = time_range.searchsorted(start_time)
        end_idx = time_range.searchsorted(end_time)

        if date_idx_start == date_idx_end:
            heatmap_data_sleep[start_idx:end_idx, date_idx_start] = 1
        else:
            heatmap_data_sleep[start_idx:, date_idx_start] = 1
            heatmap_data_sleep[:end_idx, date_idx_end] = 1
            if date_idx_end - date_idx_start > 1:
                heatmap_data_sleep[:, date_idx_start + 1:date_idx_end] = 1

    # Function to limit the display to a specific date range
    # Convert string dates to datetime objects
    start_date = pd.Timestamp(start_date_str).date()
    end_date = pd.Timestamp(end_date_str).date()

    # Find the indices for the specified date range
    start_idx = np.where(date_range.date >= start_date)[0]
    end_idx = np.where(date_range.date <= end_date)[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        print(f"Warning: Specified date range {start_date_str} to {end_date_str} is outside the available data range.")
        return

    start_idx = start_idx[0]
    end_idx = end_idx[-1]

    # Extract the portion of the heatmap data for the specified date range
    date_slice = slice(start_idx, end_idx + 1)
    limited_date_range = date_range[date_slice]
    limited_heatmap_bottle = heatmap_data_bottle[:, date_slice]
    limited_heatmap_sleep = heatmap_data_sleep[:, date_slice]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Add horizontal bands
    ax.axhspan(0, 7 * 4, facecolor='slateblue', alpha=0.2)  # 0-7 hours (4 steps per hour)
    ax.axhspan(7 * 4, 18 * 4, facecolor='gold', alpha=0.2)  # 7-18 hours
    ax.axhspan(18 * 4, 24 * 4, facecolor='slateblue', alpha=0.2)  # 18-24 hours

    # Create custom colormaps
    cmap_bottle = ListedColormap(['none', 'red'])
    cmap_sleep = ListedColormap(['white', 'blue'])

    # Plot both heatmaps
    im_sleep = ax.imshow(limited_heatmap_sleep, aspect='auto', cmap=cmap_sleep, interpolation='nearest')
    im_bottle = ax.imshow(limited_heatmap_bottle, aspect='auto', cmap=cmap_bottle, interpolation='nearest', alpha=0.8)

    # Set x-axis ticks and labels
    num_dates = len(limited_date_range)
    # Adjust tick interval based on the range length
    tick_interval = max(1, num_dates // 10)  # Show at most ~10 date labels
    tick_positions = np.arange(0, num_dates, tick_interval)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(limited_date_range[tick_positions].strftime('%Y-%m-%d'), rotation=45, ha='right')

    # Set y-axis ticks and labels (every hour)
    hour_indices = np.arange(0, len(time_range), 4)
    ax.set_yticks(hour_indices)
    ax.set_yticklabels([t.strftime('%H:%M') for t in time_range[hour_indices]])
    plt.grid(True, color='gray', alpha=0.2, linestyle=':')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Time')
    ax.set_title(f'Bottle Feeding and Sleep Heatmap ({start_date_str} to {end_date_str})')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Bottle Feeding'),
        Patch(facecolor='blue', edgecolor='blue', label='Sleep')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def to_datetime(full_df):
    date_fields = ['enteredDate', 'leftStart', 'leftEnd']
    for field in date_fields:
        full_df[field] = pd.to_datetime(full_df[field], format='%Y-%m-%dT%H:%M:%S.%fZ').dt.tz_localize(
            'UTC').dt.tz_convert('US/Eastern')
    return full_df


def to_numeric(full_df):
    num_fields = ['leftSeconds',
                  'rightSeconds',
                  'bottleAmount',
                  'bottleAmountOunce',
                  'weight',
                  'height',
                  'headCirc',
                  'temperature',
                  'weightPounds',
                  'heightInches',
                  'headCircInches',
                  'temperatureFah', 'singleTimerSeconds']
    for field in num_fields:
        full_df[field] = pd.to_numeric(full_df[field])
    return full_df


def get_comments(df, min_length=1):
    """
    Get entries with non-empty comments.

    Parameters:
    min_length (int): Minimum length of comments to include

    Returns:
    pd.DataFrame: DataFrame with non-empty comments
    """
    return df[df['customComment'].str.strip().str.len() >= min_length]
