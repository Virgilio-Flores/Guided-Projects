import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def convert_to_am_pm(time):
    """Converts a 24-hour time to AM/PM format.

    Args:
        time (int): A 24-hour time.

    Returns:
        str: The time in AM/PM format.
    """
    hour = time
    if hour == 0:
        return f"12 AM"
    elif hour < 12:
        return f"{hour} AM"
    elif hour == 12:
        return f"12 PM"
    else:
        hour -= 12
        return f"{hour} PM"
    
    
    
def create_year_column(df,date_column):
    """Extracts the year from a date column and creates a new column in the dataframe.

    Args:
        df (DataFrame): The pandas DataFrame containing the date column.
        date_column (str): The name of the date column.

    Returns:
        int: The year extracted from the date column.
    """
    return df[date_column].dt.year


def get_non_null_dates(df):
    """Returns the rows in the DataFrame where the 'date_time' column is not null.

    Args:
        df (DataFrame): The pandas DataFrame containing the 'date_time' and 'holiday' columns.

    Returns:
        DataFrame: A DataFrame containing the 'date_time' and 'holiday' columns where 'holiday' is not null.
    """
    return df.loc[df['holiday'].notnull(), ['date_time', 'holiday']]


def plot_holiday_traffic_volume_comparison(df, years, groupby_cols=['hour','holiday'], func=0, y_lim=(2500, 4000), ptype="line"):
    """Plots the mean traffic volume by holiday type for different years.

    Args:
        df (DataFrame): The pandas DataFrame containing the traffic data.
        years (array): An array of years to plot.
        groupby_cols (list, optional):  Defaults to ['hour','holiday'].
        func (int, optional):  Defaults to 0.
        y_lim (tuple, optional):  Defaults to (2500, 4000).
        ptype (str, optional):  Defaults to "line".
    """
    try:
        assert 'date_time' in df.columns
        assert 'hour' in df.columns
        assert 'holiday' in df.columns
        assert 'traffic_volume' in df.columns
    except AssertionError:
        print("One or more required columns are missing in the dataframe.")

    
    fig, axs = plt.subplots(2, 4, figsize=(20, 12))  
    axs = axs.flatten()

    # Initialize an empty list to collect handles and labels for the legend
    handles, labels = [], []
    
    plot_funcs = [plot_holiday_traffic_volume_for_year, plot_monthly_traffic_volume_for_year]
    
    for i, year in enumerate(years):
        
        if func==0:
            current_handles, current_labels = plot_funcs[func](df, year, axs[i], groupby_cols, ptype=ptype)
        elif func==1:
            current_handles, current_labels = plot_funcs[func](df, year, axs[i], groupby_cols, y_lim, ptype=ptype)
        elif func==2:
            current_handles, current_labels = plot_gen_traffic_volume_for_year(df, year, axs[i], groupby_cols, ptype=ptype)
            
        handles.extend(current_handles)
        labels.extend(current_labels)

    # After plotting, create a unified legend. Use 'unique' to avoid duplicate labels.
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if func==0:
        fig.legend(*zip(*unique), loc='right', bbox_to_anchor=axs[-1].get_position())
    
    if len(years) < len(axs):
        for i in range(len(years), len(axs)):
            axs[i].axis('off')  # Turn off the remaining subplots
            
    fig.suptitle('Mean Traffic Volume by Holiday Type for Different Years \n', fontsize=16)
    for i,ax in enumerate(axs):
        if i not in [0,4]:
            ax.set_ylabel('')
            ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.show()

def plot_holiday_traffic_volume_for_year(df, year, ax, groupby_cols=["hour",'holiday'], ptype="line"):
    
    """
    Plots the mean traffic volume by holiday type for a specific year.
    
    Used in the plot_holiday_traffic_volume_comparison function to plot the mean traffic volume by holiday type for different years.
    
    Args:
        df (DataFrame): The pandas DataFrame containing the traffic data.
        year (int): The year to plot.
        ax (axes subplot): The axis to plot the data on.
        groupby_cols (list, optional): Defaults to ["hour",'holiday'].

    Returns:
        axis handles and labels for the legend
    """
    
    
    try:
        assert len(groupby_cols) > 1
    except AssertionError:
        print("Provide a valid groupby column.")
    mean_traffic_volume = df.loc[df["date_time"].dt.year == year].groupby(groupby_cols)['traffic_volume'].mean()
    unique_vals = df[groupby_cols[1]].dropna().unique()

    for val in unique_vals:
        filtered_data = mean_traffic_volume.loc[mean_traffic_volume.index.get_level_values(groupby_cols[1]) == val]
        if ptype=="scatter":
            ax.scatter(filtered_data.index.get_level_values(groupby_cols[0]), filtered_data.values, label=val)
        else:
            ax.plot(filtered_data.index.get_level_values(groupby_cols[0]), filtered_data.values, label=val)

    ax.set_xlabel('Time of day')
    ax.set_ylabel('Mean Traffic Volume')
    ax.set_title(f'{year}')
    ax.set_ylim(0, 7000)

    # Return handles and labels for the current subplot
    return ax.get_legend_handles_labels()

def plot_monthly_traffic_volume_for_year(df, year, ax, groupby_cols=None, y_lim=(2500, 4000), ptype="line"):
    """Plots the mean traffic volume by month for a specific year.

    Args:
        df (DataFrame): The pandas DataFrame containing the traffic data.
        year (int):  The year to plot.
        ax (axis subplot): The axis to plot the data on.
        groupby_cols (_type_, optional): . Defaults to None.
        y_lim (tuple, optional): . Defaults to (2500, 4000).
        ptype (str, optional): . Defaults to "line".

    Returns:
        handles, labels: axis handles and labels for the legend
    """
  
    mean_traffic_volume = df.loc[df["date_time"].dt.year == year].groupby("month")['traffic_volume'].mean().reset_index()

    mean_traffic_volume = mean_traffic_volume.sort_values(by="month")
    ax.plot(mean_traffic_volume["month"], mean_traffic_volume["traffic_volume"])
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Traffic Volume')
    ax.set_title(f'{year}')
    ax.set_ylim(y_lim)
    ax.set_xlim(1, 12)
    ax.axvspan(10, 12, color="black", alpha=0.1)
    ax.axvspan(1, 2, color="black", alpha=0.1)

    ax.axvspan(5, 10, color="orange", alpha=0.2)
    ax.axvspan(2, 5, color="orange", alpha=0.1)
    # Return handles and labels for the current subplot
    return ax.get_legend_handles_labels()



def plot_holiday_traffic_volume_heatmap(df, year=None, gby_col='holiday', ):
    # Ensure 'date_time' is a datetime object and create necessary columns
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['hour'] = df['date_time'].dt.hour
    
    if year is not None:
        df = df.loc[df['year']==year]

    # Aggregate mean traffic volume by year, holiday, and hour
    mean_traffic_volume = df.groupby(['year', gby_col, 'hour'])['traffic_volume'].mean().reset_index()

    # Pivot the data for heatmap; adjust this based on the desired granularity (e.g., specific holiday)
    pivot_table = mean_traffic_volume.pivot_table(index='hour', columns=gby_col, values='traffic_volume')

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Mean Traffic Volume'})
    plt.title('Mean Traffic Volume by Year and Time of Day')
    plt.xlabel(gby_col.capitalize())
    plt.ylabel('Hour of Day')
    plt.show()
    
    
def plot_weather_traffic_volume_for_year(df, year, ax, groupby_cols=["hour",'weather_main']):
    """
    Plots the mean traffic volume by weather type for different years.
    
    Used in the plot_holiday_traffic_volume_comparison function to plot the mean traffic volume by weather type for different years.

    Args:
        df (DataFrame): The pandas DataFrame containing the traffic data.
        year (int): The year to plot.
        ax (axes subplot): The axis to plot the data on.
        groupby_cols (list, optional): Defaults to ["hour",'weather_main'].

    Returns:
        axis handles and labels for the legend
    """
    try:
        assert len(groupby_cols) > 1
    except AssertionError:
        print("Provide a valid groupby column.")
    mean_traffic_volume = df.loc[df["date_time"].dt.year == year].groupby(groupby_cols)['traffic_volume'].mean()
    unique_vals = df[groupby_cols[1]].dropna().unique()

    for val in unique_vals:
        filtered_data = mean_traffic_volume.loc[mean_traffic_volume.index.get_level_values(groupby_cols[1]) == val]
        ax.plot(filtered_data.index.get_level_values(groupby_cols[0]), filtered_data.values, label=val)

    ax.set_xlabel('Time of day')
    ax.set_ylabel('Mean Traffic Volume')
    ax.set_title(f'{year}')
    ax.set_ylim(0, 7000)

    # Return handles and labels for the current subplot
    return ax.get_legend_handles_labels()


def plot_gen_traffic_volume_for_year(df, year, ax, groupby_cols=None, y_lim=(2500, 4000)):
  
    mean_traffic_volume = df.loc[df["date_time"].dt.year == year].groupby("temp")['traffic_volume'].mean().reset_index()

    mean_traffic_volume = mean_traffic_volume.sort_values(by="temp")
    ax.scatter(mean_traffic_volume["temp"], mean_traffic_volume["traffic_volume"])
    
    ax.set_xlabel('Temp')
    ax.set_ylabel('Mean Traffic Volume')
    ax.set_title(f'{year}')
    ax.set_ylim(y_lim)

    return ax.get_legend_handles_labels()


def plot_heatmap(df, 
                 title, 
                 time_range, 
                 years=[2013, 2016, 2017], 
                 ax=None, 
                 xlabel='Temperature', 
                 ylabel='Traffic Volume', 
                 cbar_label='Number of occurrences',

                 min_y=None, 
                 max_y=None,
                 ):
    
    """Plots a heatmap of the traffic volume based on temperature and time of day.

    Args:
        df (DataFrame): The pandas DataFrame containing the traffic data.
        title (str): The title of the plot.
        time_range (tuple): A tuple of two integers representing the start and end of the time range.
        years (list, optional): A list of years to include in the plot. Defaults to [2013, 2016, 2017].
        ax (Axis, optional): An Axis object to plot the heatmap on. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Temperature'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Traffic Volume'.
        cbar_label (str, optional): The label for the colorbar. Defaults to 'Number of occurrences'.
        min_y (int, optional): The minimum value for the y-axis. Defaults to None.
        max_y (int, optional): The maximum value for the y-axis. Defaults to None.
        
    Returns:
        _type_: _description_
    """
    
    # Extracting the temperature and traffic volume columns based on the conditions
    year_bool = df["date_time"].dt.year.isin(years) # Creating a boolean mask for the years
    time_bool = (df["hour"] >= time_range[0]) & (df["hour"] <= time_range[1]) # Creating a boolean mask for the time range
    temp = df.loc[(year_bool & time_bool), "temp"] # Extracting the temperature and traffic volume columns based on the conditions
    traffic_volume = df.loc[(year_bool & time_bool), "traffic_volume"] if time_range is not None else df.loc[year_bool, "traffic_volume"]

    if ax is None:
        # Creating the heatmap using hist2d
        plt.hist2d(temp, traffic_volume, bins=(30, 30), cmap='viridis', cmin=1)
        plt.colorbar(label='Number of occurrences')

        # Setting labels for clarity
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.xlim(240, 310)
        plt.ylim(0, 7000)
        plt.title(title)
        plt.show()

    else:
        # Creating the heatmap using hist2d
        quadmesh = ax.hist2d(temp, traffic_volume, bins=(30, 30), cmap='viridis', cmin=1)
        
        # Adding a colorbar
        if cbar_label:
            cbar = plt.colorbar(quadmesh[3], ax=ax, label='Number of occurrences')
            cbar.set_label('Number of occurrences')

        # Setting labels for clarity
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlim(240, 310)
        ax.set_ylim(0, 7000)
        ax.set_title(title)
        
        if max_y:
            ax.set_yticks(np.arange(0, max_y, 3000))

        return ax
