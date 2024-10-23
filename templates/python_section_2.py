import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance')
    
    np.fill_diagonal(distance_matrix.values, 0)
    
    distance_matrix = distance_matrix.combine_first(distance_matrix.T)
    
    # Fill NaN values with 0
    distance_matrix = distance_matrix.fillna(0)

    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_df = df.reset_index().melt(id_vars='id_start', var_name='id_end', value_name='distance')
    
    unrolled_df = unrolled_df[unrolled_df['distance'] != 0]
    
    unrolled_df = unrolled_df.reset_index(drop=True)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    avg_distances = df.groupby('id_start')['distance'].mean()
    
    reference_avg_distance = avg_distances[reference_id]
    
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1
    
    ids_within_threshold = avg_distances[(avg_distances >= lower_threshold) & (avg_distances <= upper_threshold)]
    
    result_df = pd.DataFrame({'id': ids_within_threshold.index, 'average_distance': ids_within_threshold.values})

    return result_df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    df['base_toll_rate'] = df['distance'] * 0.01
    
    for vehicle_type, rate_coeff in rate_coefficients.items():
        df[f'{vehicle_type}_toll'] = df['base_toll_rate'] * rate_coeff
    
    df = df.drop('base_toll_rate', axis=1)

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    time_intervals = [
        (time(0, 0), time(5, 0), 0.8),
        (time(5, 0), time(10, 0), 1.2),
        (time(10, 0), time(17, 0), 1.0),
        (time(17, 0), time(20, 0), 1.2),
        (time(20, 0), time(23, 59), 0.8)
    ]
    
    results = []
    
    for (start, end), group in df.groupby(['id_start', 'id_end']):
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for start_time, end_time, factor in time_intervals:
                new_row = group.iloc[0].copy()
                new_row['start_day'] = day
                new_row['start_time'] = start_time
                new_row['end_day'] = day
                new_row['end_time'] = end_time
                
                for col in ['moto_toll', 'car_toll', 'rv_toll', 'bus_toll', 'truck_toll']:
                    new_row[col] *= factor
                
                if day in ['Saturday', 'Sunday']:
                    for col in ['moto_toll', 'car_toll', 'rv_toll', 'bus_toll', 'truck_toll']:
                        new_row[col] *= 0.7
                
                results.append(new_row)
    
    time_based_df = pd.DataFrame(results)
    
    columns_order = ['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time',
                     'moto_toll', 'car_toll', 'rv_toll', 'bus_toll', 'truck_toll']
    time_based_df = time_based_df[columns_order]

    return time_based_df
