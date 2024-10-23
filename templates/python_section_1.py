from typing import Dict, List
import re
import pandas as pd
import polyline
from math import radians, sin, cos, sqrt, atan2


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        result.extend(group[::-1])
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    return result

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}
    
    def _flatten(d, parent_key=''):
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                flattened[new_key] = value
    
    _flatten(nested_dict)
    return flattened

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(counter, current_perm):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for num in counter:
            if counter[num] > 0:
                current_perm.append(num)
                counter[num] -= 1
                
                backtrack(counter, current_perm)
                
                current_perm.pop()
                counter[num] += 1

    result = []
    backtrack(Counter(nums), [])
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    return re.findall(pattern, text)

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coords = polyline.decode(polyline_str)
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000  
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
        c = 2*atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    df['distance'] = 0
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine_distance(
            df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'],
            df.loc[i, 'latitude'], df.loc[i, 'longitude']
        )
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    rotated = [[matrix[n-1-j][i] for j in range(n)] for i in range(n)]

    row_sums = [sum(row) for row in rotated]
    col_sums = [sum(col) for col in zip(*rotated)]
    result = [[row_sums[i] + col_sums[j] - rotated[i][j] for j in range(n)] for i in range(n)]
    
    return result


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()
