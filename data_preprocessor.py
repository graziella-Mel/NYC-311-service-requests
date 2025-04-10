import pandas as pd
import numpy as np
from utils import preprocessTime, addHolidays, removeOutliers


def preprocessDataset(data):
    """
    Preprocesses the 311 service request dataset by cleaning, transforming, and engineering features.

    Steps performed:
    - Converts 'Created Date' and 'Closed Date' from string to datetime format.
    - Sorts the dataset by 'Created Date' and retains only the last 1/6th of the records.
    - Drops rows with missing values.
    - Computes the response time in hours.
    - Extracts time-related features from both 'Created Date' and 'Closed Date'.
    - Adds a holiday indicator feature based on 'Created Date'.
    - Transforms latitude and longitude into cyclic features (sin/cos).
    - Removes duplicate rows.
    - Filters out rows with negative response times.
    - Removes statistical outliers from the 'response_time' column.
    - Applies a log transformation to the response time and stores it in a new column 'logResponseTime'.

    Input:
        data: Dataframe containing 311 service request records.

    Returns:
        Cleaned and feature-engineered dataframe with 'logResponseTime' as the target variable.
    """

    df = data.copy()

    df = df.dropna(subset=['Created Date', 'Closed Date'])
    # Convert string columns to datetime
    df['Created Date'] = pd.to_datetime(df['Created Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    df = df.sort_values(by='Created Date')
    sixth_size = len(df) // 4
    df = df.iloc[-sixth_size:]

    df['Closed Date'] = pd.to_datetime(df['Closed Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    # Calculate response time in hours
    df['response_time'] = (df['Closed Date'] - df['Created Date']).dt.total_seconds() / 3600

    # Preprocess time-related features
    df = preprocessTime(df, 'Created Date')
    df = preprocessTime(df, 'Closed Date')
    df = addHolidays(df, 'Created Date')

    # Convert latitude/longitude into cyclic features
    df['lat_sin'] = np.sin(np.radians(df['Latitude']))
    df['lat_cos'] = np.cos(np.radians(df['Latitude']))
    df['lon_sin'] = np.sin(np.radians(df['Longitude']))
    df['lon_cos'] = np.cos(np.radians(df['Longitude']))

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Filter non-negative response times
    df = df[df['response_time'] >= 0]

    # Remove outliers
    cleanDf, _, _, _ = removeOutliers(df, 'response_time')

    # Apply log transformation
    cleanDf['logResponseTime'] = np.log1p(cleanDf['response_time'])

    return cleanDf
