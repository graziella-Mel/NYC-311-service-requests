import pandas as pd
import numpy as np
from utils import preprocessTime, addHolidays, removeOutliers


def preprocessDataset(df, sample_frac=0.5):
    """ Preprocess a chunk of data """

    # Randomly sample a fraction of the data
    # df = df.sample(frac=sample_frac, random_state=42)

    # Convert string columns to datetime
    df['Created Date'] = pd.to_datetime(df['Created Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['Closed Date'] = pd.to_datetime(df['Closed Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    # Cutting the data to 6th its size, choosing the latest created service requests
    df = df.sort_values(by='Created Date')
    half_size = len(df) // 6
    df = df.iloc[:half_size]

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
