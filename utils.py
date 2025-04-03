import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import pytz

def preprocessTime(df, column):
    """Extracts additional time-based features from a datetime column."""
    df['hour '+ column] = df[column].dt.hour
    df['dayOfWeek '+ column] = df[column].dt.dayofweek
    df['month '+ column] = df[column].dt.month
    df['year '+column] = df[column].dt.year
    df['weekday ' + column] = (df[column].dt.dayofweek < 5).astype(int)
    df['time_of_day '+ column] = np.where((df['hour Created Date'] >= 6) & (df['hour Created Date'] < 18), 1, 0)
    return df


def removeOutliers(df, column, threshold=1.5):
    """Removes outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lowerBound = Q1 - threshold * IQR
    upperBound = Q3 + threshold * IQR
    outliers = df[(df[column] < lowerBound) | (df[column] > upperBound)]
    cleanDf = df[(df[column] >= lowerBound) & (df[column] <= upperBound)]
    return cleanDf, outliers, lowerBound, upperBound


def addHolidays(df, column, tz=pytz.timezone('US/Pacific')):

    # Get US Holidays for given range
    cal = USFederalHolidayCalendar()
    Fed_holidays = cal.holidays('1-1-2010', '12-31-2027').date

    df['Fed Holiday'] = 0
    df[f'Holiday {column}'] = 0

    # Check if a holiday
    df['Fed Holiday'] = [x.date() in Fed_holidays for x in df[column]]

    # Combine both list of holidays
    df.loc[(df[f'Holiday {column}'] == True) | (df['Fed Holiday'] == True), f'Holiday {column}'] = 1

    # Delete temp columns
    del df['Fed Holiday']

    return df