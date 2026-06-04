import pandas as pd
import numpy as np


CONTINENT_MAP = {
    'Afghanistan':'Asia','Albania':'Europe','Algeria':'Africa',
    'Angola':'Africa','Argentina':'South America','Armenia':'Asia',
    'Australia':'Oceania','Austria':'Europe','Azerbaijan':'Asia',
    'Bahrain':'Asia','Bangladesh':'Asia','Belarus':'Europe',
    'Belgium':'Europe','Bolivia':'South America','Brazil':'South America',
    'Bulgaria':'Europe','Cambodia':'Asia','Cameroon':'Africa',
    'Canada':'North America','Chile':'South America','China':'Asia',
    'Colombia':'South America','Croatia':'Europe','Cuba':'North America',
    'Cyprus':'Europe','Czech Republic':'Europe','Denmark':'Europe',
    'Dominican Republic':'North America','Ecuador':'South America',
    'Egypt':'Africa','Estonia':'Europe','Ethiopia':'Africa',
    'Finland':'Europe','France':'Europe','Georgia':'Asia',
    'Germany':'Europe','Ghana':'Africa','Greece':'Europe',
    'Guatemala':'North America','Honduras':'North America',
    'Hungary':'Europe','India':'Asia','Indonesia':'Asia',
    'Iran':'Asia','Iraq':'Asia','Ireland':'Europe','Israel':'Asia',
    'Italy':'Europe','Japan':'Asia','Jordan':'Asia',
    'Kazakhstan':'Asia','Kenya':'Africa','Kuwait':'Asia',
    'Latvia':'Europe','Lebanon':'Asia','Libya':'Africa',
    'Lithuania':'Europe','Malaysia':'Asia','Mexico':'North America',
    'Morocco':'Africa','Nepal':'Asia','Netherlands':'Europe',
    'New Zealand':'Oceania','Nigeria':'Africa','Norway':'Europe',
    'Oman':'Asia','Pakistan':'Asia','Peru':'South America',
    'Philippines':'Asia','Poland':'Europe','Portugal':'Europe',
    'Qatar':'Asia','Romania':'Europe','Russia':'Europe',
    'Saudi Arabia':'Asia','Serbia':'Europe','Singapore':'Asia',
    'South Africa':'Africa','South Korea':'Asia','Spain':'Europe',
    'Sri Lanka':'Asia','Sweden':'Europe','Switzerland':'Europe',
    'Syria':'Asia','Taiwan':'Asia','Thailand':'Asia',
    'Tunisia':'Africa','Turkey':'Asia','Ukraine':'Europe',
    'United Arab Emirates':'Asia','United Kingdom':'Europe',
    'United States':'North America','Vietnam':'Asia',
}

COLS_TO_DROP = [
    'temperature_fahrenheit', 'wind_mph', 'pressure_in',
    'precip_in', 'feels_like_fahrenheit', 'visibility_miles',
    'gust_mph', 'last_updated_epoch',
]

NUMERIC_COLS = [
    'temperature_celsius', 'humidity', 'pressure_mb',
    'wind_kph', 'precip_mm', 'visibility_km', 'uv_index',
    'air_quality_PM2.5', 'air_quality_PM10'
]


def load_raw(path):
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")
    return df


def parse_datetime(df):
    """Parse last_updated and extract time features."""
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['year']         = df['last_updated'].dt.year
    df['month']        = df['last_updated'].dt.month
    df['day']          = df['last_updated'].dt.day
    df['hour']         = df['last_updated'].dt.hour
    df['season']       = df['month'].map({
        12:'Winter', 1:'Winter', 2:'Winter',
        3:'Spring',  4:'Spring', 5:'Spring',
        6:'Summer',  7:'Summer', 8:'Summer',
        9:'Autumn',  10:'Autumn',11:'Autumn'
    })
    return df


def drop_redundant_cols(df):
    """Drop imperial/duplicate columns."""
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])


def cap_outliers(df, cols=None):
    """Cap outliers using IQR winsorizing."""
    if cols is None:
        cols = NUMERIC_COLS
    for col in cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
    return df


def add_continent(df):
    """Map countries to continents."""
    df['continent'] = df['country'].map(CONTINENT_MAP).fillna('Other')
    return df


def clean_pipeline(path):
    """Run full cleaning pipeline and return cleaned DataFrame."""
    df = load_raw(path)
    df = df.drop_duplicates()
    df = parse_datetime(df)
    df = drop_redundant_cols(df)
    df = cap_outliers(df)
    df = add_continent(df)
    print(f"Cleaning complete: {df.shape}")
    return df