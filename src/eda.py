import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_temperature_distribution(df, figures_dir):
    """Plot global temperature distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['temperature_celsius'].dropna(), bins=60,
            color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(df['temperature_celsius'].mean(), color='red',
               linestyle='--', label=f"Mean: {df['temperature_celsius'].mean():.1f}°C")
    ax.set_title('Global Temperature Distribution', fontweight='bold')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / 'temp_distribution.png', dpi=150)
    plt.close()


def plot_correlation_heatmap(df, cols, figures_dir):
    """Plot correlation heatmap for selected columns."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / 'correlation_heatmap.png', dpi=150)
    plt.close()


def get_country_stats(df):
    """Return average weather stats per country."""
    return df.groupby('country').agg(
        avg_temp    =('temperature_celsius', 'mean'),
        avg_humidity=('humidity',            'mean'),
        avg_precip  =('precip_mm',           'mean'),
        avg_wind    =('wind_kph',            'mean'),
        count       =('temperature_celsius', 'count')
    ).round(2).sort_values('avg_temp', ascending=False)


def get_continent_stats(df):
    """Return average weather stats per continent."""
    return df.groupby('continent').agg(
        avg_temp    =('temperature_celsius', 'mean'),
        avg_humidity=('humidity',            'mean'),
        avg_precip  =('precip_mm',           'mean'),
        count       =('temperature_celsius', 'count')
    ).round(2).sort_values('avg_temp', ascending=False)