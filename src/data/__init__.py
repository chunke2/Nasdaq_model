"""NASDAQ Event-Factor Model — data layer.

Submodules:
- fetcher: abstract base for data fetching
- cache: local Parquet-based caching
- config_loader: load settings.yaml and secrets.yaml
- price: price/volume data via yfinance
- alpha_vantage: price data via Alpha Vantage API (free: 25 calls/day, 100 data points)
- polygon: price + earnings via Polygon.io API (free: 5 calls/min, 2yr lookback)
- fred: macro data via FRED API (free: 120 calls/min, unlimited)
- synthetic: offline synthetic price data for development/testing
- pipeline: unified data pipeline with fallback chain
"""

from src.data.alpha_vantage import AlphaVantageFetcher
from src.data.polygon import PolygonFetcher
from src.data.fred import FREDFetcher
from src.data.synthetic import SyntheticPriceFetcher
from src.data.pipeline import DataPipeline

__all__ = [
    "PriceFetcher",
    "AlphaVantageFetcher",
    "PolygonFetcher",
    "FREDFetcher",
    "SyntheticPriceFetcher",
    "DataPipeline",
]


def __getattr__(name: str):
    """Lazy import for heavy/dependency-sensitive modules."""
    if name == "PriceFetcher":
        from src.data.price import PriceFetcher
        return PriceFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
