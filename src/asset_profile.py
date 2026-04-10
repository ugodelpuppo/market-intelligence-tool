import yfinance as yf


def get_asset_profile(ticker: str) -> dict:
    """
    Récupère quelques informations de base sur l'actif.
    """
    asset = yf.Ticker(ticker)
    info = asset.info

    profile = {
        "name": info.get("shortName", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "country": info.get("country", "N/A"),
        "currency": info.get("currency", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "business_summary": info.get("longBusinessSummary", "N/A"),
    }

    return profile