import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"Aucune donnée récupérée pour le ticker '{ticker}'.")

    # Nettoyage des colonnes multi-index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 🔥 SUPPRIME le nom des colonnes (le "Price")
    data.columns.name = None

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.dropna(inplace=True)

    return data

def load_multiple_prices(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Charge les prix de clôture de N actifs.
    Retourne un DataFrame avec une colonne par ticker.
    """
    closes = {}
    for ticker in tickers:
        data = load_price_data(ticker)
        closes[ticker] = data["Close"]

    ## Alignement sur les mêmes dates — dropna supprime les jours manquants
    return pd.DataFrame(closes).dropna()