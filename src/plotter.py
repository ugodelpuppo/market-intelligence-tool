import matplotlib.pyplot as plt
import pandas as pd


def plot_price_and_moving_averages(data: pd.DataFrame, ticker: str) -> None:
    """
    Affiche le prix de clôture et les moyennes mobiles.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close")
    
    if "MA_20" in data.columns:
        plt.plot(data.index, data["MA_20"], label="MA 20")
    if "MA_50" in data.columns:
        plt.plot(data.index, data["MA_50"], label="MA 50")

    plt.title(f"{ticker} - Prix de clôture et moyennes mobiles")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_return_distribution(returns: pd.Series, ticker: str, 
                              var_95: float, cvar_95: float) -> None:
    """
    Affiche la distribution des rendements avec VaR et CVaR.
    """
    plt.figure(figsize=(12, 6))
    
    ## Histogramme des rendements
    plt.hist(returns, bins=50, edgecolor="white", color="steelblue", alpha=0.7)
    
    ## Ligne verticale VaR 95%
    plt.axvline(x=var_95, color="red", linestyle="--", label=f"VaR 95% : {var_95*100:.2f}%")
    
    ## Ligne verticale CVaR 95%
    plt.axvline(x=cvar_95, color="darkred", linestyle="--", label=f"CVaR 95% : {cvar_95*100:.2f}%")
    
    plt.title(f"{ticker} - Distribution des rendements journaliers")
    plt.xlabel("Rendement journalier")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(prices: pd.Series, ticker: str) -> None:
    """
    Affiche le drawdown dans le temps.
    """
    plt.figure(figsize=(12, 6))

    ## Calcul du pic cumulatif et du drawdown
    peak = prices.cummax()
    drawdown = (prices - peak) / peak

    ## Tracé du drawdown
    plt.plot(prices.index, drawdown, color="red", label="Drawdown")
    
    ## Zone colorée sous la courbe
    plt.fill_between(prices.index, drawdown, 0, alpha=0.3, color="red")

    plt.title(f"{ticker} - Drawdown dans le temps")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier(stats_1: dict, stats_2: dict,
                             correlation: float,
                             ticker_1: str, ticker_2: str) -> None:
    """
    Trace la frontière efficiente de Markowitz pour deux actifs.
    Chaque point = un portefeuille avec une pondération différente.
    La courbe montre le couple rendement/risque de 1000 portefeuilles simulés.
    """
    import numpy as np
    from indicators import compute_portfolio_metrics

    ## 1000 pondérations entre 0% et 100%
    weights = np.linspace(0, 1, 1000)

    returns_list = []
    volatility_list = []

    ## Calcul rendement/volatilité pour chaque pondération
    for w1 in weights:
        w2 = 1 - w1
        portfolio = compute_portfolio_metrics(stats_1, stats_2, correlation, w1, w2)
        returns_list.append(portfolio["return"])
        volatility_list.append(portfolio["volatility"])

    plt.figure(figsize=(10, 6))

    ## Frontière efficiente — couleur selon le rendement
    plt.scatter(volatility_list, returns_list, c=returns_list, cmap="viridis", s=10)
    plt.colorbar(label="Rendement attendu")

    ## Points individuels des deux actifs
    plt.scatter(stats_1["std_annual"], stats_1["mean_annual"],
                color="red", s=100, zorder=5, label=ticker_1)
    plt.scatter(stats_2["std_annual"], stats_2["mean_annual"],
                color="blue", s=100, zorder=5, label=ticker_2)

    plt.title("Frontière Efficiente de Markowitz")
    plt.xlabel("Volatilité annualisée")
    plt.ylabel("Rendement annualisé")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    """
    Affiche la matrice de corrélation entre N actifs.
    Rouge = corrélation négative, Bleu = corrélation positive.
    Diagonale toujours = 1 (corrélation d'un actif avec lui-même).
    """
    import numpy as np

    ## Calcul de la matrice de corrélation
    corr_matrix = returns.corr()

    plt.figure(figsize=(8, 6))

    ## Affichage de la heatmap
    plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Corrélation")

    ## Labels des axes
    tickers = corr_matrix.columns.tolist()
    plt.xticks(range(len(tickers)), tickers)
    plt.yticks(range(len(tickers)), tickers)

    ## Valeurs dans chaque cellule
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            plt.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}",
                    ha="center", va="center", color="black")

    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()

def plot_var_backtest(backtest: pd.DataFrame, ticker: str) -> None:
    """
    Affiche le backtesting de la VaR.
    Ligne bleue = rendements réalisés
    Ligne rouge = VaR estimée
    Points rouges = exceptions (violations de la VaR)
    """
    plt.figure(figsize=(12, 6))

    ## Rendements réalisés
    plt.plot(backtest.index, backtest["Realized"], 
             color="steelblue", alpha=0.7, label="Rendement réalisé")

    ## VaR estimée
    plt.plot(backtest.index, backtest["VaR"], 
             color="red", linewidth=1.5, label="VaR 95%")

    ## Exceptions — points où le rendement viole la VaR
    exceptions = backtest[backtest["Exception"]]
    plt.scatter(exceptions.index, exceptions["Realized"],
                color="red", zorder=5, s=30, label=f"Exceptions : {len(exceptions)}")

    ## Ligne zéro
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.title(f"{ticker} - Backtesting VaR 95%")
    plt.xlabel("Date")
    plt.ylabel("Rendement journalier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_monte_carlo(returns: pd.DataFrame, optimal: dict,
                     tickers: list, n_portfolios: int = 5000,
                     risk_free_rate: float = 0.03) -> None:
    """
    Simule n_portfolios portefeuilles aléatoires et trace le nuage.
    Couleur = Sharpe Ratio. Point rouge étoile = portefeuille optimal.
    """
    import numpy as np
    from indicators import compute_portfolio_metrics_multi

    n = len(tickers)

    port_returns = []
    port_vols = []
    port_sharpes = []

    for _ in range(n_portfolios):
        ## Poids aléatoires normalisés — somme = 1
        weights = np.random.random(n)
        weights = weights / weights.sum()

        ## Métriques du portefeuille
        port = compute_portfolio_metrics_multi(returns, weights)
        sharpe = (port["return"] - risk_free_rate) / port["volatility"]

        port_returns.append(port["return"])
        port_vols.append(port["volatility"])
        port_sharpes.append(sharpe)

    plt.figure(figsize=(10, 6))

    ## Nuage de portefeuilles coloré par Sharpe
    sc = plt.scatter(port_vols, port_returns,
                     c=port_sharpes, cmap="viridis",
                     alpha=0.5, s=10)
    plt.colorbar(sc, label="Sharpe Ratio")

    ## Portefeuille optimal — étoile rouge
    plt.scatter(optimal["volatility"], optimal["return"],
                color="red", s=200, zorder=5,
                marker="*", label="Optimal Sharpe")

    plt.title("Simulation Monte-Carlo — Portefeuilles aléatoires")
    plt.xlabel("Volatilité annualisée")
    plt.ylabel("Rendement annualisé")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()