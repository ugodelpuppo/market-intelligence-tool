import pandas as pd
from data_loader import load_price_data, load_multiple_prices
from indicators import (add_daily_return, compute_return_statistics,
                        compute_correlation, compute_portfolio_metrics,
                        compute_portfolio_metrics_multi, add_moving_averages,
                        compute_var, compute_cvar, compute_sharpe,
                        compute_max_drawdown)
from asset_profile import get_asset_profile
from plotter import (plot_price_and_moving_averages, plot_return_distribution,
                     plot_drawdown, plot_efficient_frontier)
import numpy as np
from plotter import plot_correlation_heatmap
from indicators import compute_optimal_sharpe
from indicators import compute_var_backtest
from plotter import plot_var_backtest
from plotter import plot_monte_carlo

def display_asset_info(profile: dict) -> None:
    print(f"Nom        : {profile['name']}")
    print(f"Secteur    : {profile['sector']}")
    print(f"Industrie  : {profile['industry']}")
    print(f"Pays       : {profile['country']}")
    print(f"Devise     : {profile['currency']}")
    print(f"Market Cap : {profile['market_cap']}")


def display_statistics(ticker: str, stats: dict) -> None:
    print(f"\n=== STATISTIQUES DES RENDEMENTS - {ticker} ===")
    print(f"Rendement journalier moyen : {stats['mean_daily']*100:.4f}%")
    print(f"Rendement annualisé        : {stats['mean_annual']*100:.2f}%")
    print(f"Volatilité journalière     : {stats['std_daily']*100:.4f}%")
    print(f"Volatilité annualisée      : {stats['std_annual']*100:.2f}%")
    print(f"Variance                   : {stats['variance']:.6f}")
    print(f"Skewness                   : {stats['skewness']:.4f}")
    print(f"Kurtosis                   : {stats['kurtosis']:.4f}")
    print(f"Rendement min              : {stats['min_return']*100:.2f}%")
    print(f"Rendement max              : {stats['max_return']*100:.2f}%")


def display_risk_metrics(ticker: str, stats: dict,
                          prices: pd.Series, returns: pd.Series) -> dict:
    print(f"\n=== MÉTRIQUES DE RISQUE - {ticker} ===")

    ## VaR : perte maximale dans 95% et 99% des cas
    var_95 = compute_var(returns, confidence_level=0.95)
    var_99 = compute_var(returns, confidence_level=0.99)
    print(f"VaR 95%  : {var_95*100:.2f}%")
    print(f"VaR 99%  : {var_99*100:.2f}%")

    ## CVaR : perte moyenne dans les cas extrêmes (Bâle III)
    cvar_95 = compute_cvar(returns, confidence_level=0.95)
    cvar_99 = compute_cvar(returns, confidence_level=0.99)
    print(f"CVaR 95% : {cvar_95*100:.2f}%")
    print(f"CVaR 99% : {cvar_99*100:.2f}%")

    ## Sharpe : rendement par unité de risque
    sharpe = compute_sharpe(returns)
    print(f"Sharpe Ratio : {sharpe:.4f}")

    ## Max Drawdown : pire perte pic à creux sur la période
    mdd = compute_max_drawdown(prices)
    print(f"Max Drawdown : {mdd*100:.2f}%")

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "sharpe": sharpe,
        "mdd": mdd
    }


def main() -> None:
    ## Saisie des tickers
    tickers_input = input("Entre les tickers séparés par des virgules (ex: AAPL,MSFT,GOOGL) : ")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    period_input = input("Période d'analyse (ex: 1y, 2y, 5y) : ").strip().lower()

    try:
        ## Chargement des prix multi-actifs
        prices = load_multiple_prices(tickers, period=period_input)

        ## Calcul des rendements journaliers
        returns = prices.pct_change().dropna()

        ## Ajout des moyennes mobiles et stats individuelles
        for ticker in tickers:

            ## Profil de l'actif
            profile = get_asset_profile(ticker)
            print(f"\n=== INFORMATIONS ACTIF - {ticker} ===")
            display_asset_info(profile)

            ## Statistiques
            stats = compute_return_statistics(
                returns[[ticker]].rename(columns={ticker: "Daily Return"}))
            display_statistics(ticker, stats)

            ## Métriques de risque
            risk = display_risk_metrics(
                ticker, stats, prices[ticker], returns[ticker])

            ## Graphiques individuels
            prices_with_ma = add_moving_averages(
                prices[[ticker]].rename(columns={ticker: "Close"}))
            prices_with_ma["Daily Return"] = returns[ticker]

            plot_price_and_moving_averages(prices_with_ma, ticker)
            plot_return_distribution(returns[ticker], ticker,
                                      risk["var_95"], risk["cvar_95"])
            plot_drawdown(prices[ticker], ticker)

            ## Backtesting VaR
            backtest = compute_var_backtest(returns[ticker], window=60)
            if len(backtest) == 0:
                print(f"Pas assez de données pour le backtesting de {ticker}")
            else:
                nb_exceptions = backtest["Exception"].sum()
                print(f"\n=== BACKTESTING VAR 95% - {ticker} ===")
                print(f"Nombre d'exceptions : {nb_exceptions}")
                print(f"Taux d'exceptions   : {nb_exceptions/len(backtest)*100:.2f}%")
                print(f"Seuil Bâle III      : 12 exceptions max sur 252 jours")
                plot_var_backtest(backtest, ticker)

        ## Portefeuille équipondéré
        n = len(tickers)
        weights = np.array([1/n] * n)  ## Poids égaux

        portfolio = compute_portfolio_metrics_multi(returns, weights)

        print("\n=== PORTEFEUILLE ÉQUIPONDÉRÉ ===")
        print(f"Actifs      : {', '.join(tickers)}")
        print(f"Poids       : {[round(1/n*100, 1) for _ in tickers]}%")
        print(f"Rendement   : {portfolio['return']*100:.2f}%")
        print(f"Volatilité  : {portfolio['volatility']*100:.2f}%")

        ## Heatmap de corrélation — pour N actifs
        plot_correlation_heatmap(returns)

        ## Portefeuille optimal Sharpe
        optimal = compute_optimal_sharpe(returns)

        print("\n=== PORTEFEUILLE OPTIMAL (MAX SHARPE) ===")
        for ticker, weight in zip(tickers, optimal["weights"]):
            print(f"{ticker} : {weight*100:.1f}%")
        print(f"Rendement  : {optimal['return']*100:.2f}%")
        print(f"Volatilité : {optimal['volatility']*100:.2f}%")
        print(f"Sharpe     : {optimal['sharpe']:.4f}")

        ## Monte-Carlo
        plot_monte_carlo(returns, optimal, tickers)

        ## Frontière efficiente — seulement pour 2 actifs
        if len(tickers) == 2:
            stats_0 = compute_return_statistics(
                returns[[tickers[0]]].rename(columns={tickers[0]: "Daily Return"}))
            stats_1 = compute_return_statistics(
                returns[[tickers[1]]].rename(columns={tickers[1]: "Daily Return"}))
            corr = returns[tickers[0]].corr(returns[tickers[1]])
            plot_efficient_frontier(stats_0, stats_1, corr, tickers[0], tickers[1])

    except Exception as e:
        print(f"\nErreur : {e}")


if __name__ == "__main__":
    main()