import pandas as pd


## ─── RENDEMENTS ────────────────────────────────────────────────────────────────

def add_daily_return(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne de rendement journalier arithmétique.
    Formule : (Pt - Pt-1) / Pt-1
    """
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change()
    return df


## ─── STATISTIQUES DESCRIPTIVES ─────────────────────────────────────────────────

def compute_return_statistics(data: pd.DataFrame) -> dict:
    """
    Calcule les statistiques de base des rendements journaliers.
    La volatilité est annualisée par multiplication par √252
    (252 = nombre de jours ouvrés par an, hypothèse : rendements i.i.d.)
    """
    returns = data["Daily Return"].dropna()

    mean_daily = returns.mean()
    std_daily = returns.std()

    stats = {
        "mean_daily": mean_daily,
        "mean_annual": mean_daily * 252,        ## Rendement annualisé
        "variance": returns.var(),              ## Variance journalière
        "std_daily": std_daily,                 ## Volatilité journalière
        "std_annual": std_daily * (252 ** 0.5), ## Volatilité annualisée
        "skewness": returns.skew(),             ## Asymétrie de la distribution
        "kurtosis": returns.kurt(),             ## Épaisseur des queues de distribution
        "min_return": returns.min(),            ## Pire journée
        "max_return": returns.max(),            ## Meilleure journée
    }

    return stats


## ─── CORRÉLATION & COVARIANCE ──────────────────────────────────────────────────

def compute_correlation(data1: pd.DataFrame, data2: pd.DataFrame) -> dict:
    """
    Calcule la corrélation et la covariance entre deux actifs.
    Corrélation = mesure de co-mouvement (entre -1 et +1)
    Covariance = corrélation × σ1 × σ2 (non normalisée)
    """
    returns1 = data1["Daily Return"]
    returns2 = data2["Daily Return"]

    ## Alignement des deux séries sur les mêmes dates
    combined = returns1.to_frame("r1").join(returns2.to_frame("r2")).dropna()

    correlation = combined["r1"].corr(combined["r2"])
    covariance = combined["r1"].cov(combined["r2"])

    return {
        "correlation": correlation,
        "covariance": covariance
    }


## ─── PORTEFEUILLE ───────────────────────────────────────────────────────────────

def compute_portfolio_metrics(stats_1: dict, stats_2: dict, correlation: float,
                               w1: float = 0.5, w2: float = 0.5) -> dict:
    """
    Calcule le rendement et la volatilité d'un portefeuille à deux actifs.
    Formule variance : w1²σ1² + w2²σ2² + 2·w1·w2·σ1·σ2·ρ
    Le terme 2·w1·w2·σ1·σ2·ρ représente l'effet de diversification.
    """
    r1 = stats_1["mean_annual"]
    r2 = stats_2["mean_annual"]

    vol1 = stats_1["std_annual"]
    vol2 = stats_2["std_annual"]

    ## Rendement attendu = moyenne pondérée des rendements individuels
    portfolio_return = w1 * r1 + w2 * r2

    ## Volatilité du portefeuille (formule de Markowitz)
    portfolio_vol = (
        (w1**2 * vol1**2) +
        (w2**2 * vol2**2) +
        (2 * w1 * w2 * vol1 * vol2 * correlation)
    ) ** 0.5

    return {
        "return": portfolio_return,
        "volatility": portfolio_vol
    }

def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les moyennes mobiles sur 20 et 50 jours.
    MM20 = tendance court terme (1 mois)
    MM50 = tendance moyen terme (2,5 mois)
    Croisement MM20/MM50 = signal haussier ou baissier
    """
    df = data.copy()
    df["MA_20"] = df["Close"].rolling(window=20).mean()  ## Moyenne mobile 20 jours
    df["MA_50"] = df["Close"].rolling(window=50).mean()  ## Moyenne mobile 50 jours
    return df

## ─── MESURES DE RISQUE ──────────────────────────────────────────────────────────

def compute_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Value at Risk historique.
    Répond à : "quelle est la perte maximale dans X% des cas ?"
    Méthode : lecture directe du quantile sur la distribution historique.
    Ex : VaR 95% = quantile à 5% des rendements (queue gauche)
    """
    ## 1 - confidence_level = 0.05 pour un niveau de confiance à 95%
    return returns.quantile(1 - confidence_level)


def compute_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall (ES).
    Répond à : "en moyenne, combien perd-on dans les cas extrêmes ?"
    Métrique imposée par Bâle III — capture la sévérité des pertes,
    là où la VaR s'arrête au seuil.
    """
    ## Seuil VaR = frontière de la queue de distribution
    var = compute_var(returns, confidence_level)

    ## On filtre uniquement les rendements en dessous du seuil VaR
    tail = returns[returns < var]

    ## Moyenne des pertes extrêmes
    return tail.mean()


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Sharpe Ratio = rendement ajusté du risque.
    Formule : (Rp - Rf) / σp
    Rf = taux sans risque (ex: OAT 10 ans, ~3% en zone euro)
    Plus le Sharpe est élevé, meilleur est le rendement par unité de risque.
    """
    R = returns.mean() * 252          ## Rendement annualisé
    V = returns.std() * 252 ** 0.5    ## Volatilité annualisée
    S = (R - risk_free_rate) / V      ## Sharpe Ratio
    return S


def compute_max_drawdown(prices: pd.Series) -> float:
    """
    Max Drawdown = perte maximale entre un pic et un creux successif.
    Formule : (Creux - Pic) / Pic
    Exprimé en négatif (c'est une perte).
    Attention : le creux doit arriver APRÈS le pic — on utilise cummax()
    pour calculer le pic cumulatif à chaque instant.
    """
    peak = prices.cummax()        ## Pic cumulatif à chaque date
    D = (prices - peak) / peak    ## Drawdown à chaque instant
    return D.min()                ## Pire drawdown sur la période

def compute_portfolio_metrics_multi(returns: pd.DataFrame,
                                     weights) -> dict:
    """
    Calcule rendement et volatilité d'un portefeuille multi-actifs.
    Utilise l'algèbre matricielle : σ²p = w^T · Σ · w
    weights : vecteur numpy de poids, doit sommer à 1
    """
    import numpy as np

    ## Rendements moyens annualisés de chaque actif
    mean_returns = returns.mean() * 252

    ## Matrice de covariance annualisée
    cov_matrix = returns.cov() * 252

    ## Rendement du portefeuille = somme pondérée des rendements
    portfolio_return = np.dot(weights, mean_returns)

    ## Variance du portefeuille = w^T · Σ · w
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

    ## Volatilité = racine carrée de la variance
    portfolio_vol = np.sqrt(portfolio_variance)

    return {"return": portfolio_return,"volatility": portfolio_vol}

def compute_optimal_sharpe(returns: pd.DataFrame, 
                            risk_free_rate: float = 0.03) -> dict:
    """
    Trouve le portefeuille qui maximise le Sharpe Ratio.
    Utilise scipy.optimize.minimize sur -Sharpe.
    Contraintes : somme des poids = 1, poids >= 0.
    """
    from scipy.optimize import minimize
    import numpy as np

    n = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    ## Fonction à minimiser = -Sharpe
    def neg_sharpe(weights):
        port = compute_portfolio_metrics_multi(returns, weights)
        sharpe = (port["return"] - risk_free_rate) / port["volatility"]
        return -sharpe

    ## Poids initiaux : équipondéré
    w0 = np.array([1/n] * n)

    ## Contraintes
    constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1}]

    ## Bornes : chaque poids entre 0 et 1
    bounds = [(0, 1)] * n

    ## Optimisation
    result = minimize(neg_sharpe, w0, 
                     method="SLSQP",
                     bounds=bounds, 
                     constraints=constraints)

    optimal_weights = result.x
    optimal_portfolio = compute_portfolio_metrics_multi(returns, optimal_weights)

    return {
        "weights": optimal_weights,
        "return": optimal_portfolio["return"],
        "volatility": optimal_portfolio["volatility"],
        "sharpe": (optimal_portfolio["return"] - risk_free_rate) / optimal_portfolio["volatility"]
    }

def compute_var_backtest(returns: pd.Series, 
                          confidence_level: float = 0.95,
                          window: int = 252) -> pd.DataFrame:
    """
    Backtesting de la VaR historique sur fenêtre glissante.
    Pour chaque jour t : calcule la VaR sur les 252 jours précédents,
    compare au rendement réalisé le jour t.
    Exception = rendement réalisé < VaR estimée.
    """
    var_estimates = []
    exceptions = []
    dates = []

    ## Boucle sur chaque jour à partir de la fenêtre initiale
    for i in range(window, len(returns)):
        
        ## Fenêtre glissante : 252 jours précédents
        window_returns = returns.iloc[i-window:i]
        
        ## VaR estimée sur la fenêtre
        var = compute_var(window_returns, confidence_level)
        
        ## Rendement réalisé le jour suivant
        realized = returns.iloc[i]
        
        ## Exception si rendement < VaR
        exception = realized < var
        
        var_estimates.append(var)
        exceptions.append(exception)
        dates.append(returns.index[i])

    results = pd.DataFrame({
        "VaR": var_estimates,
        "Realized": returns.iloc[window:].values,
        "Exception": exceptions
    }, index=dates)

    return results