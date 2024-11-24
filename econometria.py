import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import statsmodels.formula.api as smf
import wooldridge as woo
# ------------------------------------------------

class Econometria:
    def __init__(self):
        self.modelo = None
        self.coeficientes = None
        self.se_coeficientes = None
        self.ddof = None

    # Método para ajustar um modelo OLS
    def ajustar_ols(self, formula, data):
        modelo = smf.ols(formula, data=data).fit()
        self.modelo = modelo
        self.coeficientes = modelo.params
        self.se_coeficientes = modelo.bse
        self.ddof = int(modelo.df_resid)  # Graus de liberdade residuais
        return modelo

    # Método para calcular a estatística t
    def gerar_estat_t(self, beta_hat, c, se_beta_hat):
        t_estat = (beta_hat - c) / se_beta_hat
        return t_estat

    # Método para calcular a estatística t de um coeficiente específico
    def gerar_estat_t_para_coef(self, coef_nome, c=0):
        if not hasattr(self, 'modelo'):
            raise ValueError("Você precisa ajustar um modelo antes de calcular estatísticas t.")
        beta_hat = self.coeficientes[coef_nome]
        se_beta_hat = self.se_coeficientes[coef_nome]
        return self.gerar_estat_t(beta_hat=beta_hat, c=c, se_beta_hat=se_beta_hat)

    # Método para calcular o p-valor
    def calcular_p_valor(self, coef_nome, alternative='two-sided'):
        estat_t = self.gerar_estat_t_para_coef(coef_nome)
        ddof = self.ddof
        t_dist = t(ddof)
        if alternative == 'two-sided':
            p_value = 2 * (1 - t_dist.cdf(abs(estat_t)))
        elif alternative == 'greater':
            p_value = 1 - t_dist.cdf(estat_t)
        elif alternative == 'less':
            p_value = t_dist.cdf(estat_t)
        return p_value

    # Método para plotar a distribuição t
    def plot_t(self, alpha, ddof, estat_t, alternative='two-sided'):
        x_min = min(-4, estat_t - 0.5)
        x_max = max(4, estat_t + 0.5)
        x = np.linspace(x_min, x_max, 1000)
        t_dist = t(ddof)

        if alternative == 'two-sided':
            t_crit = t.ppf(1 - alpha / 2, ddof)
        else:
            t_crit = t.ppf(1 - alpha, ddof)

        plt.figure(figsize=(12, 6))
        plt.plot(x, t_dist.pdf(x), 'b-', lw=2, label='Distribuição t')

        if alternative == 'two-sided':
            x_rej_right = np.linspace(t_crit, x_max, 100)
            x_rej_left = np.linspace(x_min, -t_crit, 100)
            plt.fill_between(x_rej_right, t_dist.pdf(x_rej_right), color='red', alpha=0.3, label=f'Região de rejeição (α={alpha})')
            plt.fill_between(x_rej_left, t_dist.pdf(x_rej_left), color='red', alpha=0.3)
        elif alternative == 'greater':
            x_rej_right = np.linspace(t_crit, x_max, 100)
            plt.fill_between(x_rej_right, t_dist.pdf(x_rej_right), color='red', alpha=0.3, label=f'Região de rejeição (α={alpha})')
        elif alternative == 'less':
            x_rej_left = np.linspace(x_min, t.ppf(alpha, ddof), 100)
            plt.fill_between(x_rej_left, t_dist.pdf(x_rej_left), color='red', alpha=0.3, label=f'Região de rejeição (α={alpha})')

        plt.axvline(x=estat_t, color='g', linestyle='--', label='Estatística t')
        if alternative == 'two-sided':
            plt.axvline(x=t_crit, color='r', linestyle='-', label='Valor crítico')
            plt.axvline(x=-t_crit, color='r', linestyle='-')
        elif alternative == 'greater':
            plt.axvline(x=t_crit, color='r', linestyle='-', label='Valor crítico')
        elif alternative == 'less':
            plt.axvline(x=t.ppf(alpha, ddof), color='r', linestyle='-', label='Valor crítico')

        if alternative == 'two-sided':
            p_value = 2 * (1 - t_dist.cdf(abs(estat_t)))
        elif alternative == 'greater':
            p_value = 1 - t_dist.cdf(estat_t)
        elif alternative == 'less':
            p_value = t_dist.cdf(estat_t)

        plt.text(0, 0.2, f'p-valor = {p_value:.4f}', horizontalalignment='center')
        plt.text(estat_t, 0.05, f't = {estat_t:.3f}', horizontalalignment='left')
        if alternative == 'two-sided':
            plt.text(t_crit, 0.05, f't_crit = ±{t_crit:.3f}', horizontalalignment='left')
        elif alternative in ['greater', 'less']:
            plt.text(t_crit, 0.05, f't_crit = {t_crit:.3f}', horizontalalignment='left')

        plt.title(f'Distribuição t, Regiões de Rejeição (α={alpha}, alternative={alternative}) e Estatística t')
        plt.xlabel('Valor t')
        plt.ylabel('Densidade')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # Método para plotar a distribuição t para um coeficiente específico
    def plot_t_para_coef(self, coef_nome, alpha=0.05, alternative='two-sided'):
        if not hasattr(self, 'modelo'):
            raise ValueError("Você precisa ajustar um modelo antes de gerar gráficos.")
        estat_t = self.gerar_estat_t_para_coef(coef_nome)
        self.plot_t(alpha=alpha, ddof=self.ddof, estat_t=estat_t, alternative=alternative)

    # Método para gerar um resumo completo com gráficos
    def resumo_completo(self, formula, data, coef_nome, alpha=0.05, alternative='two-sided'):
        self.ajustar_ols(formula, data)
        estat_t = self.gerar_estat_t_para_coef(coef_nome)
        p_valor = self.calcular_p_valor(coef_nome, alternative)
        print(f"Coeficiente: {coef_nome}")
        print(f"Estatística t: {estat_t:.3f}")
        print(f"P-valor: {p_valor:.4f}")
        self.plot_t_para_coef(coef_nome, alpha=alpha, alternative=alternative)

# ------------------------------------------------

df = woo.dataWoo('wage1')
econometria = Econometria()
econometria.resumo_completo('wage ~ educ', data=df, coef_nome='educ', alpha=0.05, alternative='two-sided')