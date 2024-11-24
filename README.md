# Econometria
Este projeto contém uma classe Python para realizar análises econométricas. Tais como regressões lineares, cálculos de estatísticas t, p-valores, e visualizações da distribuição t.

## Exemplo de Uso
```python
from econometria import Econometria
import wooldridge as woo

df = woo.dataWoo('wage1')

econometria = Econometria()

# Sumário da Regressão Linear
print(econometria.ajustar_ols('wage ~ educ + exper', data=df).summary()) 

# Significância estatística do coeficiente 'educ' dentro da regressão.
econometria.resumo_completo('wage ~ educ', data=df, coef_nome='educ', alpha=0.05, alternative='two-sided')
