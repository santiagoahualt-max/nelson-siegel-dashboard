# Nelson-Siegel Yield Curve & Z-Score Dashboard

Este proyecto es una herramienta analítica interactiva y ligera diseñada para estimar la curva de rendimientos soberanos/corporativos utilizando el modelo de **Nelson-Siegel (1987)**, calcular los **Z-scores** de los activos respecto a la curva teórica y generar señales de arbitraje (barato/caro) en puntos básicos (basis points).

---

## 🎯 Contexto Financiero y Objetivos

El objetivo principal es identificar oportunidades de **Relative Value** (valor relativo) en el mercado de renta fija:
1. **$\tau$ (plazo)**: Representado por la **Modified Duration** (`MDuration`), ya que refleja la sensibilidad del precio del bono ante cambios en la tasa.
2. **$y(\tau)$ (rendimiento)**: Representado por la **TIR** (Tasa Interna de Retorno anualizada o TEA).
3. **Z-Score**:
   - Calcula el desvío estándar de los residuos (TIR Real - TIR Teórica).
   - Un residuo positivo alto ($Z > 1.5$) indica un bono **Barato** (rinde más de lo que debería para su duración).
   - Un residuo negativo bajo ($Z < -1.5$) indica un bono **Caro** (rinde menos de lo que debería para su duración).

---

## 📐 Formulación Matemática y Ajuste

### 1. Modelo de Nelson-Siegel
La tasa teórica $y(\tau)$ para un plazo $\tau$ se modela como:

$$y(\tau) = \beta_0 + \beta_1 \left( \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} \right) + \beta_2 \left( \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda} \right)$$

Donde:
* **$\beta_0$ (Nivel)**: Tasa de largo plazo. Representa la asíntota de la curva.
* **$\beta_1$ (Pendiente)**: Componente de corto plazo. Determina si la curva es normal (negativo) o invertida (positivo).
* **$\beta_2$ (Curvatura)**: Componente de mediano plazo. Modela la joroba o panza de la curva.
* **$\lambda$ (Decaimiento)**: Determina la velocidad de decaimiento y la posición del pico de la curvatura.

### 2. Algoritmo de Optimización (SciPy)
Para garantizar estabilidad numérica y consistencia financiera, el ajuste en Python (`app.py` / `nelson_siegel_analysis.py`) utiliza una aproximación híbrida:
1. **Grid Search + OLS**: Se realiza una búsqueda lineal del parámetro no lineal $\lambda$ en el rango $[0.1, 15.0]$. Para cada $\lambda$, se resuelve de forma exacta el vector $\beta$ óptimo mediante mínimos cuadrados ordinarios (OLS).
2. **SLSQP Constrained Optimization**: A partir del mejor punto encontrado, se refina el ajuste utilizando el método `SLSQP` para poder aplicar la siguiente restricción indispensable:
   $$\beta_0 + \beta_1 \ge 0.0165\%$$
   * **Restricción de No-Negatividad**: Evita que la tasa de interés a plazo cero ($y(0) = \beta_0 + \beta_1$) caiga en terreno negativo, lo cual ocurriría con frecuencia al no tener bonos de muy corto plazo (duración < 1 año).

---

## 🗂️ Arquitectura del Proyecto

```
📁 / (Raíz del proyecto)
├── 📄 app.py                  # Backend Flask: limpia datos de Excel en memoria, corre el fit y provee API JSON.
├── 📄 index.html              # Frontend SPA: Dashboard oscuro, gráficos Chart.js y tabla interactiva ordenable.
├── 📄 nelson_siegel_analysis.py # Script CLI: Corre el fit autónomamente y guarda reporte físico y gráfico en disco.
├── 📄 requirements.txt        # Dependencias de Python (NumPy, Pandas, SciPy, Matplotlib, Flask, Gunicorn).
├── 📄 .gitignore              # Excluye cachés de Python y datos locales del control de versiones.
├── 📄 input_data.txt          # Caché del último input pegado (Tab-separated).
└── 📄 output_data.csv         # Reporte delimitado por ";" exportado automáticamente, amigable con Excel en español.
```

---

## 🛠️ Guía para Futuros Desarrolladores e Inteligencias Artificiales

Si vas a modificar o extender esta base de código, ten en cuenta las siguientes directrices clave:

### 1. Robustez en la Limpieza de Datos (Parser)
El script acepta copias directas de celdas de Excel (`Ctrl+C` / `Ctrl+V`). El formateo en español de Excel suele usar comas como separadores decimales (ej: `6,83%` o `4,49`) y símbolos de moneda o porcentajes. 
* *Regla*: Mantener siempre la función `clean_value(val)` que normaliza estos caracteres a flotantes válidos antes de la optimización.
* *Prioridad de Columnas*: Si existen columnas llamadas tanto `Duration` (Macaulay) como `MDuration` (Modified), el script debe priorizar `MDuration` (o patrones con la letra "M") para el eje X, tal como requiere la teoría de Relative Value.

### 2. Evitar el Hito Artificial del Corto Plazo (Zoom de Ejes)
Dado que las curvas corporativas suelen carecer de bonos con duraciones menores a 1.5 años, la curva teórica graficada desde 0 tiende a mostrar formas erráticas en el tramo $[0, 1]$.
* *Regla*: El gráfico del frontend (`index.html`) implementa la opción **"Acotar a Duración de Bonos"** activa por defecto. Esto calcula dinámicamente los límites del eje X:
  $$X_{\min} = \max(0, \min(\text{duraciones}) - 0.5)$$
  Evita remover esta opción para prevenir visualizaciones erróneas.

### 3. Mantener Compatibilidad con Excel Local
El archivo `output_data.csv` se exporta usando el separador punto y coma (`;`) y la codificación `utf-8-sig`. 
* *Regla*: No cambies el delimitador a coma (`,`), ya que las versiones en español de Excel interpretarían las comas internas de los decimales de forma incorrecta, rompiendo la estructura de columnas al hacer doble clic sobre el archivo.

---

## 🗺️ Hoja de Ruta de Próximas Versiones (Roadmap)

Sugerencias de valor agregado para futuros sprints de desarrollo:

### Fase 1: Extensión del Modelo Matemático
* **Nelson-Siegel-Svensson (NSS)**:
  Expandir el modelo incorporando un término adicional de curvatura ($\beta_3$ y un segundo decay $\lambda_2$). Esto es muy útil si se agregan activos a más de 10 años de duración, ya que permite modelar dos curvas/jorobas de rendimiento independientes.
* **Ajuste con Pesos (Weighted Least Squares - WLS)**:
  Permitir ponderar los bonos en el ajuste de la curva en función de su liquidez (volumen operado o bid-ask spread). Los bonos más líquidos deberían tener un residuo menor (forzar a la curva a pasar más cerca de ellos).

### Fase 2: Automatización de Inputs
* **Conexión a Web APIs**:
  En lugar de pegar manualmente los datos de Excel, integrar un endpoint que consulte APIs locales (ej: APIs de AlyCs locales, Yahoo Finance, o plataformas de BYMA) para actualizar la curva en tiempo real cada 5 minutos.

### Fase 3: Analítica Avanzada en Interfaz
* **Calculadora de Cobertura (Duration Hedging)**:
  Agregar en el panel desplegable de la tabla una calculadora interactiva. Si un usuario tiene una posición larga en un bono "barato" (BUY), permitirle calcular qué cantidad nominal de un bono "caro" (SELL) o futuro de tasa debe vender en corto para inmunizar la duración de la cartera.
