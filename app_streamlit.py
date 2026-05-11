import numpy as np # usada para cálculos
import pandas as pd # usado para a tabela, leitura do CSV e organização dos dados
import streamlit as st # usado para criar a interface visual web
from scipy.stats import gamma, norm # scipy.stats usado para funcoes estatisticas prontas; gamma = distribuicao Gama; norm = distribuicao Normal
import plotly.express as px # utilizado para parte II - gráficos interativos (histograma, boxplot, etc.)

st.title("Distribuição Gama e NHANES")
st.subheader("Visualização de dados com Streamlit")

parte = st.selectbox( # cria uma caixa para utilizador escolher qual pretende visualizar
    "Escolhe a parte do projeto",
    ["Parte I - Distribuição Gama", "Parte II - NHANES"]
)

# PARTE I - Distribuição Gama

if parte == "Parte I - Distribuição Gama":

    st.header("1. Parâmetros")
    st.write("A distribuição Gama é contínua e está definida para valores positivos.")
    st.write("- alpha: controla o formato da distribuição.")
    st.write("- beta: controla a escala.")
    st.write("- seed: fixa a geração aleatória das simulações.")

    #slider para escolher alpha
    alpha = st.slider(
        "Escolhe o valor de alpha",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    #slider para escolher beta
    beta = st.slider(
        "Escolhe o valor de beta",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5
    )
    
    #number_input para escolher seed manualmente
    seed = st.number_input(
        "Escolhe a seed",
        min_value=1,
        max_value=9999,
        value=123,
        step=1
    )

    st.header("2. Medidas teóricas e quartis")
    
    # Calculamos media, variancia e desvio padrao com formulas da distribuicao Gama parametrizada por alpha e beta.
    # Depois usamos gamma.ppf para obter quartis teoricos.
    # ppf = percent point function = quantil inverso.
    media = alpha / beta
    variancia = alpha / (beta ** 2)
    dp = np.sqrt(alpha) / beta

    q1 = gamma.ppf(0.25, a=alpha, scale=1 / beta)
    mediana = gamma.ppf(0.50, a=alpha, scale=1 / beta)
    q3 = gamma.ppf(0.75, a=alpha, scale=1 / beta)

    # organizar os resultados numa tabela
    resumo_gama = pd.DataFrame({
        "Medida": ["Média", "Variância", "Desvio padrão", "Q1", "Mediana", "Q3"],
        "Valor": [media, variancia, dp, q1, mediana, q3]
    })

    st.dataframe(resumo_gama.round(3), hide_index=True)
    
    # Estes gráficos são estáticos para comparação: usamos alpha = 1, 2 e 5, com beta fixo em 1.
    st.header("3. Função de densidade (PDF)")

    x = np.linspace(0, 10, 300)

    alpha1 = 1
    alpha2 = 2
    alpha3 = 5
    beta_pdf = 1

    pdf_df = pd.DataFrame({
        "x": x,
        "alpha = 1": gamma.pdf(x, a=alpha1, scale=1 / beta_pdf),
        "alpha = 2": gamma.pdf(x, a=alpha2, scale=1 / beta_pdf),
        "alpha = 5": gamma.pdf(x, a=alpha3, scale=1 / beta_pdf)
    })

    st.line_chart(pdf_df, x="x", y=["alpha = 1", "alpha = 2", "alpha = 5"])
    st.caption("Com beta fixo, a forma da distribuição altera-se à medida que alpha varia.")

    st.header("4. Função acumulada (CDF)")

    cdf_df = pd.DataFrame({
        "x": x,
        "alpha = 1": gamma.cdf(x, a=alpha1, scale=1 / beta_pdf),
        "alpha = 2": gamma.cdf(x, a=alpha2, scale=1 / beta_pdf),
        "alpha = 5": gamma.cdf(x, a=alpha3, scale=1 / beta_pdf)
    })

    st.line_chart(cdf_df, x="x", y=["alpha = 1", "alpha = 2", "alpha = 5"])
    st.caption("A função acumulada representa a probabilidade acumulada até cada valor de x.")

    st.header("5. Lei dos Grandes Números")

    n_lgn = st.slider(
        "Escolhe o tamanho da amostra (n_lgn)",
        min_value=50,
        max_value=5000,
        value=1000,
        step=50
    )
    
    # gerador aleatorio com seed fixa para reproduzir resultados
    rng_lgn = np.random.default_rng(seed)

    media_teorica = alpha / beta
    amostra_lgn = rng_lgn.gamma(shape=alpha, scale=1 / beta, size=n_lgn)
    medias_acumuladas = np.cumsum(amostra_lgn) / np.arange(1, n_lgn + 1)

    lgn_df = pd.DataFrame({
        "Número de observações": np.arange(1, n_lgn + 1),
        "Média acumulada": medias_acumuladas,
        "Média teórica": np.repeat(media_teorica, n_lgn)
    })

    st.line_chart(lgn_df, x="Número de observações", y=["Média acumulada", "Média teórica"])
    st.caption("A média acumulada tende a aproximar-se da média teórica à medida que o número de observações aumenta.")

    # Calculamos um histograma em densidade das médias amostrais e comparamos essa aproximação com a curva normal teórica.
    st.header("6. Teorema Limite Central")

    n = st.slider(
        "Escolhe o tamanho de cada amostra (n)",
        min_value=5,
        max_value=200,
        value=30,
        step=5
    )

    N = st.slider(
        "Escolhe o número de amostras (N)",
        min_value=100,
        max_value=3000,
        value=1000,
        step=100
    )

    rng_tlc = np.random.default_rng(seed + 1)

    medias = np.zeros(N)
    
    # em cada iteracao geramos uma amostra e guardamos a sua media
    for i in range(N):
        amostra = rng_tlc.gamma(shape=alpha, scale=1 / beta, size=n)
        medias[i] = np.mean(amostra)
    
    # transformamos as medias num histograma em densidade
    hist_counts, hist_edges = np.histogram(medias, bins=30, density=True)
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    # parametros teoricos da distribuicao das medias amostrais
    mu = alpha / beta
    sigma = np.sqrt(alpha) / beta
    curva_normal = norm.pdf(hist_centers, loc=mu, scale=sigma / np.sqrt(n))

    tlc_df = pd.DataFrame({
        "Médias amostrais": hist_centers,
        "Histograma (densidade)": hist_counts,
        "Curva normal teórica": curva_normal
    })

    st.line_chart(tlc_df, x="Médias amostrais", y=["Histograma (densidade)", "Curva normal teórica"])
    st.caption("Mesmo com distribuição original assimétrica, as médias amostrais tendem para uma forma aproximadamente normal.")

# PARTE II - BASE DE DADOS NHANES

else:
    
    # read_csv carrega o ficheiro para um DataFrame pandas.
    df = pd.read_csv("nhanes_limpo.csv")

    st.header("1. Base de dados")
    st.write("Número de linhas:", df.shape[0]) # shape[0] = numero de linhas
    st.write("Número de colunas:", df.shape[1]) # shape[1] = numero de colunas
    st.write("Variáveis usadas: Gender, Weight, Height, BMI, BPSysAve (Pressão Arterial Sistólica)")

     # O utilizador escolhe qual a variavel quantitativa quer analisar.
    st.header("2. Variável quantitativa")

    variavel = st.selectbox(
        "Escolha uma variável",
        ["Weight", "Height", "BMI", "BPSysAve"]
    )

    st.header("3. Resumo estatístico")
    st.write(df[variavel].describe())  # describe() devolve contagem, media, desvio padrao, minimo, quartis e maximo da variavel escolhida.

    # plotly.express e usado aqui porque permite graficos interativos
    st.header("4. Histograma da variável")
    fig_hist = px.histogram(
        df,
        x=variavel,
        nbins=20,  # nbins define o numero de classes/barras.
        title=f"Histograma de {variavel}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # O boxplot mostra mediana, quartis, dispersao e outliers.
    st.header("5. Boxplot da variável")
    fig_box = px.box(
        df,
        y=variavel,
        title=f"Boxplot de {variavel}"
    )
    st.plotly_chart(fig_box, use_container_width=True)


    # Separamos o BMI (índice de massa corporal) por género para comparar as distribuições.
    st.header("6. Boxplot do BMI por género")
    fig_box_genero = px.box(
        df,
        x="Gender",
        y="BMI",
        color="Gender",
        title="Boxplot do BMI por género",
        color_discrete_map={  # color_discrete_map permite definir manualmente as cores.
            "male": "#4A90E2",
            "female": "#FF69B4",
            "Male": "#4A90E2",
            "Female": "#FF69B4"
        }
    )
    st.plotly_chart(fig_box_genero, use_container_width=True)


    st.header("7. Gráfico de pontos - BMI vs BPSysAve")
    # O scatter plot ajuda a ver se existe associacao entre duas variaveis quantitativas.
    fig_scatter = px.scatter(
        df,
        x="BMI",
        y="BPSysAve",
        opacity=0.6,  # opacity evita que os pontos tapem demasiado uns aos outros.
        title="Dispersão entre BMI e Pressão Arterial Sistólica"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.header("8. Medidas descritivas")

    # Calculamos varias medidas para as quatro variaveis: media, desvio padrao, minimo, maximo e amplitude.
    # amplitude = maximo - minimo
    medidas_df = pd.DataFrame({
        "Média": df[["Weight", "Height", "BMI", "BPSysAve"]].mean(),
        "Desvio padrão": df[["Weight", "Height", "BMI", "BPSysAve"]].std(),
        "Mínimo": df[["Weight", "Height", "BMI", "BPSysAve"]].min(),
        "Máximo": df[["Weight", "Height", "BMI", "BPSysAve"]].max(),
        "Amplitude": df[["Weight", "Height", "BMI", "BPSysAve"]].max() - df[["Weight", "Height", "BMI", "BPSysAve"]].min()
    })

    st.dataframe(medidas_df.round(2))

    st.header("9. Intervalo de confiança para a média do BMI")
    
    # Aqui calculamos um IC de 95% usando a aproximacao normal:  (media +/- 1.96 * erro_padrao)
    # erro_padrao = desvio / raiz(n)

    dados_bmi = df["BMI"].dropna()
    n_ic = len(dados_bmi)
    media_bmi = dados_bmi.mean()
    desvio_bmi = dados_bmi.std()
    margem_erro = 1.96 * (desvio_bmi / np.sqrt(n_ic))

    limite_inferior = media_bmi - margem_erro
    limite_superior = media_bmi + margem_erro

    ic_df = pd.DataFrame({
        "Medida": ["Média amostral do BMI", "Limite inferior", "Limite superior", "Tamanho da amostra"],
        "Valor": [round(media_bmi, 3), round(limite_inferior, 3), round(limite_superior, 3), n_ic]
    })

    st.dataframe(ic_df, hide_index=True)