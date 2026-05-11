# Projeto de Grupo— Laboratorios de Estatistica II

# 1 — Instalar packages 

# install.packages("ggplot2")   # graficos
# install.packages("NHANES")    # base de dados parte II

# Carregar os packages
library(ggplot2)
library(NHANES)

# 2 — PARTE I: Distribuicao Gama~

# Parametros:
# shape/alpha - forma/formato
# rate/beta  - escala

# Formulas:
# Media = alpha / beta
# Variancia = alpha / beta^2
# Desvio Padrão = sqrt(alpha) / beta (raiz quadrada)


# Definir parametros e calcular medidas

alpha <- 2   
beta <- 1  

media <- alpha / beta
variancia <- alpha / beta^2
dp <- sqrt(variancia)

cat("Media:    ", media, "\n")
cat("Variancia: ", variancia, "\n")
cat("Desvio Padrão:       ", dp, "\n")


# Quartis 

q1 <- qgamma(0.25, shape = alpha, rate = beta)
mediana <- qgamma(0.50, shape = alpha, rate = beta)
q3 <- qgamma(0.75, shape = alpha, rate = beta)

cat("\n1o quartil:", round(q1, 3),
    "\nMediana:   ", round(mediana, 3),
    "\n3o quartil:", round(q3, 3))

# Probabilidades
cat("\nProbabilidades:\n")
cat("  P(X <= 2)     :", round(pgamma(2, alpha, beta), 4), "\n")
cat("  P(X >  3)     :", round(1 - pgamma(3, alpha, beta), 4), "\n")
cat("  P(1 < X < 3)  :", round(pgamma(3, alpha, beta) - pgamma(1, alpha, beta), 4), "\n")

# Função de densidade (PDF)

alpha1 <- 1
alpha2 <- 2
alpha3 <- 5
beta <- 1

curve(dgamma(x, shape = alpha1, rate = beta),
      from = 0, to = 10,
      col = "blue", lwd = 2,
      xlab = "x", ylab = "f(x)",
      main = "Função de Densidade da Distribuição Gama")

curve(dgamma(x, shape = alpha2, rate = beta),
      from = 0, to = 10,
      col = "red", lwd = 2, add = TRUE)

curve(dgamma(x, shape = alpha3, rate = beta),
      from = 0, to = 10,
      col = "darkgreen", lwd = 2, add = TRUE)

legend("topright",
       legend = c("alpha = 1", "alpha = 2", "alpha = 5"),
       col = c("blue", "red", "darkgreen"),
       lwd = 2)

# Função acumulada (CDF)

x <- seq(0, 10, length.out = 200)
alphas <- c(1, 2, 5)

# criar data frame
df <- data.frame()

for (a in alphas) {
  temp <- data.frame(
    x = x,
    cdf = pgamma(x, shape = a, rate = beta),
    alpha = factor(a)
  )
  df <- rbind(df, temp)
}

# gráfico CDF
ggplot(df, aes(x = x, y = cdf, color = alpha)) +
  geom_line(size = 1.2) +
  labs(
    title = "Função acumulada (CDF) da Distribuição Gama",
    x = "x",
    y = "F(x) = P(X ≤ x)",
    color = expression(alpha)
  ) +
  theme_minimal()

# Lei dos Grandes Números

set.seed(42)

alpha <- 2
beta <- 1
n_lgn <- 1000

media_teorica <- alpha / beta
amostra_lgn <- rgamma(n_lgn, shape = alpha, rate = beta)
medias_acumuladas <- cumsum(amostra_lgn) / (1:n_lgn)

plot(1:n_lgn, medias_acumuladas, type = "l",
     col = "blue", lwd = 2,
     xlab = "Número de observações",
     ylab = "Média acumulada",
     main = "Lei dos Grandes Números - Gama(2,1)")

abline(h = media_teorica, col = "red", lwd = 2, lty = 2)

legend("topright",
       legend = c("Média acumulada", "Média teórica"),
       col = c("blue", "red"),
       lwd = 2, lty = c(1, 2))

# Teorema Limite Central

set.seed(123) 

# parâmetros 
alpha <- 2
beta <- 1

n <- 30        # tamanho da amostra
N <- 1000      # número de amostras

# vetor para guardar médias
medias <- numeric(N)

# gerar amostras e calcular médias
for (i in 1:N) {
  amostra <- rgamma(n, shape = alpha, rate = beta)
  medias[i] <- mean(amostra)
}

# Gráfico

# histograma das médias
hist(medias, probability = TRUE,
     col = "lightblue",
     main = "Teorema do Limite Central",
     xlab = "Médias amostrais",
     ylab = "Frequência")

# parâmetros teóricos
mu <- alpha / beta
sigma <- sqrt(alpha) / beta

# curva normal teórica
curve(dnorm(x, mean = mu, sd = sigma / sqrt(n)),
      col = "red", lwd = 2, add = TRUE)

# grelha
grid()

# Parte II 
data(NHANES)

# Ver todos dados da bd NHANES
str(NHANES)
summary(NHANES)
names(NHANES)
dim(NHANES)

# Variáveis escolhidas
nhanes_sub <- NHANES[, c("Gender", "Weight", "Height", "BMI", "BPSysAve")]

# Verificar NA
is.na(nhanes_sub)
colSums(is.na(nhanes_sub))

# Remover observações com NA
nhanes_limpo <- na.omit(nhanes_sub)

# Confirmar dados da bd limpa
str(nhanes_limpo)
summary(nhanes_limpo)
dim(nhanes_limpo)

# guardar CSV para o streamlit 
write.csv(nhanes_limpo, "nhanes_limpo.csv", row.names = FALSE)