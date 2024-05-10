load("trabalhosalarios.RData")
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(glmnet)

# Padronizar os resultados
set.seed(75)

# Passando o dataset em uma variavel para ficar mais facil a chamada
dat <- trabalhosalarios

#Tirando a coluna earns pois não será utilizada
dat$earns <- NULL 

# Criar um indice e particionar o dataset em 80:20
index = sample(1:nrow(dat),0.8*nrow(dat))
train = dat[index,]  
test = dat[-index,] 

# Dimensão das bases, ambas com 17 colunas
dim(train) # 2059 amostras
dim(test) # 515 amostras

# Colunas que necessitam de padronização e não são binárias exceto lwage
cols = c('husage', 'husearns', 'huseduc', 'hushrs',
         'age', 'educ', 'exper', 'lwage')

# Padronizando a base de treinamento e teste
pre_proc_val <- preProcess(train[,cols], 
                           method = c("center", "scale"))
train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

#############################################################
#                        Genéricos                          #
#############################################################

# Construindo uma matriz de dados para a predicao

# Criando o dataframe para predição no futuro
husage_norm <- (40-pre_proc_val[["mean"]][["husage"]])/
  pre_proc_val[["std"]][["husage"]]
husearns_norm <- (600-pre_proc_val[["mean"]][["husearns"]])/
  pre_proc_val[["std"]][["husearns"]]
huseduc_norm <- (13-pre_proc_val[["mean"]][["huseduc"]])/
  pre_proc_val[["std"]][["huseduc"]]
hushrs_norm <- (40-pre_proc_val[["mean"]][["hushrs"]])/
  pre_proc_val[["std"]][["hushrs"]]
age_norm <- (38-pre_proc_val[["mean"]][["age"]])/
  pre_proc_val[["std"]][["age"]]
educ_norm <- (13-pre_proc_val[["mean"]][["educ"]])/
  pre_proc_val[["std"]][["educ"]]
exper_norm <- (18-pre_proc_val[["mean"]][["exper"]])/
  pre_proc_val[["std"]][["exper"]]

our_pred <- as.matrix(data.frame(husage=husage_norm, 
                                 husunion=0,
                                 husearns=husearns_norm,
                                 huseduc=huseduc_norm,
                                 husblck=1,
                                 hushisp=0,
                                 hushrs=hushrs_norm,
                                 kidge6=1,
                                 age=age_norm,
                                 black=0,
                                 educ=educ_norm,
                                 hispanic=1,
                                 union=0,
                                 exper=exper_norm,
                                 kidlt6=1))


# Variaveis dummies para organizar os datasets em objetos tipo matriz

# Objeto com os valores do modelo
cols_reg = c('husage', 'husunion', 'husearns', 'huseduc', 'husblck', 
             'hushisp', 'hushrs', 'kidge6', 'age', 'black', 'educ',
             'hispanic', 'union', 'exper', 'kidlt6', 'lwage')

# Estamos interessado em estimar o 
# salário-hora da esposa em logaritmo neperiano (lwage)
dummies <- dummyVars(lwage~husage+husunion+husearns+huseduc+
                       husblck+hushisp+hushrs+kidge6+age+
                       black+educ+hispanic+union+exper+kidlt6, 
                     data = dat[,cols_reg])
train_dummies = predict(dummies, newdata = train[,cols_reg])
test_dummies = predict(dummies, newdata = test[,cols_reg])
print(dim(train_dummies)); print(dim(test_dummies))

# Vamos guardar a matriz de dados de treinamento das 
# variaveis explicativas para o modelo em um objeto 
# chamado "x"
x = as.matrix(train_dummies)

# Vamos guardar o vetor de dados de treinamento da 
# variavel dependente para o modelo em um objeto
# chamado "y_train"
y_train = train$lwage

# Vamos guardar a matriz de dados de teste das variaveis
# explicativas para o modelo em um objeto chamado
# "x_test"
x_test = as.matrix(test_dummies)

# Vamos guardar o vetor de dados de teste da variavel
# dependente para o modelo em um objeto chamado "y_test"
y_test = test$lwage

# Lambda padrao para Ridge e Lasso
lambdas <- 10^seq(2, -3, by = -.1)

#############################################################
#                     REGRESSAO RIDGE                       #
#############################################################

# Vamos calcular o valor otimo de lambda; 
# alpha = "0", eh para regressao Ridge
# Vamos testar os lambdas de 10^-3 ate 10^2, a cada 0.1

# Calculando o lambda:
ridge_lamb <- cv.glmnet(x, y_train, alpha = 0, 
                        lambda = lambdas)
# Vamos ver qual o lambda otimo 
best_lambda_ridge <- ridge_lamb$lambda.min
best_lambda_ridge

# Estimando o modelo Ridge
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, 
                   family = 'gaussian', 
                   lambda = best_lambda_ridge)

# Vamos ver o resultado (valores) da estimativa 
# (coeficientes)
ridge_reg[["beta"]]

# Vamos calcular o R^2 dos valores verdadeiros e 
# preditos conforme a seguinte funcao:
ridge_eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # As metricas de performace do modelo:
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# Predicao e avaliacao nos dados de treinamento:
ridge_predictions_train <- predict(ridge_reg, 
                                   s = best_lambda_ridge,
                                   newx = x)

# As metricas da base de treinamento sao:
ridge_eval_results(y_train, ridge_predictions_train, train)

# Predicao e avaliacao nos dados de teste:
ridge_predictions_test <- predict(ridge_reg, 
                                  s = best_lambda_ridge, 
                                  newx = x_test)

# As metricas da base de teste sao:
ridge_eval_results(y_test, ridge_predictions_test, test)

# Fazendo a predicao:
predict_our_ridge <- predict(ridge_reg, 
                             s = best_lambda_ridge, 
                             newx = our_pred)
# Normalizando o resultado:
lwage_pred_ridge=(predict_our_ridge*
                    pre_proc_val[["std"]][["lwage"]])+
  pre_proc_val[["mean"]][["lwage"]]

# Antilog no resultado:
ridge_lwage <- exp(lwage_pred_ridge)

# O resultado da predicao é:
predict_our_ridge

# O resultado normalizado é:
lwage_pred_ridge

# O salário hora da esposa é (antilog logaritmo neperiano)
ridge_lwage

# Este eh o valor predito do salario por hora (US$), 
# segundo as caracteristicas que atribuimos

# O intervalo de confianca para o nosso exemplo eh:
n_ridge <- nrow(train) # tamanho da amostra
m_ridge <- ridge_lwage # valor medio predito
s_ridge <- pre_proc_val[["std"]][["lwage"]] # desvio padrao
dam_ridge <- s_ridge/sqrt(n_ridge) # distribuicao da amostragem da media
CIlwr_ridge <- m_ridge + (qnorm(0.025))*dam_ridge # intervalo inferior
CIupr_ridge <- m_ridge - (qnorm(0.025))*dam_ridge # intervalo superior

# Os valores sao:
CIlwr_ridge
CIupr_ridge

#############################################################
#                     REGRESSAO LASSO                       #
#############################################################

# NOTA: Não é necessário gerar novamente as variáveis para dataset pois
# iremos reaproveitar da regressão Ridge

# Vamos atribuir alpha = 1 para implementar a regressao
# lasso
lasso_lamb <- cv.glmnet(x, y_train, alpha = 1, 
                        lambda = lambdas, 
                        standardize = TRUE, nfolds = 5)

# Vamos guardar o lambda "otimo" em um objeto chamado
# best_lambda_lasso
best_lambda_lasso <- lasso_lamb$lambda.min 
best_lambda_lasso

# Vamos estimar o modelo Lasso 
lasso_model <- glmnet(x, y_train, alpha = 1, 
                      lambda = best_lambda_lasso, 
                      standardize = TRUE)

# Vamos visualizar os coeficientes estimados
lasso_model[["beta"]]
# Perceba que alguns coeficientes estao zerados pois 
# nao sao significativos

# Vamos fazer as predicoes na base de treinamento e
# avaliar a regressao Lasso 
lasso_predictions_train <- predict(lasso_model, 
                             s = best_lambda_lasso,
                             newx = x)

# Vamos calcular o R^2 dos valores verdadeiros e 
# preditos conforme a seguinte funcao:
lasso_eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # As metricas de performace do modelo:
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# As metricas da base de treinamento sao:
lasso_eval_results(y_train, lasso_predictions_train, train)

# Vamos fazer as predicoes na base de teste
lasso_predictions_test <- predict(lasso_model, 
                            s = best_lambda_lasso, 
                            newx = x_test)

# As metricas da base de teste sao:
lasso_eval_results(y_test, lasso_predictions_test, test)

# Vamos para a predicao
predict_our_lasso <- predict(lasso_model, 
                             s = best_lambda_lasso, 
                             newx = our_pred)

# Novamente, o resultado esta padronizado, nos temos de
# converte-lo para valor compativel com o dataset original
lwage_pred_lasso=(predict_our_lasso*
                   pre_proc_val[["std"]][["lwage"]])+
  pre_proc_val[["mean"]][["lwage"]]

# Antilog no resultado:
lasso_lwage <- exp(lwage_pred_lasso)

# O resultado da predicao é:
predict_our_lasso

# O resultado normalizado é:
lwage_pred_lasso

# O salário hora da esposa é (antilog logaritmo neperiano)
lasso_lwage

# Vamos criar o intervalo de confianca para o nosso
# exemplo
n_lasso <- nrow(train)
m_lasso <- lasso_lwage
s_lasso <- pre_proc_val[["std"]][["lwage"]]
dam_lasso <- s_lasso/sqrt(n_lasso)
CIlwr_lasso <- m_lasso + (qnorm(0.025))*dam_lasso
CIupr_lasso <- m_lasso - (qnorm(0.025))*dam_lasso

# O intervalo de confianca eh:
CIlwr_lasso
CIupr_lasso

#############################################################
#                     REGRESSAO ELASTICNET                  #
#############################################################

train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)

# Vamos treinar o modelo
elastic_reg <- train(lwage~husage+husunion+husearns+huseduc+
                       husblck+hushisp+hushrs+kidge6+age+
                       black+educ+hispanic+union+exper+kidlt6,
                     data = train,
                     method = "glmnet",
                     tuneLength = 10,
                     trControl = train_cont)

# O melhor parametro alpha escolhido eh:
elastic_reg$bestTune

# E os parametros sao:
elastic_reg[["finalModel"]][["beta"]]

# Vamos fazer as predicoes e avaliar a performance do
# modelo

# Vamos fazer as predicoes no modelo de treinamento:
en_predictions_train <- predict(elastic_reg, x)

# Vamos calcular o R^2 dos valores verdadeiros e 
# preditos conforme a seguinte funcao:
en_eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # As metricas de performace do modelo:
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# As metricas de performance na base de treinamento
# sao:
en_eval_results(exp(y_train), en_predictions_train, exp(train)) 

# Vamos fazer as predicoes na base de teste
en_predictions_test <- predict(elastic_reg, exp(x_test))

# As metricas de performance na base de teste sao:
en_eval_results(y_test, en_predictions_test, test)

# Vamos fazer a predicao com base nos parametros que
# selecionamos
predict_our_elastic <- predict(elastic_reg,our_pred)

# Novamente, o resultado eh padronizado, nos temos que
# reverte-lo para o nivel dos valores originais do
# dataset, vamos fazer isso:
lwage_pred_elastic=(predict_our_elastic*
                     pre_proc_val[["std"]][["lwage"]])+
  pre_proc_val[["mean"]][["lwage"]]

elastic_lwage <- exp(lwage_pred_elastic)

# O resultado da predicao é:
predict_our_elastic

# O resultado normalizado é:
lwage_pred_elastic

# O salário hora da esposa é (antilog logaritmo neperiano)
elastic_lwage

# Vamos criar o intervalo de confianca para o nosso
# exemplo
n_elastic <- nrow(train)
m_elastic <- elastic_lwage
s_elastic <- pre_proc_val[["std"]][["lwage"]]
dam_elastic <- s_elastic/sqrt(n_elastic)
CIlwr_elastic <- m_elastic + (qnorm(0.025))*dam_elastic
CIupr_elastic <- m_elastic - (qnorm(0.025))*dam_elastic

# Os valores minimo e maximo sao:
CIlwr_elastic
CIupr_elastic
