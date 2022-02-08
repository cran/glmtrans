## ---- echo = FALSE------------------------------------------------------------
library(formatR)

## ---- eval=FALSE--------------------------------------------------------------
#  install.packages("glmtrans", repos = "http://cran.us.r-project.org")

## -----------------------------------------------------------------------------
library(glmtrans)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
set.seed(1, kind = "L'Ecuyer-CMRG")
D.training <- models(family = "binomial", type = "all", cov.type = 2, Ka = 3, K = 5, s = 10, n.target = 100, n.source = rep(100, 5))

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70), message=FALSE---------------
fit.oracle <- glmtrans(target = D.training$target, source = D.training$source, family = "binomial", transfer.source.id = 1:3, cores = 2)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70)------------------------------
fit.detection <- glmtrans(target = D.training$target, source = D.training$source, family = "binomial", transfer.source.id = "auto", cores = 2)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70), message=FALSE---------------
library(glmnet)
fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y, family = "binomial")
fit.pooled <- glmtrans(target = D.training$target, source = D.training$source, family = "binomial", transfer.source.id = "all",  cores = 2)

## ---- tidy=TRUE, tidy.opts=list(width.cutoff=70), message=FALSE---------------
beta <- c(0, rep(0.5, 10), rep(0, 500-10))
er <- numeric(4)
names(er) <- c("Lasso", "Pooled-Trans-GLM", "Trans-GLM", "Oracle-Trans-GLM")
er["Lasso"] <- sqrt(sum((coef(fit.lasso)-beta)^2))
er["Pooled-Trans-GLM"] <- sqrt(sum((fit.pooled$beta-beta)^2))
er["Trans-GLM"] <- sqrt(sum((fit.detection$beta-beta)^2))
er["Oracle-Trans-GLM"] <- sqrt(sum((fit.oracle$beta-beta)^2))
er

## ---- tidy=TRUE---------------------------------------------------------------
plot(fit.detection)

