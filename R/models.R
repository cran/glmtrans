#' Generate data from Gaussian, logistic and Poisson models.
#'
#' Generate data from Gaussian, logistic and Poisson models used in the simulation part of Tian, Y., & Feng, Y. (2023).
#' @export
#' @param family response type. Can be "gaussian", "binomial" or "poisson". Default = "gaussian".
#' \itemize{
#' \item "gaussian": Gaussian distribution.
#' \item "binomial": logistic distribution. When \code{family = "binomial"}, the input response in both \code{target} and \code{source} should be 0/1.
#' \item "poisson": poisson distribution. When \code{family = "poisson"}, the input response in both \code{target} and \code{source} should be non-negative.
#' }
#' @param type the type of generated data. Can be "all", "source" or "target".
#' \itemize{
#' \item "all": generate a list with a target data set of size \code{n.target} and K source data set of size \code{n.source}.
#' \item "source": generate a list with K source data set of size \code{n.source}.
#' \item "target": generate a list with a target data set of size \code{n.target}.
#' }
#' @param cov.type the type of covariates. Can be 1 or 2 (numerical). If it equals to 1, the predictors will be generated from the distribution used in Section 4.1.1 (Ah-Trans-GLM) in the latest version of Tian, Y., & Feng, Y. (2023). If it equals to 2, the predictors will be generated from the distribution used in Section 4.1.2 (When transferable sources are unknown).
#' @param h measures the deviation (\eqn{l_1}-norm) of transferable source coefficient from the target coefficient. Default = 5.
#' @param K the number of source data sets. Default = 5.
#' @param n.target the sample size of target data. Should be a positive integer. Default = 100.
#' @param n.source the sample size of each source data. Should be a vector of length \code{K}. Default is a \code{K}-vector with all elements 150.
#' @param s how many components in the target coefficient are non-zero, which controls the sparsity of target problem. Default = 15.
#' @param p the dimension of data. Default = 1000.
#' @param Ka the number of transferable sources. Should be an integer between 0 and \code{K}. Default = K.
#' @return a list of data sets which depend on the value of \code{type}.
#' \itemize{
#' \item \code{type} = "all": a list of two components named "target" and "source" storing the target and source data, respectively. Component source is a list containing \code{K} components with the first \code{Ka} ones \code{h}-transferable and the remaining ones \code{h}-nontransferable. The target data set and each source data set have components "x" and "y", as the predictors and responses, respectively.
#' \item \code{type} = "source": a list with a signle component "source". This component contains a list of \code{K} components with the first \code{Ka} ones \code{h}-transferable and the remaining ones \code{h}-nontransferable. Each source data set has components "x" and "y", as the predictors and responses, respectively.
#' \item \code{type} = "target": a list with a signle component "target". This component contains another list with components "x" and "y", as the predictors and responses of target data, respectively.
#' }
#' @seealso \code{\link{glmtrans}}.
#' @references
#' Tian, Y., & Feng, Y. (2023). \emph{Transfer learning under high-dimensional generalized linear models. Journal of the American Statistical Association, 118(544), 2684-2697.}
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#'
#' D.all <- models("binomial", type = "all")
#' D.target <- models("binomial", type = "target")
#' D.source <- models("binomial", type = "source")
#'

models <- function(family = c("gaussian", "binomial", "poisson"), type = c("all", "source", "target"), cov.type = 1, h = 5, K = 5, n.target = 200, n.source = rep(100, K), s = 5, p = 500, Ka = K) {
  family <- match.arg(family)
  target <- NULL
  source <- NULL

  type <- match.arg(type)
  sig.strength <- 0.5

  if (family == "gaussian" || family == "binomial") {
    if(type == "all" || type == "target") {
      wk <- c(rep(sig.strength, s), rep(0, p-s))
      if (cov.type == 1) {
        Sigma <- outer(1:p, 1:p, function(x,y){
          0.5^(abs(x-y))
        })
        R <- chol(Sigma)
        target <- list(x = NULL, y = NULL)

        target$x <- matrix(rnorm(n.target*p), nrow = n.target) %*% R

      } else if (cov.type == 2) {
        Sigma <- outer(1:p, 1:p, function(x,y){
          0.9^(abs(x-y))
        })
        R <- chol(Sigma)
        target$x <- matrix(rnorm(n.target*p), nrow = n.target) %*% R
      }

      if (family == "gaussian") {
        target$y <- as.numeric(target$x %*% wk + rnorm(n.target))
      } else if (family == "binomial") {
        pr <- 1/(1+exp(-target$x %*% wk))
        target$y <- sapply(1:n.target, function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
      }
    }

    if(type == "all" || type == "source") {
     if (cov.type == 1) {
        Sigma <- outer(1:p, 1:p, function(x,y){
          0.5^(abs(x-y))
        })
        eps <- rnorm(p, sd = 0.3)
        Sigma <- Sigma + eps %*% t(eps)

        R <- chol(Sigma)
      }

      source <- sapply(1:K, function(k){
        if (k <= Ka){
          wk <- c(rep(sig.strength, s), rep(0, p-s)) + h/p*sample(c(-1,1), size = p, replace = TRUE)
        } else {
          sig.index <- c(s+1:s, sample((2*s+1):p, s))
          wk <- rep(0, p)
          wk[sig.index] <- sig.strength
          wk <- wk + 2*h/p*sample(c(-1,1), size = p, replace = TRUE)
        }
        if (cov.type == 1) {
          x <- matrix(rnorm(n.source[k]*p), nrow = n.source[k]) %*% R
        } else if (cov.type == 2) {
          x <- matrix(rt(n.source[k]*p, df = 4), nrow = n.source[k])
        }

        if (family == "gaussian") {
          y <- as.numeric(0.5*I(k > Ka) + x %*% wk + rnorm(n.source[k]))
        } else if (family == "binomial") {
          pr <- 1/(1+exp(-0.5*I(k > Ka) -x %*% wk))
          y <- sapply(1:n.source[k], function(i){
            sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))
          })
        }
        list(x = x, y = y)
      }, simplify = FALSE)
    }


  } else { # model == "poisson
    if(type == "all" || type == "target") {
      if (cov.type == 1) {
        Sigma <- outer(1:p, 1:p, function(x,y){
          0.5^(abs(x-y))
        })
        R <- chol(Sigma)
      } else if (cov.type == 2) {
        Sigma <- outer(1:p, 1:p, function(x,y){
          0.9^(abs(x-y))
        })
        R <- chol(Sigma)
      }
      wk <- c(rep(sig.strength, s), rep(0, p-s))

      target$x <- matrix(rnorm(n.target*p), nrow = n.target) %*% R

      target$x[target$x > 0.5] <- 0.5
      target$x[target$x < -0.5] <- -0.5
      lambda <- as.numeric(exp(target$x %*% wk))
      target$y <- rpois(n.target, lambda)
    }

    if(type == "all" || type == "source") {
      if (cov.type == 1) {
        Sigma <- outer(1:p, 1:p, function(x,y){
          0.5^(abs(x-y))
        })
        eps <- rnorm(p, sd = 0.3)
        Sigma <- Sigma + eps %*% t(eps)

        R <- chol(Sigma)
      }

      source <- sapply(1:K, function(k){
        if (k <= Ka){
          wk <- c(rep(sig.strength, s), rep(0, p-s)) + h/p*sample(c(-1,1), size = p, replace = TRUE)
        } else {
          sig.index <- c(s+1:s, sample((2*s+1):p, s))
          wk <- rep(0, p)
          wk[sig.index] <- sig.strength
          wk <- wk + 2*h/p*sample(c(-1,1), size = p, replace = TRUE)
        }
        if (cov.type == 1) {
          x <- matrix(rnorm(n.source[k]*p), nrow = n.source[k]) %*% R
        } else if (cov.type == 2) {
          x <- matrix(rt(n.source[k]*p, df = 4), nrow = n.source[k])
        }
        x[x > 0.5] <- 0.5
        x[x < -0.5] <- -0.5
        lambda <- as.numeric(exp(0.5*I(k > Ka) + x %*% wk))
        y <- rpois(n.source[k], lambda)

        list(x = x, y = y)
      }, simplify = FALSE)
    }

  }


  if (type == "all") {
    return(list(target = target, source = source))
  } else if (type == "target") {
    return(list(target = target))
  } else {
    return(list(source = source))
  }

}
