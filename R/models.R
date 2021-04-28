#' Generate data from Gaussian, logistic and Poisson models.
#'
#' Generate data from Gaussian, logistic and Poisson models used in the simulation part of Tian, Y. and Feng, Y., 2021.
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
#' @param h measures the deviation (\eqn{l_1}-norm) of transferable source coefficient from the target coefficient.
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
#' Tian, Y. and Feng, Y., 2021. \emph{Transfer learning with high-dimensional generalized linear models. Submitted.}
#' @examples
#' set.seed(1, kind = "L'Ecuyer-CMRG")
#'
#' D.all <- models("binomial", type = "all")
#' D.target <- models("binomial", type = "target")
#' D.source <- models("binomial", type = "source")
#'
models <- function(family = c("gaussian", "binomial", "poisson"), type = c("all", "source", "target"), h = 5, K = 5, n.target = 100, n.source = rep(150, K), s = 15, p = 1000, Ka = K) {
  family <- match.arg(family)
  sign <- "flip"
  target <- NULL
  source <- NULL

  type <- match.arg(type)

  if (family == "gaussian" || family == "binomial") {
    Sigma <- outer(1:p, 1:p, function(x,y){
      0.5^(abs(x-y))
    })
    R <- chol(Sigma)


    if(type == "all" || type == "target") {
      target <- list(x = NULL, y = NULL)
      wk <- c(rep(0.5, s), rep(0, p-s))
      target$x <- x <- matrix(rnorm(n.target*p), nrow = n.target) %*% R
      if (family == "gaussian") {
        target$y <- as.numeric(x %*% wk + rnorm(n.target))
      } else {
        pr <- 1/(1+exp(-x %*% wk))
        target$y <- sapply(1:n.target, function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
      }
    }

    if(type == "all" || type == "source") {
      source <- sapply(1:K, function(k){
        if (k <= Ka){
          if (sign == "flip") {
            wk <- c(rep(0.5, s), rep(0, p-s)) + h/p*sample(c(-1,1), size = p, replace = TRUE)
          } else if (sign == "neg") {
            wk <- c(rep(0.5, s), rep(0, p-s)) - h/p
          }

        } else {
          sig.index <- c(s+1:s, sample((2*s+1):p, s))
          wk <- rep(0, p)
          wk[sig.index] <- 0.5
          wk <- wk + 2*h/p*sample(c(-1,1), size = p, replace = TRUE)
        }

        x <- matrix(rnorm(n.source[k]*p), nrow = n.source[k]) %*% R
        if (family == "gaussian") {
          y <- as.numeric(0.5 + x %*% wk + rnorm(n.source[k]))
        } else {
          pr <- 1/(1+exp(-0.5 -x %*% wk))
          y <- sapply(1:n.source[k], function(i){
            sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))
          })
        }
        list(x = x, y = y)
      }, simplify = FALSE)
    }


  } else { # model == "poisson
    Sigma <- outer(1:p, 1:p, function(x,y){
      0.5^(abs(x-y))
    })
    R <- chol(Sigma)


    if(type == "all" || type == "target") {
      wk <- c(rep(0.5, s), rep(0, p-s))
      target$x <- matrix(rnorm(n.target*p), nrow = n.target) %*% R
      target$x[target$x > 1] <- 1
      target$x[target$x < -1] <- -1
      lambda <- as.numeric(exp(target$x %*% wk))
      target$y <- rpois(n.target, lambda)
    }

    if(type == "all" || type == "source") {
      source <- sapply(1:K, function(k){
        if (k <= Ka){
          if (sign == "flip") {
            wk <- c(rep(0.5, s), rep(0, p-s)) + h/p*sample(c(-1,1), size = p, replace = TRUE)
          } else if (sign == "neg") {
            wk <- c(rep(0.5, s), rep(0, p-s)) - h/p
          }
        } else {
          sig.index <- c(s+1:s, sample((2*s+1):p, s))
          wk <- rep(0, p)
          wk[sig.index] <- 0.5
          wk <- wk + 2*h/p*sample(c(-1,1), size = p, replace = TRUE)
        }

        x <- matrix(rnorm(n.source[k]*p), nrow = n.source[k]) %*% R
        x[x > 1] <- 1
        x[x < -1] <- -1
        lambda <- as.numeric(exp(0.5 + x %*% wk))
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
