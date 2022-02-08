#' Calculate asymptotic confidence intervals based on desparsified Lasso and two-step transfer learning method.
#'
#' Given the point esimate of the coefficient vector from \code{glmtrans}, calculate the asymptotic confidence interval of each component. The detailed inference algorithm can be found as Algorithm 3 in the latest version of Tian, Y. and Feng, Y., 2021. The algorithm is consructed based on a modified version of desparsified Lasso (Van de Geer, S. et al, 2014; Dezeure, R. et al, 2015).
#' @export
#' @param target target data. Should be a list with elements x and y, where x indicates a predictor matrix with each row/column as a(n) observation/variable, and y indicates the response vector.
#' @param source source data. Should be a list with some sublists, where each of the sublist is a source data set, having elements x and y with the same meaning as in target data.
#' @param family response type. Can be "gaussian", "binomial" or "poisson". Default = "gaussian".
#' \itemize{
#' \item "gaussian": Gaussian distribution.
#' \item "binomial": logistic distribution. When \code{family = "binomial"}, the input response in both \code{target} and \code{source} should be 0/1.
#' \item "poisson": poisson distribution. When \code{family = "poisson"}, the input response in both \code{target} and \code{source} should be non-negative.
#' }
#' @param beta.hat initial estimate of the coefficient vector (the intercept should be the first component). Can be from the output of function \code{glmtrans}.
#' @param nodewise.transfer.source.id transferable source indices in the infernce (the set A in Algorithm 3 of Tian, Y. and Feng, Y., 2021). Can be either a subset of \code{{1, ..., length(source)}}, "all" or \code{NULL}. Default = \code{"all"}.
#' \itemize{
#' \item a subset of \code{{1, ..., length(source)}}: only transfer sources with the specific indices.
#' \item "all": transfer all sources.
#' \item NULL: don't transfer any sources and only use target data.
#' }
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param level the level of confidence interval. Default = 0.95. Note that the level here refers to the asymptotic level of confidence interval of a single component rather than the multiple intervals.
#' @param intercept whether the model includes the intercept or not. Default = TRUE. Should be set as TRUE if the intercept of \code{beta.hat} is not zero.
#' @param ... additional arguments.
#' @return a list of output. b.hat = b.hat, beta.hat = beta.hat, CI = CI, var.est = var.est
#' \item{b.hat}{the center of confidence intervals. A \code{p}-dimensional vector, where \code{p} is the number of predictors.}
#' \item{beta.hat}{the initial estimate of the coefficient vector (the same as input).}
#' \item{CI}{confidence intervals (CIs) with the specific level. A \code{p} by 3 matrix, where three columns indicate the center, lower limit and upper limit of CIs, respectively. Each row represents a coefficient component.}
#' \item{var.est}{the estimate of variances in the CLT (Theta transpose times Sigma times Theta, in section 2.5 of Tian, Y. and Feng, Y., 2021). A \code{p}-dimensional vector, where \code{p} is the number of predictors.}
#' @seealso \code{\link{glmtrans}}.
#' @references
#' Tian, Y. and Feng, Y., 2021. \emph{Transfer Learning under High-dimensional Generalized Linear Models. arXiv preprint arXiv:2105.14328.}
#'
#' Van de Geer, S., Bühlmann, P., Ritov, Y.A. and Dezeure, R., 2014. \emph{On asymptotically optimal confidence regions and tests for high-dimensional models. The Annals of Statistics, 42(3), pp.1166-1202.}
#'
#' Dezeure, R., Bühlmann, P., Meier, L. and Meinshausen, N., 2015. \emph{High-dimensional inference: confidence intervals, p-values and R-software hdi. Statistical science, pp.533-558.}
#' @examples
#' \dontrun{
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#'
#' # generate binomial data
#' D.training <- models("binomial", type = "all", K = 2, p = 200)
#'
#' # fit a logistic regression model via two-step transfer learning method
#' fit.binomial <- glmtrans(D.training$target, D.training$source, family = "binomial")
#'
#' # calculate the CI based on the point estimate from two-step transfer learning method
#' fit.inf <- glmtrans_inf(target = D.training$target, source = D.training$source,
#' family = "binomial", beta.hat = fit.binomial$beta, cores = 2)
#' }

glmtrans_inf <- function(target, source = NULL, family = c("gaussian", "binomial", "poisson"), beta.hat = NULL, nodewise.transfer.source.id = "all", cores = 1, level = 0.95, intercept = TRUE, ...) {
  family <- match.arg(family)
  options(warn=1)
  if (cores <= 1) {
    warning("Only a single core is used. The calculation can be slow, especially when the dimension is large. Multi-cores are suggested.")
  }
  options(warn=0)
  registerDoParallel(cores)
  # cl <- makeCluster(cores, outfile="")
  # registerDoParallel(cl)
  r.level <- level + (1-level)/2
  j <- 0

  if (!is.null(colnames(beta.hat))) {
    beta.hat.names <- colnames(beta.hat)
  } else if (!is.null(names(beta.hat))) {
    beta.hat.names <- names(beta.hat)
  } else {
    beta.hat.names <- NULL
  }

  beta.hat <- as.vector(beta.hat)

  if (!is.null(source) && (is.string(nodewise.transfer.source.id) && nodewise.transfer.source.id == "all")) { # transfer all source data
    nodewise.transfer.source.id <- 1:length(source)
  } else if (is.null(source) || is.null(nodewise.transfer.source.id)) { # don't transfer any source
    nodewise.transfer.source.id <- NULL
  }

  D <- list(target = target, source = source)
  for (k in 1:(length(nodewise.transfer.source.id)+1)) {
    if (k == 1) {
      D$target$x <- as.matrix(D$target$x)
    } else {
      D$source[[nodewise.transfer.source.id[k-1]]]$x <- as.matrix(D$source[[nodewise.transfer.source.id[k-1]]]$x)
    }
  }

  if (family == "gaussian") {
    D.centeralized <- D

    if (intercept) {
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          D.centeralized$source[[nodewise.transfer.source.id[k-1]]]$x <- cbind(1, D.centeralized$source[[nodewise.transfer.source.id[k-1]]]$x)
        } else {
          D.centeralized$target$x <- cbind(1, D.centeralized$target$x)
        }
      }
    }

    p <- ncol(D.centeralized$target$x)

    X.comb <- foreach(k = unique(c(0, nodewise.transfer.source.id)), .combine = "rbind") %do% {
      if (k > 0) {
        D.centeralized$source[[k]]$x
      } else {
        D.centeralized$target$x
      }
    }

    Sigma.hat <- t(X.comb) %*% X.comb/nrow(X.comb)


    L <- foreach(j = 1:p, .combine = "rbind") %dopar% {
      D1 <- D.centeralized
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          D1$source[[nodewise.transfer.source.id[k-1]]]$y <- D1$source[[nodewise.transfer.source.id[k-1]]]$x[, j]
          D1$source[[nodewise.transfer.source.id[k-1]]]$x <- D1$source[[nodewise.transfer.source.id[k-1]]]$x[, -j]
        } else {
          D1$target$y <- D1$target$x[, j]
          D1$target$x <- D1$target$x[, -j]
        }
      }
      node.lasso <- glmtrans(target = D1$target, source = D1$source, family = "gaussian", alg = "ori", transfer.source.id = nodewise.transfer.source.id, intercept = FALSE, detection.info = FALSE, ...)
      gamma <- node.lasso$beta[-1]
      tau2 <- Sigma.hat[j, j] - Sigma.hat[j, -j, drop = F] %*% gamma
      theta <- rep(1, p)
      theta[-j] <- -gamma
      c(theta, tau2)
    }
    Theta.hat <- solve(diag(L[,p+1])) %*% L[1:p, 1:p]
    Z <- D$target$y - D$target$x %*% beta.hat[-1] - beta.hat[1]

    if (intercept) {
      b.hat <- as.matrix(beta.hat) + Theta.hat %*% t(cbind(1, D$target$x)) %*% Z/length(D$target$y)
    } else {
      b.hat <- beta.hat[-1] + Theta.hat %*% t(D$target$x) %*% Z/length(D$target$y)
    }

    var.est <- diag(Theta.hat %*% Sigma.hat %*% t(Theta.hat))
    CI <- data.frame(b.hat = b.hat, lb = b.hat - qnorm(r.level)*sqrt(var.est/length(D$target$y)), ub = b.hat + qnorm(r.level)*sqrt(var.est/length(D$target$y)))

  } else if (family == "binomial") {
    Dw <- D

    if (intercept) {
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          uk <- Dw$source[[nodewise.transfer.source.id[k-1]]]$x %*% beta.hat[-1] + beta.hat[1]
          wk <- as.vector(sqrt(exp(-uk)/((1+exp(-uk))^2)))
          Dw$source[[nodewise.transfer.source.id[k-1]]]$x <- cbind(wk, diag(wk, nrow = length(wk)) %*% Dw$source[[nodewise.transfer.source.id[k-1]]]$x)
        } else {
          uk <- Dw$target$x %*% beta.hat[-1] + beta.hat[1]
          wk <- as.vector(sqrt(exp(-uk)/((1+exp(-uk))^2)))
          Dw$target$x <- cbind(wk, diag(wk, nrow = length(wk)) %*% Dw$target$x)
        }
      }
    } else {
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          uk <- Dw$source[[nodewise.transfer.source.id[k-1]]]$x %*% beta.hat[-1]
          wk <- as.vector(sqrt(exp(-uk)/((1+exp(-uk))^2)))
          Dw$source[[nodewise.transfer.source.id[k-1]]]$x <- diag(wk) %*% Dw$source[[nodewise.transfer.source.id[k-1]]]$x
        } else {
          uk <- Dw$target$x %*% beta.hat[-1]
          wk <- as.vector(sqrt(exp(-uk)/((1+exp(-uk))^2)))
          Dw$target$x <- diag(wk) %*% Dw$target$x
        }
      }
    }

    Xw <- foreach(k = unique(c(0, nodewise.transfer.source.id)), .combine = "rbind") %do% {
      if (k > 0) {
        Dw$source[[k]]$x
      } else {
        Dw$target$x
      }
    }

    p <- ncol(Dw$target$x)
    Sigma.hat <- (t(Xw) %*% Xw)/nrow(Xw)

    L <- foreach(j = 1:p, .combine = "rbind") %dopar% {
      # if (j %% 100 == 1) {
      #   print(j)
      # }
      D1 <- Dw
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          D1$source[[nodewise.transfer.source.id[k-1]]]$y <- D1$source[[nodewise.transfer.source.id[k-1]]]$x[, j]
          D1$source[[nodewise.transfer.source.id[k-1]]]$x <- D1$source[[nodewise.transfer.source.id[k-1]]]$x[, -j]
        } else {
          D1$target$y <- D1$target$x[, j]
          if (all(abs(D1$target$y) <= 1e-20)) {
            D1$target$y <- rep(1e-20, length(D1$target$y))
          }
          D1$target$x <- D1$target$x[, -j]
        }
      }

      node.lasso <- try(glmtrans(target = D1$target, source = D1$source, family = "gaussian", alg = "ori", transfer.source.id = nodewise.transfer.source.id, intercept = FALSE, ...))
      if (class(node.lasso) == "try-error") {
        stop(paste("errors happened in feature", j))
      }
      gamma <- node.lasso$beta[-1]
      tau2 <- Sigma.hat[j, j] - Sigma.hat[j, -j, drop = F] %*% gamma
      theta <- rep(1, p)
      theta[-j] <- -gamma
      c(theta, tau2)
    }

    tau2 <- L[,p+1]
    tau2.inv <- 1/tau2
    tau2.inv[abs(tau2) <= 1e-20] <- 0
    Theta.hat <- diag(tau2.inv) %*% L[1:p, 1:p]
    # Theta.hat <- solve(diag(L[,p+1])) %*% L[1:p, 1:p]
    u.target <- D$target$x %*% beta.hat[-1] + beta.hat[1]
    Z <- D$target$y - 1/(1+exp(-u.target))

    if (intercept) {
      b.hat <- as.matrix(beta.hat) + Theta.hat %*% t(cbind(1, D$target$x)) %*% Z/length(D$target$y)
    } else {
      b.hat <- beta.hat[-1] + Theta.hat %*% t(D$target$x) %*% Z/length(D$target$y)
    }


    var.est <- diag(Theta.hat %*% Sigma.hat %*% t(Theta.hat))
    CI <- data.frame(b.hat = b.hat, lb = b.hat - qnorm(r.level)*sqrt(var.est/length(D$target$y)), ub =  b.hat + qnorm(r.level)*sqrt(var.est/length(D$target$y)))
  } else if (family == "poisson") {
    Dw <- D

    if (intercept) {
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          uk <- Dw$source[[nodewise.transfer.source.id[k-1]]]$x %*% beta.hat[-1] + beta.hat[1]
          wk <- as.vector(exp(uk/2))
          Dw$source[[nodewise.transfer.source.id[k-1]]]$x <- cbind(wk, diag(wk) %*% Dw$source[[nodewise.transfer.source.id[k-1]]]$x)
        } else {
          uk <- Dw$target$x %*% beta.hat[-1] + beta.hat[1]
          wk <- as.vector(exp(uk/2))
          Dw$target$x <- cbind(wk, diag(wk) %*% Dw$target$x)
        }
      }
    }

    Xw <- foreach(k = unique(c(0, nodewise.transfer.source.id)), .combine = "rbind") %do% {
      if (k > 0) {
        Dw$source[[k]]$x
      } else {
        Dw$target$x
      }
    }

    p <- ncol(Dw$target$x)
    Sigma.hat <- (t(Xw) %*% Xw)/nrow(Xw)

    L <- foreach(j = 1:p, .combine = "rbind") %dopar% {
      D1 <- Dw
      for (k in 1:(length(nodewise.transfer.source.id)+1)) {
        if (k > 1) {
          D1$source[[nodewise.transfer.source.id[k-1]]]$y <- D1$source[[nodewise.transfer.source.id[k-1]]]$x[, j]
          D1$source[[nodewise.transfer.source.id[k-1]]]$x <- D1$source[[nodewise.transfer.source.id[k-1]]]$x[, -j]
        } else {
          D1$target$y <- D1$target$x[, j]
          D1$target$x <- D1$target$x[, -j]
        }
      }
      node.lasso <- glmtrans(target = D1$target, source = D1$source, family = "gaussian", alg = "ori", transfer.source.id = nodewise.transfer.source.id, intercept = FALSE, ...)
      gamma <- node.lasso$beta[-1]
      tau2 <- Sigma.hat[j, j] - Sigma.hat[j, -j, drop = F] %*% gamma
      theta <- rep(1, p)
      theta[-j] <- -gamma
      c(theta, tau2)
    }

    Theta.hat <- solve(diag(L[,p+1])) %*% L[1:p, 1:p]
    u.target <- D$target$x %*% beta.hat[-1] + beta.hat[1]
    Z <- D$target$y - exp(u.target)

    if (intercept) {
      b.hat <- as.matrix(beta.hat) + Theta.hat %*% t(cbind(1, D$target$x)) %*% Z/length(D$target$y)
    } else {
      b.hat <- beta.hat[-1] + Theta.hat %*% t(D$target$x) %*% Z/length(D$target$y)
    }

    var.est <- diag(Theta.hat %*% Sigma.hat %*% t(Theta.hat))
    CI <- data.frame(b.hat = b.hat, lb = b.hat - qnorm(r.level)*sqrt(var.est/length(D$target$y)), ub =  b.hat + qnorm(r.level)*sqrt(var.est/length(D$target$y)))

  }

  stopImplicitCluster()
  # stopCluster(cl)

  if (!is.null(beta.hat.names)) {
    rownames(CI) <- beta.hat.names
    names(var.est) <- beta.hat.names
    names(b.hat) <- beta.hat.names
    names(beta.hat) <- beta.hat.names
  }
  return(list(b.hat = b.hat, beta.hat = beta.hat, CI = CI, var.est = var.est))
}
