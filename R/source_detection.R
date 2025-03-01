#' Transferable source detection for GLM transfer learning algorithm.
#'
#' Detect transferable sources from multiple source data sets. Currently can deal with Gaussian, logistic and Poisson models.
#' @export
#' @param target target data. Should be a list with elements x and y, where x indicates a predictor matrix with each row/column as a(n) observation/variable, and y indicates the response vector.
#' @param source source data. Should be a list with some sublists, where each of the sublist is a source data set, having elements x and y with the same meaning as in target data.
#' @param family response type. Can be "gaussian", "binomial" or "poisson". Default = "gaussian".
#' \itemize{
#' \item "gaussian": Gaussian distribution.
#' \item "binomial": logistic distribution. When \code{family = "binomial"}, the input response in both \code{target} and \code{source} should be 0/1.
#' \item "poisson": poisson distribution. When \code{family = "poisson"}, the input response in both \code{target} and \code{source} should be non-negative.
#' }
#' @param alpha the elasticnet mixing parameter, with \eqn{0 \leq \alpha \leq 1}. The penality is defined as \deqn{(1-\alpha)/2||\beta||_2^2+\alpha ||\beta||_1}. \code{alpha = 1} encodes the lasso penalty while \code{alpha = 0} encodes the ridge penalty. Default = 1.
#' @param standardize the logical flag for x variable standardization, prior to fitting the model sequence. The coefficients are always returned on the original scale. Default is \code{TRUE}.
#' @param intercept the logical indicator of whether the intercept should be fitted or not. Default = \code{TRUE}.
#' @param nfolds the number of folds. Used in the cross-validation for GLM elastic net fitting procedure. Default = 10. Smallest value allowable is \code{nfolds = 3}.
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param valid.nfolds the number of folds used in cross-validation procedure when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"}. Default = 3.
#' @param lambda lambda (the penalty parameter) used in the transferable source detection algorithm. Can be either "lambda.min" or "lambda.1se". Default = "lambda.1se".
#' @param lambda.seq the sequence of lambda candidates used in the algorithm. Should be a vector of numerical values. Default = NULL, which means the algorithm will determine the sequence automatically, based on the same method used in \code{cv.glmnet}.
#' @param target.weights weight vector for each target instance. Should be a vector with the same length of target response. Default = \code{NULL}, which makes all instances equal-weighted.
#' @param source.weights a list of weight vectors for the instances from each source. Should be a list with the same length of the number of sources. Default = \code{NULL}, which makes all instances equal-weighted.
#' @param C0 the constant used in the transferable source detection algorithm. See Algorithm 2 in Tian, Y. and Feng, Y., 2021. Default = 2.
#' \itemize{
#' \item "lambda.min": value of lambda that gives minimum mean cross-validated error in the sequence of lambda.
#' \item "lambda.1se": largest value of lambda such that error is within 1 standard error of the minimum.
#' }
#' @param detection.info the logistic flag indicating whether to print detection information or not. Useful only when \code{transfer.source.id = "auto"}. Default = \code{TURE}.
#' @param ... additional arguments.
#' @return An object with S3 class \code{"glmtrans_source_detection"}.
#' \item{transfer.source.id}{the index of transferable sources.}
#' \item{source.loss}{the loss on each source data. Only available when \code{transfer.source.id = "auto"}.}
#' \item{target.valid.loss}{the validation (or cross-validation) loss on target data. Only available when \code{transfer.source.id = "auto"}.}
#' \item{threshold}{the threshold to determine transferability. Only available when \code{transfer.source.id = "auto"}.}
#' @seealso \code{\link{glmtrans}}, \code{\link{predict.glmtrans}}, \code{\link{models}}, \code{\link{plot.glmtrans}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[glmnet]{glmnet}}.
#' @note \code{source.loss} and \code{threshold} outputed by \code{source_detection} can be visualized by function \code{plot.glmtrans}.
#' @references
#' Tian, Y., & Feng, Y. (2023). \emph{Transfer learning under high-dimensional generalized linear models. Journal of the American Statistical Association, 118(544), 2684-2697.}
#'
#' Li, S., Cai, T.T. & Li, H., (2020). \emph{Transfer learning for high-dimensional linear regression: Prediction, estimation, and minimax optimality. arXiv preprint arXiv:2006.10593.}
#'
#' Friedman, J., Hastie, T. & Tibshirani, R., (2010). \emph{Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), p.1.}
#'
#' Zou, H. & Hastie, T., (2005). \emph{Regularization and variable selection via the elastic net. Journal of the royal statistical society: series B (statistical methodology), 67(2), pp.301-320.}
#'
#' Tibshirani, R., (1996). \emph{Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), pp.267-288.}
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#'
#' # study the linear model
#' D.training <- models("gaussian", type = "all", K = 2, p = 500, Ka = 1, n.target = 100, cov.type = 2)
#' detection.gaussian <- source_detection(D.training$target, D.training$source)
#' detection.gaussian$transfer.source.id
#'
#' \donttest{
#' # study the logistic model
#' D.training <- models("binomial", type = "all", K = 2, p = 500, Ka = 1, n.target = 100, cov.type = 2)
#' detection.binomial <- source_detection(D.training$target, D.training$source,
#' family = "binomial", cores = 2)
#' detection.binomial$transfer.source.id
#'
#'
#' # study Poisson model
#' D.training <- models("poisson", type = "all", K = 2, p = 500, Ka = 1, n.target = 100, cov.type = 2)
#' detection.poisson <- source_detection(D.training$target, D.training$source,
#' family = "poisson", cores = 2)
#' detection.poisson$transfer.source.id
#' }
source_detection <- function(target, source = NULL, family = c("gaussian", "binomial", "poisson"), alpha = 1, standardize = TRUE,
                             intercept = TRUE, nfolds = 10, cores = 1, valid.nfolds = 3,
                             lambda = "lambda.1se", lambda.seq = NULL, detection.info = TRUE, target.weights = NULL, source.weights = NULL, C0 = 2, ...) {

  family <- match.arg(family)

  if (is.null(target.weights)) {
    target.weights <- rep(1, length(target$y))
  }

  if (is.null(source.weights)) {
    source.weights <- sapply(1:length(source), function(i){
      rep(1, length(source[[i]]$y))
    }, simplify = FALSE)
  }



  if (cores > 1) {
    registerDoParallel(cores)
  }

  if (family != "binomial") {
    folds <- createFolds(target$y, valid.nfolds)
  } else {
    folds <- createFolds_binary(target$y, valid.nfolds)
  }

  num_try <- 20

  # evaluate loss for each lambda candidate through CV
  loss.cv <- t(sapply(1:valid.nfolds, function(i){
    source.loss <- sapply(1:length(source), function(k){
      if (lambda == "lambda.1se") {
        n_try <- 0
        while(T) {
          n_try <- n_try + 1
          wa <- try(coef(cv.glmnet(x = as.matrix(rbind(target$x[-folds[[i]], , drop = F], source[[k]]$x)), y = c(target$y[-folds[[i]]], source[[k]]$y), weights = c(target.weights[-folds[[i]]], source.weights[[k]]), family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, nfolds = nfolds, lambda = lambda.seq, ...)), silent = TRUE)
          # wa <- try(coef(cv.glmnet(x = as.matrix(source[[k]]$x), y = source[[k]]$y, family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, nfolds = nfolds, ...)), silent=TRUE)
          if (!inherits(wa, "try-error")) {
            break
          } else if (n_try > 20) {
            wa <- numeric(ncol(target$x)+1)
            break
          }

        }
      } else if (lambda == "lambda.min") {
        n_try <- 0
        while(T) {
          n_try <- n_try + 1
          wa.cv <- try(cv.glmnet(x = as.matrix(rbind(target$x[-folds[[i]], , drop = F], source[[k]]$x)), y = c(target$y[-folds[[i]]], source[[k]]$y), weights = c(target.weights[-folds[[i]]], source.weights[[k]]), family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, nfolds = nfolds, lambda = lambda.seq, ...), silent = TRUE)
          # wa.cv <- try(cv.glmnet(x = as.matrix(source[[k]]$x), y = source[[k]]$y, family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, nfolds = nfolds, ...), silent=TRUE)
          if (!inherits(wa.cv, "try-error")) {
            wa <- c(wa.cv$glmnet.fit$a0[which(wa.cv$lambda == wa.cv$lambda.min)], wa.cv$glmnet.fit$beta[, which(wa.cv$lambda == wa.cv$lambda.min)])
            break
          } else if (n_try > 20) {
            wa <- numeric(ncol(target$x)+1)
            break
          }
        }

      }

      loss(wa, as.matrix(target$x[folds[[i]], , drop = F]), target$y[folds[[i]]], family)
    })

    # calculate the estimator for each lambda candidate
    if (lambda == "lambda.1se") {
      n_try <- 0
      while(T) {
        n_try <- n_try + 1
        wa.target <- try(coef(cv.glmnet(x = as.matrix(target$x[-folds[[i]], , drop = F]), y = target$y[-folds[[i]]], weights = target.weights[-folds[[i]]], family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, nfolds = nfolds, lambda = lambda.seq, ...)), silent=TRUE)
        if (!inherits(wa.target, "try-error")) {
          break
        } else if (n_try > 20) {
          wa.target <- numeric(ncol(target$x)+1)
          break
        }
      }
    } else if (lambda == "lambda.min") {
      n_try <- 0
      while(T) {
        n_try <- n_try + 1
        wa.cv <- try(cv.glmnet(x = as.matrix(target$x[-folds[[i]], , drop = F]), y = target$y[-folds[[i]]], weights = target.weights[-folds[[i]]], family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, nfolds = nfolds, lambda = lambda.seq, ...), silent=TRUE)
        if (!inherits(wa.cv, "try-error")) {
          wa.target <- c(wa.cv$glmnet.fit$a0[which(wa.cv$lambda == wa.cv$lambda.min)], wa.cv$glmnet.fit$beta[, which(wa.cv$lambda == wa.cv$lambda.min)])
          break
        } else if (n_try > 20) {
          wa.target <- numeric(ncol(target$x)+1)
          break
        }
      }

    }
    target.loss <- loss(wa.target, as.matrix(target$x[folds[[i]], , drop = F]), target$y[folds[[i]]], family)
    c(source.loss, target.loss)
  }))

  source.loss <- colMeans(loss.cv)[1:(ncol(loss.cv)-1)]
  target.valid.loss <- colMeans(loss.cv)[ncol(loss.cv)]
  target.valid.loss.sd <- sd(loss.cv[, ncol(loss.cv)])

  threshold <- target.valid.loss + C0*max(target.valid.loss.sd, 0.01)
  transfer.source.id <- which(source.loss <= threshold)
  if (detection.info) {
    cat(paste0("Loss difference between source data and the threshold: (negative to be transferable)", "\n"))
    for (i in 1:length(source.loss)) {
      cat(paste0("Source ", i, ": ", format(round(source.loss[i]-threshold, 6), nsmall = 6) , "\n"))
    }
    cat("\n")
    cat(paste("Source data set(s)", paste(transfer.source.id, collapse = ", "), "are transferable!\n"))
  }

  if(cores > 1) {
    stopImplicitCluster()
  }


  obj <- list(transfer.source.id = transfer.source.id, source.loss = source.loss, target.valid.loss = target.valid.loss,
              threshold = threshold)
  class(obj) <- "glmtrans_source_detection"
  return(obj)
}
