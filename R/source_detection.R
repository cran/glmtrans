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
#' @param epsilon0 a positive number. Useful only when \code{transfer.source.id = "auto"}. The threshold to determine transferability will be set as \eqn{(1+epsilon0)*(validation (or cross-validation) loss of target data)}. Default = 0.01. For details, refer to Algorithm 3 in Tian, Y. and Feng, Y., 2021.
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param valid.proportion the proportion of target data to be used as validation data when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"}. Default = \code{NULL}, meaning that the cross-validation will be applied.
#' @param valid.nfolds the number of folds used in cross-validation procedure when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"} and \code{valid.proportion = NULL}. Default = 3.
#' @param lambda.detection lambda (the penalty parameter) used in the transferable source detection algorithm. Can be either "lambda.min" or "lambda.1se". Default = "lambda.min".
#' \itemize{
#' \item "lambda.min": value of lambda that gives minimum mean cross-validated error in the sequence of lambda.
#' \item "lambda.1se": largest value of lambda such that error is within 1 standard error of the minimum.
#' }
#' @param detection.info the logistic flag indicating whether to print detection information or not. Useful only when \code{transfer.source.id = "auto"}. Default = \code{TURE}.
#' @param ... additional arguments.
#' @return An object with S3 class \code{"glmtrans_source_detection"}.
#' \item{target.valid.loss}{the validation (or cross-validation) loss on target data. Only available when \code{transfer.source.id = "auto"}.}
#' \item{source.loss}{the loss on each source data. Only available when \code{transfer.source.id = "auto"}.}
#' \item{epsilon0}{the threshold to determine transferability will be set as \eqn{(1+epsilon0)*loss of validation (cv) target data}. Only available when \code{transfer.source.id = "auto"}.}
#' \item{threshold}{the threshold to determine transferability. Only available when \code{transfer.source.id = "auto"}.}
#' @seealso \code{\link{glmtrans}}, \code{\link{predict.glmtrans}}, \code{\link{models}}, \code{\link{plot.glmtrans}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[glmnet]{glmnet}}.
#' @note \code{source.loss} and \code{threshold} outputed by \code{source_detection} can be visualized by function \code{plot.glmtrans}.
#' @references
#' Tian, Y. and Feng, Y., 2021. \emph{Transfer learning with high-dimensional generalized linear models. Submitted.}
#'
#' Li, S., Cai, T.T. and Li, H., 2020. \emph{Transfer learning for high-dimensional linear regression: Prediction, estimation, and minimax optimality. arXiv preprint arXiv:2006.10593.}
#'
#' Friedman, J., Hastie, T. and Tibshirani, R., 2010. \emph{Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), p.1.}
#'
#' Zou, H. and Hastie, T., 2005. \emph{Regularization and variable selection via the elastic net. Journal of the royal statistical society: series B (statistical methodology), 67(2), pp.301-320.}
#'
#' Tibshirani, R., 1996. \emph{Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), pp.267-288.}
#'
#' @examples
#' set.seed(1, kind = "L'Ecuyer-CMRG")
#'
#' # study the linear model
#' D.training <- models("gaussian", type = "all", K = 2, p = 500, Ka = 1)
#' detection.gaussian <- source_detection(D.training$target, D.training$source)
#' detection.gaussian$transferable.source.id
#'
#' \donttest{
#' # study the logistic model
#' D.training <- models("binomial", type = "all", p = 500)
#' detection.binomial <- source_detection(D.training$target, D.training$source,
#' family = "binomial", cores = 2)
#' detection.binomial$transferable.source.id
#'
#'
#' # study Poisson model
#' D.training <- models("poisson", type = "all", p = 200)
#' detection.poisson <- source_detection(D.training$target, D.training$source,
#' family = "poisson", cores = 2)
#' detection.poisson$transferable.source.id
#' }
source_detection <- function(target, source = NULL, family = c("gaussian", "binomial", "poisson"), alpha = 1, standardize = TRUE,
                             intercept = TRUE, nfolds = 10, epsilon0 = 0.01, cores = 1, valid.proportion = NULL, valid.nfolds = 3,
                             lambda.detection = "lambda.min", detection.info = TRUE, ...) {

  family <- match.arg(family)

  if (cores > 1) {
    registerDoParallel(cores)
  }


  if (!is.null(valid.proportion)) {
    target.valid.id <- sample(1:length(target$y), size = floor(length(target$y)*valid.proportion))
    if (lambda.detection == "lambda.1se") {
      wa.target <- coef(cv.glmnet(x = as.matrix(target$x[-target.valid.id, , drop = F]), y = target$y[-target.valid.id], family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...))
    } else {
      wa.cv <- cv.glmnet(x = as.matrix(target$x[-target.valid.id, , drop = F]), y = target$y[-target.valid.id], family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...)
      wa.target <- c(wa.cv$glmnet.fit$a0[which(wa.cv$lambda == wa.cv$lambda.min)], wa.cv$glmnet.fit$beta[, which(wa.cv$lambda == wa.cv$lambda.min)])
    }
    target.valid.loss <- loss(wa.target, as.matrix(target$x[target.valid.id, , drop = F]), target$y[target.valid.id], family)
    source.loss <- sapply(1:length(source), function(k){
      if (lambda.detection == "lambda.1se") {
        wa <- coef(cv.glmnet(x = as.matrix(rbind(target$x[-target.valid.id, , drop = F], source[[k]]$x)), y = c(target$y[-target.valid.id], source[[k]]$y), family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...))
      } else {
        wa.cv <- cv.glmnet(x = as.matrix(rbind(target$x[-target.valid.id, , drop = F], source[[k]]$x)), y = c(target$y[-target.valid.id], source[[k]]$y), family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...)
        wa <- c(wa.cv$glmnet.fit$a0[which(wa.cv$lambda == wa.cv$lambda.min)], wa.cv$glmnet.fit$beta[, which(wa.cv$lambda == wa.cv$lambda.min)])
      }
      loss(wa, as.matrix(target$x[target.valid.id, , drop = F]), target$y[target.valid.id], family)
    })
  } else { # valid.proportion == NULL, cross-validation
    folds <- createFolds(target$y, valid.nfolds)
    loss.cv <- t(sapply(1:valid.nfolds, function(i){
      source.loss <- sapply(1:length(source), function(k){
        if (lambda.detection == "lambda.1se") {
          wa <- coef(cv.glmnet(x = as.matrix(rbind(target$x[-folds[[i]], , drop = F], source[[k]]$x)), y = c(target$y[-folds[[i]]], source[[k]]$y), family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...))
        } else {
          wa.cv <- cv.glmnet(x = as.matrix(rbind(target$x[-folds[[i]], , drop = F], source[[k]]$x)), y = c(target$y[-folds[[i]]], source[[k]]$y), family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...)
          wa <- c(wa.cv$glmnet.fit$a0[which(wa.cv$lambda == wa.cv$lambda.min)], wa.cv$glmnet.fit$beta[, which(wa.cv$lambda == wa.cv$lambda.min)])
        }
        loss(wa, as.matrix(target$x[folds[[i]], , drop = F]), target$y[folds[[i]]], family)
      })

      if (lambda.detection == "lambda.1se") {
        wa.target <- coef(cv.glmnet(x = as.matrix(target$x[-folds[[i]], , drop = F]), y = target$y[-folds[[i]]], family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...))
      } else {
        wa.cv <- cv.glmnet(x = as.matrix(target$x[-folds[[i]], , drop = F]), y = target$y[-folds[[i]]], family = family, alpha = alpha, parallel = I(cores > 1), standardize = standardize, intercept = intercept, ...)
        wa.target <- c(wa.cv$glmnet.fit$a0[which(wa.cv$lambda == wa.cv$lambda.min)], wa.cv$glmnet.fit$beta[, which(wa.cv$lambda == wa.cv$lambda.min)])
      }
      target.loss <- loss(wa.target, as.matrix(target$x[folds[[i]], , drop = F]), target$y[folds[[i]]], family)
      c(source.loss, target.loss)
    }))
    source.loss <- colMeans(loss.cv)[1:(ncol(loss.cv)-1)]
    target.valid.loss <- colMeans(loss.cv)[ncol(loss.cv)]
  }




  transfer.source.id <- which(source.loss <= (1+epsilon0)*target.valid.loss)
  threshold <- (1+epsilon0)*target.valid.loss
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

  obj <- list(transfer.source.id = transfer.source.id, source.loss = source.loss, target.valid.loss = target.valid.loss, epsilon0 = epsilon0, threshold = threshold)
  class(obj) <- "glmtrans_source_detection"
  return(obj)
}
