#' Fit a transfer learning generalized linear model (GLM) with elasticnet regularization.
#'
#' Fit a transfer learning generalized linear model through elastic net regularization with target data set and multiple source data sets. It also implements a transferable source detection algorithm, which helps avoid negative transfer in practice. Currently can deal with Gaussian, logistic and Poisson models.
#' @export
#' @importFrom caret createFolds
#' @importFrom doParallel registerDoParallel
#' @importFrom doParallel stopImplicitCluster
#' @importFrom foreach foreach
#' @importFrom foreach %dopar%
#' @importFrom foreach %do%
#' @importFrom parallel detectCores
#' @importFrom stats predict
#' @importFrom stats rnorm
#' @importFrom stats rpois
#' @importFrom stats dpois
#' @importFrom stats coef
#' @importFrom glmnet glmnet
#' @importFrom glmnet cv.glmnet
#' @importFrom glmnet predict.glmnet
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 aes_string
#' @importFrom ggplot2 geom_point
#' @importFrom ggplot2 ylim
#' @importFrom ggplot2 geom_line
#' @importFrom assertthat is.string
#' @importFrom formatR tidy_eval
#' @param target target data. Should be a list with elements x and y, where x indicates a predictor matrix with each row/column as a(n) observation/variable, and y indicates the response vector.
#' @param source source data. Should be a list with some sublists, where each of the sublist is a source data set, having elements x and y with the same meaning as in target data.
#' @param family response type. Can be "gaussian", "binomial" or "poisson". Default = "gaussian".
#' \itemize{
#' \item "gaussian": Gaussian distribution.
#' \item "binomial": logistic distribution. When \code{family = "binomial"}, the input response in both \code{target} and \code{source} should be 0/1.
#' \item "poisson": poisson distribution. When \code{family = "poisson"}, the input response in both \code{target} and \code{source} should be non-negative.
#' }
#' @param transfer.source.id transferable source index. Can be either a subset of \code{{1, ..., length(source)}}, "all" or "auto". Default = \code{"auto"}.
#' \itemize{
#' \item a subset of \code{{1, ..., length(source)}}: only transfer sources with the specific index.
#' \item "all": transfer all sources.
#' \item "auto": run transferable source detection algorithm to automatically detect which sources to transfer. For the algorithm, refer to the documentation of function \code{source_detection}.
#' }
#' @param alpha the elasticnet mixing parameter, with \eqn{0 \leq \alpha \leq 1}. The penality is defined as \deqn{(1-\alpha)/2||\beta||_2^2+\alpha ||\beta||_1}. \code{alpha = 1} encodes the lasso penalty while \code{alpha = 0} encodes the ridge penalty. Default = 1.
#' @param standardize the logical flag for x variable standardization, prior to fitting the model sequence. The coefficients are always returned on the original scale. Default is \code{TRUE}.
#' @param intercept the logical indicator of whether the intercept should be fitted or not. Default = \code{TRUE}.
#' @param nfolds the number of folds. Used in the cross-validation for GLM elastic net fitting procedure. Default = 10. Smallest value allowable is \code{nfolds = 3}.
#' @param epsilon0 a positive number. Useful only when \code{transfer.source.id = "auto"}. The threshold to determine transferability will be set as \eqn{(1+epsilon0)*(validation (or cross-validation) loss of target data)}. Default = 0.01. For details, refer to Algorithm 3 in Tian, Y. and Feng, Y., 2021.
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param valid.proportion the proportion of target data to be used as validation data when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"}. Default = \code{NULL}, meaning that the cross-validation will be applied.
#' @param valid.nfolds the number of folds used in cross-validation procedure when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"} and \code{valid.proportion = NULL}. Default = 3.
#' @param lambda.transfer lambda (the penalty parameter) used in transferrring step. Can be either "lambda.min" or "lambda.1se". Default = "lambda.1se". The sequence of lambda will be genenated automatically by \code{cv.glmnet}. For more details about lambda choice, see the documentation of \code{cv.glmnet} in package \code{glmnet}.
#' \itemize{
#' \item "lambda.min": value of lambda that gives minimum mean cross-validated error in the sequence of lambda.
#' \item "lambda.1se": largest value of lambda such that error is within 1 standard error of the minimum.
#' }
#' @param lambda.debias lambda (the penalty parameter) used in debiasing step. Can be either "lambda.min" or "lambda.1se". Default = "lambda.min".
#' @param lambda.detection lambda (the penalty parameter) used in the transferable source detection algorithm. Can be either "lambda.min" or "lambda.1se". Default = "lambda.min".
#' @param detection.info the logistic flag indicating whether to print detection information or not. Useful only when \code{transfer.source.id = "auto"}. Default = \code{TURE}.
#' @param ... additional arguments.
#' @return An object with S3 class \code{"glmtrans"}.
#' \item{beta}{the estimated coefficient vector.}
#' \item{family}{the response type.}
#' \item{transfer.source.id}{the transferable souce index. If in the input, \code{transfer.source.id = 1:length(source)} or \code{transfer.source.id = "all"}, then the outputed \code{transfer.source.id = 1:length(source)}. If the inputed \code{transfer.source.id = "auto"}, only transferable source detected by the algorithm will be outputed.}
#' \item{fitting.list}{a list of other parameters of the fitted model.}
#' \itemize{
#' \item{w_a}{the estimator obtained from the transferring step.}
#' \item{delta_a}{the estimator obtained from the debiasing step.}
#' \item{target.valid.loss}{the validation (or cross-validation) loss on target data. Only available when \code{transfer.source.id = "auto"}.}
#' \item{source.loss}{the loss on each source data. Only available when \code{transfer.source.id = "auto"}.}
#' \item{epsilon0}{the threshold to determine transferability will be set as \eqn{(1+epsilon0)*loss of validation (cv) target data}. Only available when \code{transfer.source.id = "auto"}.}
#' \item{threshold}{the threshold to determine transferability. Only available when \code{transfer.source.id = "auto"}.}
#' }
#' @seealso \code{\link{predict.glmtrans}}, \code{\link{source_detection}}, \code{\link{models}}, \code{\link{plot.glmtrans}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[glmnet]{glmnet}}.
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
#' # fit a linear model
#' D.training <- models("gaussian", type = "all", K = 2, p = 500)
#' D.test <- models("gaussian", type = "target", n.target = 100, p = 500)
#' fit.gaussian <- glmtrans(D.training$target, D.training$source)
#' y.pred.glmtrans <- predict(fit.gaussian, D.test$target$x)
#'
#' # compare the test MSE with classical Lasso fitted on target data
#' library(glmnet)
#' fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y)
#' y.pred.lasso <- predict(fit.lasso, D.test$target$x)
#'
#' mean((y.pred.glmtrans - D.test$target$y)^2)
#' mean((y.pred.lasso - D.test$target$y)^2)
#'
#' \donttest{
#' # fit a logistic model
#' D.training <- models("binomial", type = "all", K = 2, p = 500)
#' D.test <- models("binomial", type = "target", n.target = 100, p = 500)
#' fit.binomial <- glmtrans(D.training$target, D.training$source, family = "binomial")
#' y.pred.glmtrans <- predict(fit.binomial, D.test$target$x, type = "class")
#'
#' # compare the test error with classical Lasso fitted on target data
#' library(glmnet)
#' fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y, family = "binomial")
#' y.pred.lasso <- as.numeric(predict(fit.lasso, D.test$target$x, type = "class"))
#'
#' mean(y.pred.glmtrans != D.test$target$y)
#' mean(y.pred.lasso != D.test$target$y)
#'
#'
#' # fit a Poisson model
#' D.training <- models("poisson", type = "all", K = 2, p = 500)
#' D.test <- models("poisson", type = "target", n.target = 100, p = 500)
#' fit.poisson <- glmtrans(D.training$target, D.training$source, family = "poisson")
#' y.pred.glmtrans <- predict(fit.poisson, D.test$target$x, type = "response")
#'
#' # compare the test MSE with classical Lasso fitted on target data
#' fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y, family = "poisson")
#' y.pred.lasso <- as.numeric(predict(fit.lasso, D.test$target$x, type = "response"))
#'
#' mean((y.pred.glmtrans - D.test$target$y)^2)
#' mean((y.pred.lasso - D.test$target$y)^2)
#' }

glmtrans <- function(target, source = NULL, family = c("gaussian", "binomial", "poisson"),
                     transfer.source.id = "auto", alpha = 1, standardize = TRUE, intercept = TRUE, nfolds = 10, epsilon0 = 0.01,
                     cores = 1, valid.proportion = NULL, valid.nfolds = 3, lambda.transfer = "lambda.1se",
                     lambda.debias = "lambda.min", lambda.detection = "lambda.min", detection.info = TRUE, ...) {

  family <- match.arg(family)
  transfer.source.id.ori <- transfer.source.id
  data <- c(target, source) # to be updated
  k <- NULL




  if (is.null(transfer.source.id) || (is.string(transfer.source.id) && transfer.source.id == "all")) { # transfer all source data
    transfer.source.id <- 1:length(source)

  } else if ((is.string(transfer.source.id) && transfer.source.id == "auto")) { # automatically check which source data set to transfer
    A <- source_detection(target = target, family = family, source = source, alpha = alpha, epsilon0 = epsilon0,
                         cores = cores,  lambda.detection = lambda.detection, valid.proportion = valid.proportion,
                         valid.nfolds = valid.nfolds, detection.info = detection.info, standardize = standardize,
                         intercept = intercept, ...)
    transfer.source.id <- A$transfer.source.id
  } else if (0 %in% transfer.source.id) { # don't transfer any source
    transfer.source.id <- 0
  }



  family <- match.arg(family)

  # transferring step
  # --------------------------------------



  all.x <- as.matrix(foreach(k = unique(c(0, transfer.source.id)), .combine = "rbind") %do% {
    if (k != 0) {
      source[[k]]$x
    } else {
      target$x
    }
  })

  all.y <- foreach(k = unique(c(0, transfer.source.id)), .combine = "c") %do% {
    if (k != 0) {
      source[[k]]$y
    } else {
      target$y
    }
  }

  if (cores > 1) {
    registerDoParallel(cores)
  }


  cv.fit.trans <- cv.glmnet(x = all.x, y = all.y, family = family, alpha = alpha, nfolds = nfolds, parallel = I(cores > 1), intercept = intercept, ...)
  if (lambda.transfer == "lambda.1se") {
    wa <- as.numeric(coef(cv.fit.trans))
  } else if (lambda.transfer == "lambda.min") {
    wa <- c(cv.fit.trans$glmnet.fit$a0[which(cv.fit.trans$lambda == cv.fit.trans$lambda.min)], cv.fit.trans$glmnet.fit$beta[, which(cv.fit.trans$lambda == cv.fit.trans$lambda.min)])
  }





  # bias correcting step
  # --------------------------------------

  cv.fit.correct <- cv.glmnet(x = as.matrix(target$x), y = target$y, offset = (as.matrix(target$x) %*% wa[-1]+wa[1]), family = family, parallel = I(cores > 1), intercept = intercept, ...)

  if (lambda.debias == "lambda.1se") {
    deltaa <- as.numeric(coef(cv.fit.correct))
  } else {
    deltaa <- c(cv.fit.correct$glmnet.fit$a0[which(cv.fit.correct$lambda == cv.fit.correct$lambda.min)], cv.fit.correct$glmnet.fit$beta[, which(cv.fit.correct$lambda == cv.fit.correct$lambda.min)])
  }

  beta.hat <- wa + deltaa
  if (!all(is.null(colnames(target$x)))) {
    names(beta.hat) <- colnames(target$x)
  } else {
    names(beta.hat) <- c("intercept", paste0("V", 1:ncol(target$x)))
  }
  if(cores > 1) {
    stopImplicitCluster()
  }

  if (is.string(transfer.source.id.ori) && transfer.source.id.ori == "auto") {
    obj <- list(beta = beta.hat, family = family, transfer.source.id = transfer.source.id, fitting.list = list(w_a = wa, delta_a = deltaa, target.valid.loss = A$target.valid.loss, source.loss = A$source.loss, epsilon0 = epsilon0, threshold = A$threshold))
  } else {
    obj <- list(beta = beta.hat, family = family, transfer.source.id = transfer.source.id, fitting.list = list(w_a = wa, delta_a = deltaa))
  }
  class(obj) <- "glmtrans"
  return(obj)
}
