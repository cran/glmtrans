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
#' @importFrom parallel makeCluster
#' @importFrom parallel stopCluster
#' @importFrom stats predict
#' @importFrom stats rnorm
#' @importFrom stats rt
#' @importFrom stats sd
#' @importFrom stats qnorm
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
#' @param transfer.source.id transferable source indices. Can be either a subset of \code{{1, ..., length(source)}}, "all" or "auto". Default = \code{"auto"}.
#' \itemize{
#' \item a subset of \code{{1, ..., length(source)}}: only transfer sources with the specific indices.
#' \item "all": transfer all sources.
#' \item "auto": run transferable source detection algorithm to automatically detect which sources to transfer. For the algorithm, refer to the documentation of function \code{source_detection}.
#' }
#' @param alpha the elasticnet mixing parameter, with \eqn{0 \leq \alpha \leq 1}. The penality is defined as \deqn{(1-\alpha)/2||\beta||_2^2+\alpha ||\beta||_1}. \code{alpha = 1} encodes the lasso penalty while \code{alpha = 0} encodes the ridge penalty. Default = 1.
#' @param standardize the logical flag for x variable standardization, prior to fitting the model sequence. The coefficients are always returned on the original scale. Default is \code{TRUE}.
#' @param intercept the logical indicator of whether the intercept should be fitted or not. Default = \code{TRUE}.
#' @param nfolds the number of folds. Used in the cross-validation for GLM elastic net fitting procedure. Default = 10. Smallest value allowable is \code{nfolds = 3}.
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param valid.proportion the proportion of target data to be used as validation data when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"}. Default = \code{NULL}, meaning that the cross-validation will be applied.
#' @param valid.nfolds the number of folds used in cross-validation procedure when detecting transferable sources. Useful only when \code{transfer.source.id = "auto"} and \code{valid.proportion = NULL}. Default = 3.
#' @param lambda a vector indicating the choice of lambdas in transferring, debiasing and detection steps. Should be a vector with names "transfer", "debias", and "detection", each component of which can be either "lambda.min" or "lambda.1se". Component \code{transfer} is the lambda (the penalty parameter) used in transferrring step. Component \code{debias} is the lambda used in debiasing step. Component \code{detection} is the lambda used in the transferable source detection algorithm. Default choice of \code{lambda.transfer} and \code{lambda.detection} are "lambda.1se", while default \code{lambda.debias} = "lambda.min". If the user wants to change the default setting, input a vector with corresponding \code{lambda.transfer}/\code{lambda.debias}/\code{lambda.detection} names and corresponding values. Examples: lambda = list(transfer = "lambda.1se", debias = "lambda.min", detection = "lambda.1se"); lambda = list(transfer = 1, debias = 0.5, detection = 1).
#' \itemize{
#' \item "lambda.min": value of lambda that gives minimum mean cross-validated error in the sequence of lambda.
#' \item "lambda.1se": largest value of lambda such that error is within 1 standard error of the minimum.
#' }
#' @param lambda.seq the sequence of lambda candidates used in the algorithm. Should be a list of three vectors with names "transfer", "debias", and "detection". Default = list(transfer = NULL, debias = NULL, detection = NULL). "NULL" means the algorithm will determine the sequence automatically, based on the same method used in \code{cv.glmnet}.
#' @param detection.info the logistic flag indicating whether to print detection information or not. Useful only when \code{transfer.source.id = "auto"}. Default = \code{TURE}.
#' @param target.weights weight vector for each target instance. Should be a vector with the same length of target response. Default = \code{NULL}, which makes all instances equal-weighted.
#' @param source.weights a list of weight vectors for the instances from each source. Should be a list with the same length of the number of sources. Default = \code{NULL}, which makes all instances equal-weighted.
#' @param C0 the constant used in the transferable source detection algorithm. See Algorithm 2 in Tian, Y. & Feng, Y. (2023). Default = 2.
#' @param ... additional arguments.
#' @return An object with S3 class \code{"glmtrans"}.
#' \item{beta}{the estimated coefficient vector.}
#' \item{family}{the response type.}
#' \item{transfer.source.id}{the transferable source index. If in the input, \code{transfer.source.id = 1:length(source)} or \code{transfer.source.id = "all"}, then the outputed \code{transfer.source.id = 1:length(source)}. If the inputed \code{transfer.source.id = "auto"}, only transferable source detected by the algorithm will be outputed.}
#' \item{fitting.list}{a list of other parameters of the fitted model.}
#' \itemize{
#' \item \code{w_a}: the estimator obtained from the transferring step.
#' \item \code{delta_a}: the estimator obtained from the debiasing step.
#' \item \code{target.valid.loss}: the validation (or cross-validation) loss on target data. Only available when \code{transfer.source.id = "auto"}.
#' \item \code{source.loss}: the loss on each source data. Only available when \code{transfer.source.id = "auto"}.
#' \item \code{threshold}: the threshold to determine transferability. Only available when \code{transfer.source.id = "auto"}.
#' }
#' @seealso \code{\link{predict.glmtrans}}, \code{\link{source_detection}}, \code{\link{models}}, \code{\link{plot.glmtrans}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[glmnet]{glmnet}}.
#' @references
#' Tian, Y., & Feng, Y. (2023). \emph{Transfer learning under high-dimensional generalized linear models. Journal of the American Statistical Association, 118(544), 2684-2697.}
#'
#' Li, S., Cai, T.T. & Li, H. (2020). \emph{Transfer learning for high-dimensional linear regression: Prediction, estimation, and minimax optimality. arXiv preprint arXiv:2006.10593.}
#'
#' Friedman, J., Hastie, T. & Tibshirani, R. (2010). \emph{Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), p.1.}
#'
#' Zou, H. & Hastie, T. (2005). \emph{Regularization and variable selection via the elastic net. Journal of the royal statistical society: series B (statistical methodology), 67(2), pp.301-320.}
#'
#' Tibshirani, R. (1996). \emph{Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), pp.267-288.}
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#'
#' # fit a linear regression model
#' D.training <- models("gaussian", type = "all", n.target = 100, K = 2, p = 500)
#' D.test <- models("gaussian", type = "target", n.target = 500, p = 500)
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
#' # fit a logistic regression model
#' D.training <- models("binomial", type = "all", n.target = 100, K = 2, p = 500)
#' D.test <- models("binomial", type = "target", n.target = 500, p = 500)
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
#' # fit a Poisson regression model
#' D.training <- models("poisson", type = "all", n.target = 100, K = 2, p = 500)
#' D.test <- models("poisson", type = "target", n.target = 500, p = 500)
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
                     transfer.source.id = "auto", alpha = 1, standardize = TRUE, intercept = TRUE,
                     nfolds = 10, cores = 1, valid.proportion = NULL, valid.nfolds = 3,
                     lambda = c(transfer = "lambda.1se", debias = "lambda.min", detection = "lambda.1se"),
                     lambda.seq = list(transfer = NULL, debias = NULL, detection = NULL),
                     detection.info = TRUE, target.weights = NULL, source.weights = NULL, C0 = 2, ...) {

  family <- match.arg(family)
  if (is.na(lambda["transfer"])) {
    lambda["transfer"] <- "lambda.1se"
  }
  if (is.na(lambda["debias"])) {
    lambda["debias"] <- "lambda.min"
  }
  if (is.na(lambda["detection"])) {
    lambda["detection"] <- "lambda.1se"
  }

  transfer.source.id.ori <- transfer.source.id
  data <- c(target, source) # to be updated
  k <- NULL
  lambda.fit <- c(transfer = NA, debias = NA)

  if (is.null(target.weights)) {
    target.weights <- rep(1, length(target$y))
  }

  if (is.null(source.weights)) {
    source.weights <- sapply(1:length(source), function(i){
      rep(1, length(source[[i]]$y))
    }, simplify = FALSE)
  }


  if (!is.null(source) && (is.string(transfer.source.id) && transfer.source.id == "all")) { # transfer all source data
    transfer.source.id <- 1:length(source)
  } else if (!is.null(source) && is.string(transfer.source.id) && transfer.source.id == "auto") { # automatically check which source data set to transfer
    A <- source_detection(target = target, family = family, source = source, alpha = alpha,
                          cores = cores,  lambda = lambda["detection"], lambda.seq = lambda.seq$detection, valid.proportion = valid.proportion,
                          valid.nfolds = valid.nfolds, detection.info = detection.info, standardize = standardize,
                          intercept = intercept, nfolds = nfolds, target.weights = target.weights, source.weights = source.weights, C0 = C0, ...)

    transfer.source.id <- A$transfer.source.id
    # lambda.fit["detection"] <- A$lambda
  } else if (0 %in% transfer.source.id || is.null(source)) { # don't transfer any source
    transfer.source.id <- 0
  }


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

  p <- ncol(all.x)


  if (cores > 1) {
    registerDoParallel(cores)
  }
  w <- Reduce("c", sapply(unique(c(0, transfer.source.id)), function(k){
    if (k != 0) {
      source.weights[[k]]
    } else {
      target.weights
    }
  }, simplify = FALSE))
  n.try <- 0
  while (T) {
    cv.fit.trans <- try(cv.glmnet(x = all.x, y = all.y, family = family, alpha = alpha, weights = w, nfolds = nfolds,
                                  parallel = I(cores > 1), intercept = intercept, standardize = standardize, lambda.min.ratio = 0.01,
                                  lambda = lambda.seq$transfer, ...), silent = T)
    if (!inherits(cv.fit.trans, "try-error")) {
      break
    }
    n.try <- n.try + 1
    print(paste("tried", n.try, "times during the transferring step!"))
    if (n.try > 10) {
      stop("errors during the transferring step!!!")
    }
  }

  if (lambda["transfer"] == "lambda.1se") {
    wa <- as.numeric(coef(cv.fit.trans))
    lambda.fit["transfer"] <- cv.fit.trans$lambda.1se
  } else if (lambda["transfer"] == "lambda.min") {
    wa <- c(cv.fit.trans$glmnet.fit$a0[which(cv.fit.trans$lambda == cv.fit.trans$lambda.min)], cv.fit.trans$glmnet.fit$beta[, which(cv.fit.trans$lambda == cv.fit.trans$lambda.min)])
    lambda.fit["transfer"] <- cv.fit.trans$lambda.min
  }



  # bias correcting step
  # --------------------------------------

  offset <- as.numeric(as.matrix(target$x) %*% wa[-1] + wa[1])


  n.try <- 0
  while (T) {
    cv.fit.correct <- try(cv.glmnet(x = as.matrix(target$x), y = target$y, weights = target.weights, offset = offset, family = family,
                                    parallel = I(cores > 1), intercept = intercept, lambda.min.ratio = 0.01, nfolds = nfolds,
                                    lambda = lambda.seq$debias, ...), silent=TRUE)
    if (!inherits(cv.fit.correct, "try-error")) {
      break
    }
    n.try <- n.try + 1
    # print(paste("tried", n.try, "times during the debiasing step!"))
    if (n.try > 10) {
      stop("Errors occur during the debiasing step!!!")
    }
  }
  if (lambda["debias"] == "lambda.1se") {
    deltaa <- as.numeric(coef(cv.fit.correct))
    lambda.fit["debias"] <- cv.fit.correct$lambda.1se
  } else {
    deltaa <- c(cv.fit.correct$glmnet.fit$a0[which(cv.fit.correct$lambda == cv.fit.correct$lambda.min)], cv.fit.correct$glmnet.fit$beta[, which(cv.fit.correct$lambda == cv.fit.correct$lambda.min)])
    lambda.fit["debias"] <- cv.fit.correct$lambda.min
  }


  beta.hat <- wa + deltaa
  if (!all(is.null(colnames(target$x)))) {
    names(beta.hat) <- c("intercept", colnames(target$x))
  } else {
    names(beta.hat) <- c("intercept", paste0("V", 1:ncol(target$x)))
  }

  if(cores > 1) {
    stopImplicitCluster()
  }

  if (!is.null(source) && is.string(transfer.source.id.ori) && transfer.source.id.ori == "auto") {
    obj <- list(beta = beta.hat, family = family, transfer.source.id = transfer.source.id,
                fitting.list = list(w_a = wa, delta_a = deltaa, target.valid.loss = A$target.valid.loss, source.loss = A$source.loss,
                                    threshold = A$threshold), lambda = lambda.fit)
  } else {
    if (0 %in% transfer.source.id) {
      transfer.source.id <- NULL
    }
    obj <- list(beta = beta.hat, family = family, transfer.source.id = transfer.source.id, fitting.list = list(w_a = wa, delta_a = deltaa),
                lambda = lambda.fit)
  }
  class(obj) <- "glmtrans"
  return(obj)
}
