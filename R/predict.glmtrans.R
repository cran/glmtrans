#' Predict for new data from a "glmtrans" object.
#'
#' Predict from a "glmtrans" object based on new observation data. There are various types of output available.
#' @export
#' @param object an object from class "glmtrans", which comes from the output of function \code{glmtrans}.
#' @param newx the matrix of new values for predictors at which predictions are to be made. Should be in accordance with the data for training \code{object}.
#' @param type the type of prediction. Default = "link".
#' @param ... additional arguments.
#' \itemize{
#' \item "link": the linear predictors. When \code{family = "gaussian"}, it is the same as the predicited responses.
#' \item "response": gives the predicited probabilities when \code{family = "binomial"}, the predicited mean when \code{family = "poisson"}, and the predicited responses when \code{family = "gaussian"}.
#' \item "class": the predicited 0/1 responses for lositic distribution. Applies only when \code{family = "binomial"}.
#' \item "integral response": the predicited integral response for Poisson distribution. Applies only when \code{family = "poisson"}.
#' }
#' @return the predicted result on new data, which depends on \code{type}.
#' @seealso \code{\link{glmtrans}}.
#' @references
#' Tian, Y., & Feng, Y. (2023). \emph{Transfer learning under high-dimensional generalized linear models. Journal of the American Statistical Association, 118(544), 2684-2697.}
#'
#' @examples
#' set.seed(1, kind = "L'Ecuyer-CMRG")
#'
#' # fit a logistic model
#' D.training <- models("binomial", type = "all", K = 1, p = 500)
#' D.test <- models("binomial", type = "target", n.target = 10, p = 500)
#' fit.binomial <- glmtrans(D.training$target, D.training$source, family = "binomial")
#'
#' predict(fit.binomial, D.test$target$x, type = "link")
#' predict(fit.binomial, D.test$target$x, type = "response")
#' predict(fit.binomial, D.test$target$x, type = "class")
#'
#' \donttest{
#' # fit a Poisson model
#' D.training <- models("poisson", type = "all", K = 1, p = 500)
#' D.test <- models("poisson", type = "target", n.target = 10, p = 500)
#' fit.poisson <- glmtrans(D.training$target, D.training$source, family = "poisson")
#'
#' predict(fit.poisson, D.test$target$x, type = "response")
#' predict(fit.poisson, D.test$target$x, type = "integral response")
#' }
predict.glmtrans <- function(object, newx, type = c("link", "response", "class", "integral response"), ...) {
  type <- match.arg(type)
  newx <- as.matrix(newx)
  if (object$family == "gaussian") {
    return(as.numeric(newx %*% object$beta[-1] + object$beta[1]))
  } else if (object$family == "binomial") {
    if (type == "link") {
      return(as.numeric(newx %*% object$beta[-1] + object$beta[1]))
    } else if (type == "class") {
      cl <- as.numeric(newx %*% object$beta[-1] + object$beta[1] > 0)
      return(cl)
    } else if (type == "response") {
      return(as.numeric(1/(1+exp(-newx %*% object$beta[-1] - object$beta[1]))))
    }
  } else if (object$family == "poisson") {
    if (type == "link") {
      return(as.numeric(newx %*% object$beta[-1] + object$beta[1]))
    } else if (type == "response") {
      return(as.numeric(exp(newx %*% object$beta[-1] + object$beta[1])))
    } else if (type == "integral response"){
      mean.pred <- exp(newx %*% object$beta[-1] + object$beta[1])
      y.pred <- numeric(nrow(newx))
      prob.floor <- dpois(floor(mean.pred), lambda = mean.pred)
      prob.ceiling <- dpois(ceiling(mean.pred), lambda = mean.pred)
      y.pred[prob.floor <= prob.ceiling] <- ceiling(mean.pred)
      y.pred[prob.floor > prob.ceiling] <- floor(mean.pred)
      return(y.pred)
    }
  }
}
