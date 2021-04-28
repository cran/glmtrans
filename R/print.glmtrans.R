#' Print a fitted "glmtrans" object.
#'
#' Similar to the usual print methods, this function summarizes results from a fitted \code{"glmtrans"} object.
#' @export
#' @param x fitted \code{"glmtrans"} model object.
#' @param ... additional arguments.
#' @return No value is returned.
#' @seealso \code{\link{glmtrans}}.
#' @examples
#' set.seed(1, kind = "L'Ecuyer-CMRG")
#'
#' # fit a linear model
#' D.training <- models("gaussian", K = 2, p = 500)
#' fit.gaussian <- glmtrans(D.training$target, D.training$source)
#'
#' fit.gaussian
#'

print.glmtrans <- function(x, ...) {
  hidden <- c("fitting.list", "beta")
  print(x[!names(x) %in% hidden])
}
