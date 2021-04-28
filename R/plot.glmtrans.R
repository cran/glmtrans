#' Visualize the losses of different sources and the threshold to determine transferability.
#'
#' Plot the losses of different sources and the threshold to determine transferability for object with class "glmtrans" or "glmtrans_source_detection".
#' @export
#' @aliases plot.glmtrans_source_detection
#' @param x an object from class "glmtrans" or "glmtrans_source_detection", which are the output of functions \code{glmtrans} and \code{source_detection}, respectively.
#' @param ... additional arguments that can be passed to \code{ggplot} function.
#' @return a "ggplot" visualization with the transferable threshold and losses of different sources.
#' @seealso \code{\link{glmtrans}}, \code{\link{source_detection}}, \code{\link[ggplot2]{ggplot}}.
#' @references
#' Tian, Y. and Feng, Y., 2021. \emph{Transfer learning with high-dimensional generalized linear models. Submitted.}
#'
#' @examples
#' set.seed(1, kind = "L'Ecuyer-CMRG")
#'
#' D.training <- models("gaussian", K = 2, p = 500, Ka = 1)
#'
#' # plot for class "glmtrans"
#' fit.gaussian <- glmtrans(D.training$target, D.training$source)
#' plot(fit.gaussian)
#'
#' \donttest{
#' # plot for class "glmtrans_source_detection"
#' detection.gaussian <- source_detection(D.training$target, D.training$source)
#' plot(detection.gaussian)
#' }
#'
plot.glmtrans <- function(x, ...) {
  y <- NULL
  source.id <- NULL
  transferable <- sapply(1:length(x$fitting.list$source.loss), function(i){
    ifelse(x$fitting.list$source.loss[i] <= x$fitting.list$threshold, "Y", "N")
  })
  values <- c(x$fitting.list$source.loss, x$fitting.list$threshold)
  rg <- max(values) - min(values)
  loss.matrix <- data.frame(source.id = factor(1:length(x$fitting.list$source.loss)), loss = x$fitting.list$source.loss, transferable = transferable)
  threshold <- data.frame(x = c(-Inf, Inf), y = x$fitting.list$threshold, threshold = factor(""))
  ggplot(loss.matrix, mapping = aes(x = source.id, y = loss), ...) + geom_point(aes(color = transferable)) + geom_line(aes(x = x, y = y, linetype=threshold), threshold) +
    ylim(min(values) - 0.1*rg, max(values) +  0.1*rg)

}
