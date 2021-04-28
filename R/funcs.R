#' @export
plot.glmtrans_source_detection <- function(x, ...) {
    y <- NULL
    source.id <- NULL
    transferable <- sapply(1:length(x$source.loss), function(i){
      ifelse(x$source.loss[i] <= x$threshold, "Y", "N")
    })
    values <- c(x$source.loss, x$threshold)
    rg <- max(values) - min(values)
    loss.matrix <- data.frame(source.id = factor(1:length(x$source.loss)), loss = x$source.loss, transferable = transferable)
    threshold <- data.frame(x = c(-Inf, Inf), y = x$threshold, threshold = factor(""))
    ggplot(loss.matrix, mapping = aes(x = source.id, y = loss), ...) + geom_point(aes(color = transferable)) + geom_line(aes(x = x, y = y, linetype=threshold), threshold) +
      ylim(min(values) - 0.1*rg, max(values) +  0.1*rg)
}


loss <- function(wa, x.valid, y.valid, family) {
  if (family == "gaussian") {
    mean((y.valid - x.valid %*% wa[-1] - wa[1])^2)
  } else if (family == "binomial"){
    xb <- x.valid %*% wa[-1] + wa[1]
    as.numeric(- t(y.valid) %*% xb + sum(log(1+exp(xb))))/length(y.valid)
  } else if (family == "poisson") {
    xb <- x.valid %*% wa[-1] + wa[1]
    as.numeric(- t(y.valid) %*% xb + sum(exp(xb)) + sum(lgamma(y.valid+1)))/length(y.valid)
  }
}
