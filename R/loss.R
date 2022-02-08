loss <- function(wa, x.valid, y.valid, family, tau = NULL) {
  if (family == "gaussian") {
    mean((y.valid - x.valid %*% wa[-1] - wa[1])^2)
  } else if (family == "binomial"){
    xb <- x.valid %*% wa[-1] + wa[1]
    as.numeric(- t(y.valid) %*% xb + sum(log(1+exp(xb))))/length(y.valid)
  } else if (family == "poisson") {
    xb <- x.valid %*% wa[-1] + wa[1]
    as.numeric(- t(y.valid) %*% xb + sum(exp(xb)) + sum(lgamma(y.valid+1)))/length(y.valid)
  } else { # family == "huber
    xb <- x.valid %*% wa[-1] + wa[1]
    res <- y.valid - x.valid %*% wa[-1] - wa[1]
    mean((res^2)/2*I(abs(res)<=tau) + (tau*abs(res)-(tau^2)/2)*I(abs(res)>tau))
  }
}
