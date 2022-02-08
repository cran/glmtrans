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


createFolds_binary <- function(y, k) { # y should be binary
  ind0 <- which(y == 0)
  ind1 <- which(y == 1)
  fold0 <- createFolds(ind0, k = k)
  fold1 <- createFolds(ind1, k = k)
  sapply(1:k, function(i){
    c(ind0[fold0[[i]]], ind1[fold1[[i]]])
  }, simplify = F)
}


desparsified.lasso <- function(x = x, y = y, family = c("gaussian", "binomial", "poisson"), cores = 1, beta.hat = NULL, lambda = c("lambda.1se", "lambda.min"), level = 0.95, intercept = TRUE, ...) {
  family <- match.arg(family)
  lambda <- match.arg(lambda)
  registerDoParallel(cores)
  if (!is.null(colnames(beta.hat))) {
    colnames(beta.hat) <- NULL
  }
  r.level <- level + (1-level)/2
  j <- 0

  if (family == "gaussian") {

    if (intercept) {
      X.centralized <- cbind(1, x)
    } else {
      X.centralized <- x
    }

    Sigma.hat <- (t(X.centralized) %*% X.centralized)/nrow(x)
    p <- ncol(X.centralized)

    L <- foreach(j = 1:p, .combine = "rbind") %dopar% {
      node.lasso <- cv.glmnet(x = X.centralized[, -j], y = X.centralized[, j], family = "gaussian", intercept = FALSE, ...)
      if (lambda == "lambda.1se") {
        gamma <- coef(node.lasso)[-1]
      } else if (lambda == "lambda.min") {
        min.ind <- which(node.lasso$lambda.min == node.lasso$lambda)
        gamma <- node.lasso$glmnet.fit$beta[, min.ind]
      }
      tau2 <- Sigma.hat[j, j] - Sigma.hat[j, -j, drop = F] %*% gamma
      theta <- rep(1, p)
      theta[-j] <- -gamma
      c(theta, tau2)
    }

    Theta.hat <- solve(diag(L[,p+1])) %*% L[1:p, 1:p]

    Z <- y - x %*% beta.hat[-1] - beta.hat[1]
    if (intercept) {
      b.hat <- as.matrix(beta.hat) + Theta.hat %*% t(X.centralized) %*% Z/nrow(x)
    } else {
      b.hat <- beta.hat[-1] + Theta.hat %*% t(x) %*% Z/nrow(x)
    }

    var.est <- diag(Theta.hat %*% Sigma.hat %*% t(Theta.hat))
    CI <- data.frame(b.hat = b.hat, lb = b.hat - qnorm(r.level)*sqrt(var.est/nrow(x)), ub =  b.hat + qnorm(r.level)*sqrt(var.est/nrow(x)))

  } else if (family == "binomial") {
    u <- x %*% beta.hat[-1] + beta.hat[1]
    w <- as.vector(sqrt(exp(-u)/((1+exp(-u))^2)))
    Xw <- diag(w) %*% x

    if (intercept) {
      X.centralized <- cbind(w, Xw)
    } else {
      X.centralized <- Xw
    }

    p <- ncol(X.centralized)
    Sigma.hat <- (t(X.centralized) %*% X.centralized)/nrow(x)

    L <- foreach(j = 1:p, .combine = "rbind") %dopar% {
      if (!all(X.centralized[, j] == 0)) {
        node.lasso <- cv.glmnet(x = X.centralized[, -j], y = X.centralized[, j], family = "gaussian", intercept = FALSE, ...)
      } else {
        node.lasso <- cv.glmnet(x = X.centralized[, -j], y = X.centralized[, j] + 1e-10, family = "gaussian", intercept = FALSE, ...)
      }
      if (lambda == "lambda.1se") {
        gamma <- coef(node.lasso)[-1]
      } else if (lambda == "lambda.min") {
        min.ind <- which(node.lasso$lambda.min == node.lasso$lambda)
        gamma <- node.lasso$glmnet.fit$beta[, min.ind]
      }
      tau2 <- Sigma.hat[j, j] - Sigma.hat[j, -j, drop = F] %*% gamma
      theta <- rep(1, p)
      theta[-j] <- -gamma
      c(theta, tau2)
    }
    # tau2 <- L[,p+1]
    # tau2.inv <- 1/tau2
    # tau2.inv[abs(tau2) <= 1e-20] <- 0
    # Theta.hat <- diag(tau2.inv) %*% L[1:p, 1:p]
    Theta.hat <- solve(diag(L[,p+1])) %*% L[1:p, 1:p]
    Z <- y - 1/(1+exp(-u))

    if (intercept) {
      b.hat <- as.matrix(beta.hat) + Theta.hat %*% t(cbind(1, x)) %*% Z/nrow(x)
    } else {
      b.hat <- beta.hat[-1] + Theta.hat %*% t(x) %*% Z/nrow(x)
    }

    var.est <- diag(Theta.hat %*% Sigma.hat %*% t(Theta.hat))
    CI <- data.frame(b.hat = b.hat, lb = b.hat - qnorm(r.level)*sqrt(var.est/nrow(x)), ub = b.hat + qnorm(r.level)*sqrt(var.est/nrow(x)))

  } else if (family == "poisson") {
    u <- x %*% beta.hat[-1] + beta.hat[1]
    w <- as.vector(exp(u/2))
    Xw <- diag(w) %*% x

    if (intercept) {
      X.centralized <- cbind(w, Xw)
    } else {
      X.centralized <- Xw
    }

    p <- ncol(X.centralized)
    Sigma.hat <- (t(X.centralized) %*% X.centralized)/nrow(x)

    L <- foreach(j = 1:p, .combine = "rbind") %dopar% {
      node.lasso <- cv.glmnet(x = X.centralized[, -j], y = X.centralized[, j], family = "gaussian", intercept = FALSE, ...)
      if (lambda == "lambda.1se") {
        gamma <- coef(node.lasso)[-1]
      } else if (lambda == "lambda.min") {
        min.ind <- which(node.lasso$lambda.min == node.lasso$lambda)
        gamma <- node.lasso$glmnet.fit$beta[, min.ind]
      }
      tau2 <- Sigma.hat[j, j] - Sigma.hat[j, -j, drop = F] %*% gamma
      theta <- rep(1, p)
      theta[-j] <- -gamma
      c(theta, tau2)
    }
    Theta.hat <- solve(diag(L[,p+1])) %*% L[1:p, 1:p]
    Z <- y - exp(u)

    if (intercept) {
      b.hat <- as.matrix(beta.hat) + Theta.hat %*% t(cbind(1, x)) %*% Z/nrow(x)
    } else {
      b.hat <- beta.hat[-1] + Theta.hat %*% t(x) %*% Z/nrow(x)
    }

    var.est <- diag(Theta.hat %*% Sigma.hat %*% t(Theta.hat))
    CI <- data.frame(b.hat = b.hat, lb = b.hat - qnorm(r.level)*sqrt(var.est/nrow(x)), ub = b.hat + qnorm(r.level)*sqrt(var.est/nrow(x)))
  }

  stopImplicitCluster()
  return(list(b.hat = b.hat, beta.hat = beta.hat, CI = CI, var.est = var.est))

}










