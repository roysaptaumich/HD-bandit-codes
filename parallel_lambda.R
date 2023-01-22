#### Packages #####
library(MASS)
library(glmnet)
library(doSNOW)
library(foreach)
library(abind)
library(parallel)
library(plotly)
library(R6)
library(monomvn)
library(sparsevb)
library(ebreg)
library(functClust)
n_cores = max(detectCores()-5, 5)
cluster = makeCluster(n_cores, outfile="")
registerDoSNOW(cluster)
set.seed(2022)

source("bandit2.R")
save_file = 1 # 0 = do not save final result; 1 = save final result ; saving fromat: .csv file

### Simulation setup: Parameters ####
nrep = 10 # number of repetation
K = 10
p = 1000
s = 5
sigma = 1
Th = 400
rho2 = 0.3

type = "EC" # "EC" or "AR"

if(type == "EC"){
  V = (sigma^2 - rho2) * diag(1, p) + rho2 * matrix(1, nrow=p, ncol=p)
}else{
  V = rho2 ^ (abs(matrix(1:p - 1, nrow = p, ncol = p, byrow = TRUE) - (1:p - 1)))
}


beta_setup = 3 # (1) all signals 1; (2) all signals uniform (0.3,1) normalized; (3) all signals N(0,1) normalized

beta = numeric(p)
idx = sample(p, s)

if (beta_setup == 1){
  beta[idx] = 1
}else if(beta_setup == 2){
  beta[idx] = runif(s, min=0.3, max=1.0)
  beta = beta / base::norm(matrix(beta), type="F")
}else if(beta_setup == 3){
  beta[idx] = rnorm(s, 0, 1)
  beta = beta / base::norm(matrix(beta), type="F")
}

## ESTC params ##
zeta = 0.3
n1  = as.integer(Th^(2/3) * (s^2 * log(2*p))^(1/3) * zeta)

#### Parallel loops #####

regret_list  = foreach(i = 1:nrep, .packages = c("MASS", "sparsevb", "ebreg", "R6", "monomvn", "glmnet")) %dopar% {
  set.seed(i+1)
  .GlobalEnv$p <- p
  .GlobalEnv$K <- K
  .GlobalEnv$sigma <- sigma
  rndBandit = randomBandit$new(p=p, K=K)
  .GlobalEnv$Th = Th
  .GlobalEnv$n1 = n1
  drlasso = DR.Lasso$new(lambda1=1, lambda2=0.5, p=p, K=K, tc=1, tr=TRUE, T0=10)
  lasso.opt = Lasso.optimism$new(lambda= 0.5, tau=1, p=p, K=K, T0=2)
  estc = ESTC$new(lambda=0.5, p=p, K=K, T0=n1)
  vbts0 = VB.complexity.TS$new(p=p, K=K, u=1.001, T0=5, T1=5, lambda=1)
  vbts1 = VB.complexity.TS$new(p=p, K=K, u=1.001, T0=5, T1=5, lambda=0.2)
  vbts2 = VB.complexity.TS$new(p=p, K=K, u=1.001, T0=5, T1=5, lambda=0.3)
  vbts3 = VB.complexity.TS$new(p=p, K=K, u=1.001, T0=5, T1=5, lambda=0.4)
  vbts4 = VB.complexity.TS$new(p=p, K=K, u=1.001, T0=5, T1=5, lambda=0.5)
  
  # rewards
  drlasso.mean_rewards = numeric(Th)
  lasso_opt.mean_rewards = numeric(Th)
  estc.mean_rewards = numeric(Th)
  vbts0.mean_rewards = numeric(Th)
  vbts1.mean_rewards = numeric(Th)
  vbts2.mean_rewards = numeric(Th)
  vbts3.mean_rewards = numeric(Th)
  vbts4.mean_rewards = numeric(Th)
  opt.mean_rewards = numeric(Th)
  
  
  # times
  drlasso.time = 0
  lasso.opt.time = 0
  estc.time = 0
  vbts0.time = 0
  vbts1.time = 0
  vbts2.time = 0
  vbts3.time = 0
  vbts4.time = 0
  
  
  X_full = mvrnorm(n=K*Th, mu=numeric(p), Sigma=V)
  #X_full = X_full / apply(X_full, MARGIN=1, FUN=function(x) max(abs(x)))
  #cat(dim(X_full), '\n')
  
  for (t in 1:Th){
    # generic
    X = X_full[(K*(t-1)+1):(K*(t-1) + K), ]
    error = rnorm(1, mean=0, sd=sigma)
    opt.mean_rewards[t] = max(X %*% beta)
    
    # drlasso
    start = Sys.time()
    a1 = drlasso$choose.action(t, t(X))
    chosen.mean = X[a1,] %*% beta
    drlasso.mean_rewards[t] = chosen.mean
    drlasso$update.beta(chosen.mean + error, X[a1,], t)
    cat("nrep", i,"round",t, "DRlasso complete","\n")
    drlasso.time = drlasso.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # lasso_opt
    start = Sys.time()
    a2 = lasso.opt$choose.action(t, t(X))
    chosen.mean = X[a2,] %*% beta
    lasso_opt.mean_rewards[t] = chosen.mean
    lasso.opt$update.beta(chosen.mean + error, X[a2,], t)
    cat("nrep", i,"round",t, "Lasso_opt complete","\n")
    lasso.opt.time = lasso.opt.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # ESTC
    start = Sys.time()
    a3 = estc$choose.action(t, t(X))
    chosen.mean = X[a3,] %*% beta
    estc.mean_rewards[t] = chosen.mean
    estc$update.beta(chosen.mean + error, X[a3,], t)
    cat("nrep", i,"round",t, "ESTC complete","\n")
    estc.time = estc.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # VB_TS fixed 
    start = Sys.time()
    a4 = vbts0$choose.action(t, t(X))
    chosen.mean = X[a4,] %*% beta
    vbts0.mean_rewards[t] = chosen.mean
    vbts0$update.beta(chosen.mean + error, X[a4,], t, fixed=TRUE)
    cat("nrep", i,"round",t, "VB TS fixed complete","\n")
    vbts0.time = vbts0.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # VB_TS 0.2 
    start = Sys.time()
    a5 = vbts1$choose.action(t, t(X))
    chosen.mean = X[a5,] %*% beta
    vbts1.mean_rewards[t] = chosen.mean
    vbts1$update.beta(chosen.mean + error, X[a5,], t, fixed=FALSE)
    cat("nrep", i,"round",t, "VB TS 0.2 complete","\n")
    vbts1.time = vbts1.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # VB_TS 0.3 
    start = Sys.time()
    a6 = vbts2$choose.action(t, t(X))
    chosen.mean = X[a6,] %*% beta
    vbts2.mean_rewards[t] = chosen.mean
    vbts2$update.beta(chosen.mean + error, X[a6,], t, fixed=FALSE)
    cat("nrep", i,"round",t, "VB TS 0.3 complete","\n")
    vbts2.time = vbts2.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # VB_TS 0.4 
    start = Sys.time()
    a7 = vbts3$choose.action(t, t(X))
    chosen.mean = X[a7,] %*% beta
    vbts3.mean_rewards[t] = chosen.mean
    vbts3$update.beta(chosen.mean + error, X[a7,], t, fixed=FALSE)
    cat("nrep", i,"round",t, "VB TS 0.4 complete","\n")
    vbts3.time = vbts3.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # VB_TS 0.5 
    start = Sys.time()
    a8 = vbts4$choose.action(t, t(X))
    chosen.mean = X[a8,] %*% beta
    vbts4.mean_rewards[t] = chosen.mean
    vbts4$update.beta(chosen.mean + error, X[a8,], t, fixed=FALSE)
    cat("nrep", i,"round",t, "VB TS 0.5 complete","\n")
    vbts4.time = vbts4.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
  }
  
  regret.drlasso = cumsum(opt.mean_rewards) - cumsum(drlasso.mean_rewards)
  regret.lasso_opt = cumsum(opt.mean_rewards) - cumsum(lasso_opt.mean_rewards)
  regret.estc = cumsum(opt.mean_rewards) - cumsum(estc.mean_rewards)
  regret.vbts0 = cumsum(opt.mean_rewards) - cumsum(vbts0.mean_rewards)
  regret.vbts1 = cumsum(opt.mean_rewards) - cumsum(vbts1.mean_rewards)
  regret.vbts2 = cumsum(opt.mean_rewards) - cumsum(vbts2.mean_rewards)
  regret.vbts3 = cumsum(opt.mean_rewards) - cumsum(vbts3.mean_rewards)
  regret.vbts4 = cumsum(opt.mean_rewards) - cumsum(vbts4.mean_rewards)
  
  return(list(regret.drlasso = regret.drlasso, drlasso.time = drlasso.time,
              regret.lasso_opt = regret.lasso_opt, lasso_opt.time = lasso.opt.time,
              regret.estc = regret.estc, estc.time = estc.time,
              regret.vbts0 = regret.vbts0, vbts0.time = vbts0.time,
              regret.vbts1 = regret.vbts1, vbts1.time = vbts1.time,
              regret.vbts2 = regret.vbts2, vbts2.time = vbts2.time,
              regret.vbts3 = regret.vbts3, vbts3.time = vbts3.time,
              regret.vbts4 = regret.vbts4, vbts4.time = vbts4.time))
}
stopCluster(cluster)
cat('clusters stopped')
##### Building dataframe ######

methods = c("DRLasso", "Lasso-L1", "ESTC", "VBTS_fixed", "VBTS_0.2", "VBTS_0.3", "VBTS_0.4", "VBTS_0.5")
m = length(methods)
regret_list2 = lapply(regret_list, function(x){x[2*(1:m) - 1]})
time_list = lapply(regret_list, function(x){x[2*(1:m)]})
df = data.frame(Rounds = 1:Th)

w = 1.96/sqrt(nrep)

for (method  in methods) {
  id = which(methods == method)
  l1 = apply(do.call(rbind, lapply(regret_list2, function(x){x[[id]]})), 2, mean)
  assign(paste("meanregret.", method, sep = ""), l1)
  
  l2 = l1 + w * apply(do.call(rbind, lapply(regret_list2, function(x){x[[id]]})), 2, sd)
  #assign(paste("sdregret.", method, sep = ""), l2)
  
  l3 = l1 - w* apply(do.call(rbind, lapply(regret_list2, function(x){x[[id]]})), 2, sd)
  assign(paste("meantime.", method, sep = ""), l3)
  
  l4 = apply(do.call(rbind, lapply(time_list, function(x){x[[id]]})), 2, mean)
  #assign(paste("meantime.", method, sep = ""), l4)
  
  df_mid = data.frame(l1=l1, l2=l2, l3=l3, l4=l4)
  colnames(df_mid) = c(paste("meanregret.", method, sep = ""), paste("meanregret_high.", method, sep = ""),
                       paste("meanregret_low.", method, sep = ""), paste("meantime.", method, sep = ""))
  df = cbind(df, df_mid)
}



plot(1:Th, df$'meanregret.VBTS_fixed', col='red', type='l', ylim=c(0,600), xlab='time', ylab='regret')
lines(1:Th, df$'meanregret.VBTS_0.2', col='blue')
lines(1:Th, df$'meanregret.VBTS_0.3', col='green')
lines(1:Th, df$'meanregret.VBTS_0.4', col='orange')
lines(1:Th, df$'meanregret.VBTS_0.5', col='brown')
lines(1:Th, df$'meanregret.ESTC', col='black', lty=3)
lines(1:Th, df$'meanregret.Lasso-L1', col='pink', lty=4)
lines(1:Th, df$'meanregret.DRLasso', col='black', lty=2)


