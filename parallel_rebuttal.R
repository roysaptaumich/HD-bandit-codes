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
n_cores = max(detectCores()-5, 7)
cluster = makeCluster(n_cores, outfile="")
registerDoSNOW(cluster)
set.seed(2022)
#setwd("/Users/sunrit/Library/CloudStorage/GoogleDrive-sunritc@umich.edu/Shared drives/Bandit/Sunrit_Vola/HD_Bandits/Code/R")
#setwd("/Users/roysapta/Library/CloudStorage/GoogleDrive-roysapta@umich.edu/Shared drives/Bandit/Sunrit_Vola/HD_Bandits/Code/R")
source("bandit2.R")
save_file = 1 # 0 = do not save final result; 1 = save final result ; saving fromat: .csv file

### Simulation setup: Parameters ####
nrep = 40 # number of repetation
K = 10
p = 1000
s = 5
sigma = 1
Th = 400
rho2 = 0.3

beta_setup = 2 # (1) all signals 1; (2) all signals uniform (0.3,1) normalized; (3) IDS
X_setup = 2 # (1) equicorrelated with rho; (2) auto-regressive with rho 

if (X_setup == 1){
  V = (sigma^2 - rho2) * diag(1, p) + rho2 * matrix(1, nrow=p, ncol=p)
}else{
  V = rho2 ^ (abs(matrix(1:p - 1, nrow = p, ncol = p, byrow = TRUE) - (1:p - 1)))
}

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
  linucb = LinUCB$new(lambda=2, T0=1, p=p, K=K, error.var=sigma^2)
  drlasso = DR.Lasso$new(lambda1=1, lambda2=0.5, p=p, K=K, tc=1, tr=TRUE, T0=10)
  lasso.opt = Lasso.optimism$new(lambda= 0.5, tau=1, p=p, K=K, T0=2)
  estc = ESTC$new(lambda=0.5, p=p, K=K, T0=n1)
  lints = LinTS$new(delta=0.01, epsilon=1/log(Th), p=p, K=K, error.var=sigma^2)
  vbts = VB.complexity.TS$new(p=p, K=K, u=1.001, T0=5, T1=5, lambda =1)
  ebregts = EBREG.TS$new(p=p, K=K, T0= 20, T1=1)
  blassots = BLasso.TS$new(p=p, K=K, T0 = 10, T1 = 50)
  sa.lasso = SA.Lasso$new(p=p, K=K, lambda0=2*sigma*sqrt(log(p)), alpha=1)
  sa.ridge = SA.Lasso$new(p=p, K=K, lambda0=2*sigma*sqrt(log(p)), alpha=0)
  th.lasso = Threshold.Lasso$new(p=p, K=K, lambda0=1/log(log(p))^0.25)
  
  # rewards
  rndbandit.mean_rewards = numeric(Th)
  linucb.mean_rewards = numeric(Th)
  drlasso.mean_rewards = numeric(Th)
  lasso_opt.mean_rewards = numeric(Th)
  estc.mean_rewards = numeric(Th)
  lints.mean_rewards = numeric(Th)
  vbts.mean_rewards = numeric(Th)
  ebregts.mean_rewards = numeric(Th)
  blassots.mean_rewards = numeric(Th)
  sa_lasso.mean_rewards = numeric(Th)
  sa_ridge.mean_rewards = numeric(Th)
  th_lasso.mean_rewards = numeric(Th)
  opt.mean_rewards = numeric(Th)
  
  
  
  # times
  rndbandit.time = 0
  linucb.time = 0
  drlasso.time = 0
  lasso.opt.time = 0
  estc.time = 0
  lints.time = 0
  vbts.time = 0
  blassots.time = 0
  ebregts.time = 0
  sa_lasso.time = 0
  sa_ridge.time = 0
  th_lasso.time = 0
  
  X_full = mvrnorm(n=K*Th, mu=numeric(p), Sigma=V)
  #X_full = X_full / apply(X_full, MARGIN=1, FUN=function(x) max(abs(x)))
  #cat(dim(X_full), '\n')
  lambda_vbts = 1 #0.3 * sqrt(1:Th)
  
  for (t in 1:Th){
    # generic
    X = X_full[(K*(t-1)+1):(K*(t-1) + K), ]
    #cat(dim(X), '\n')
    #X = mvrnorm(n=K, mu=numeric(p), Sigma=V)
    error = rnorm(1, mean=0, sd=sigma)
    opt.mean_rewards[t] = max(X %*% beta)
    
    # random
    start = Sys.time()
    a0 = rndBandit$choose.action(t, t(X))
    chosen.mean = 0 #X[a0,] %*% beta
    rndbandit.mean_rewards[t] = chosen.mean
    cat("nrep", i,"round",t, "random complete","\n")
    rndbandit.time = rndbandit.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # linucb
    start = Sys.time()
    #tic()
    a1 = 1 #linucb$choose.action(t, t(X))
    chosen.mean = 0 #X[a1,] %*% beta
    linucb.mean_rewards[t] = chosen.mean
    linucb$update.beta(chosen.mean + error, X[a1,], t)
    cat("nrep", i,"round",t, "Linucb complete","\n")
    #linucb.time = linucb.time + toc()
    linucb.time = linucb.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # drlasso
    start = Sys.time()
    a2 = drlasso$choose.action(t, t(X))
    chosen.mean = 0 #X[a2,] %*% beta
    drlasso.mean_rewards[t] = chosen.mean
    drlasso$update.beta(chosen.mean + error, X[a2,], t)
    cat("nrep", i,"round",t, "DRlasso complete","\n")
    drlasso.time = drlasso.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # lasso_opt
    start = Sys.time()
    a3 = lasso.opt$choose.action(t, t(X))
    chosen.mean = 0#X[a3,] %*% beta
    lasso_opt.mean_rewards[t] = chosen.mean
    lasso.opt$update.beta(chosen.mean + error, X[a3,], t)
    cat("nrep", i,"round",t, "Lasso_opt complete","\n")
    lasso.opt.time = lasso.opt.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # ESTC
    start = Sys.time()
    a4 = estc$choose.action(t, t(X))
    chosen.mean = 0#X[a4,] %*% beta
    estc.mean_rewards[t] = chosen.mean
    estc$update.beta(chosen.mean + error, X[a4,], t)
    cat("nrep", i,"round",t, "ESTC complete","\n")
    estc.time = estc.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # LinTS
    start = Sys.time()
    a5 = lints$choose.action(t, t(X))
    chosen.mean = 0 #X[a5,] %*% beta
    lints.mean_rewards[t] = chosen.mean
    lints$update.beta(chosen.mean + error, X[a5,], t)
    cat("nrep", i,"round",t, "LinTS complete","\n")
    lints.time = lints.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # VB_TS
    start = Sys.time()
    #vbts$lambda = lambda_vbts[t]
    a6 = vbts$choose.action(t, t(X))
    chosen.mean = 0 #X[a6,] %*% beta
    vbts.mean_rewards[t] = chosen.mean
    vbts$update.beta(chosen.mean + error, X[a6,], t, fixed=FALSE)
    cat("nrep", i,"round",t, "VB TS complete","\n")
    vbts.time = vbts.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # EBREG_TS
    start = Sys.time()
    a7 = ebregts$choose.action(t, t(X))
    chosen.mean = 0 #X[a7,] %*% beta
    ebregts.mean_rewards[t] = chosen.mean
    ebregts$update.beta(chosen.mean + error, X[a7,], t)
    cat("nrep", i,"round",t, "EBREG TS complete","\n")
    ebregts.time = ebregts.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # BLASSO_TS
    start = Sys.time()
    a8 = blassots$choose.action(t, t(X))
    chosen.mean = 0 #X[a8,] %*% beta
    blassots.mean_rewards[t] = chosen.mean
    blassots$update.beta(chosen.mean + error, X[a8,], t)
    cat("nrep", i,"round",t, "BLASSO TS complete","\n")
    blassots.time = blassots.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # SA_Lasso
    start = Sys.time()
    a9 = sa.lasso$choose.action(t, t(X))
    chosen.mean = X[a9,] %*% beta
    sa_lasso.mean_rewards[t] = chosen.mean
    sa.lasso$update.beta(chosen.mean + error, X[a9,], t)
    cat("nrep", i,"round",t, "SA Lasso complete","\n")
    sa_lasso.time = sa_lasso.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # SA_Ridge
    start = Sys.time()
    a10 = sa.ridge$choose.action(t, t(X))
    chosen.mean = X[a10,] %*% beta
    sa_ridge.mean_rewards[t] = chosen.mean
    sa.ridge$update.beta(chosen.mean + error, X[a10,], t)
    cat("nrep", i,"round",t, "SA Ridge complete","\n")
    sa_ridge.time = sa_ridge.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # Threshold_Lasso
    start = Sys.time()
    a11 = th.lasso$choose.action(t, t(X))
    chosen.mean = X[a11,] %*% beta
    th_lasso.mean_rewards[t] = chosen.mean
    th.lasso$update.beta(chosen.mean + error, X[a11,], t)
    cat("nrep", i,"round",t, "Th Lasso complete","\n")
    th_lasso.time = th_lasso.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
  }
  
  regret.random = cumsum(opt.mean_rewards) - cumsum(rndbandit.mean_rewards)
  regret.linucb = cumsum(opt.mean_rewards) - cumsum(linucb.mean_rewards)
  regret.drlasso = cumsum(opt.mean_rewards) - cumsum(drlasso.mean_rewards)
  regret.lasso_opt = cumsum(opt.mean_rewards) - cumsum(lasso_opt.mean_rewards)
  regret.estc = cumsum(opt.mean_rewards) - cumsum(estc.mean_rewards)
  regret.lints = cumsum(opt.mean_rewards) - cumsum(lints.mean_rewards)
  regret.vbts = cumsum(opt.mean_rewards) - cumsum(vbts.mean_rewards)
  regret.ebregts = cumsum(opt.mean_rewards) - cumsum(ebregts.mean_rewards)
  regret.blassots = cumsum(opt.mean_rewards) - cumsum(blassots.mean_rewards)
  regret.sa_lasso = cumsum(opt.mean_rewards) - cumsum(sa_lasso.mean_rewards)
  regret.sa_ridge = cumsum(opt.mean_rewards) - cumsum(sa_ridge.mean_rewards)
  regret.th_lasso = cumsum(opt.mean_rewards) - cumsum(th_lasso.mean_rewards)
  
  return(list(regret.random = regret.random, random.time = rndbandit.time,
              regret.linucb = regret.linucb, linucb.time = linucb.time,
              regret.drlasso = regret.drlasso, drlasso.time = drlasso.time,
              regret.lasso_opt = regret.lasso_opt, lasso_opt.time = lasso.opt.time,
              regret.estc = regret.estc, estc.time = estc.time,
              regret.lints = regret.lints, lints.time = lints.time,
              regret.vbts = regret.vbts, vbts.time = vbts.time,
              regret.ebregts = regret.ebregts, ebregts.time = ebregts.time,
              regret.blassots = regret.blassots, blassots.time = blassots.time,
              regret.sa_lasso = regret.sa_lasso, sa_lasso.time = sa_lasso.time,
              regret.sa_ridge = regret.sa_ridge, sa_ridge.time = sa_ridge.time,
              regret.th_lasso = regret.th_lasso, th_lasso.time = th_lasso.time))
}
stopCluster(cluster)
cat('clusters stopped')
##### Building dataframe ######

methods = c("Random", "LinUCB", "DRLasso", "Lasso-L1", "ESTC", "LinTS", "VBTS", "EBRegTS", "BLassoTS", "SA-Lasso", "SA-Ridge", "TH-Lasso")
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

if (save_file == 1){
  if (X_setup == 1){x_name="corr"}else{x_name="AR"}
  filename = paste("csv/Rebuttal/rebuttal_regret", "d", p, "s", s, "K", K, "beta_setup", beta_setup, x_name, rho2,".csv", sep = "_")
  write.csv(df,file =  filename, row.names = F)
}

plot(1:Th, df$'meanregret.DRLasso', col='white', type='l', ylim=c(0,700), xlab='time', ylab='regret')
#lines(1:Th, df$'meanregret.VBTS', col='black')
#lines(1:Th, df$'meanregret.ESTC', col='green')
#lines(1:Th, df$'meanregret.Lasso-L1', col='blue')
lines(1:Th, df$'meanregret.SA-Lasso', col='orange')
lines(1:Th, df$'meanregret.SA-Ridge', col='cyan')
lines(1:Th, df$'meanregret.TH-Lasso', col='red')



