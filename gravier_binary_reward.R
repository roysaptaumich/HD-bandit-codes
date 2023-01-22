#library(devtools)
#install_github("ramhiser/datamicroarray")
library(datamicroarray)
describe_data()
data("gravier", package = "datamicroarray")
library(MASS)
library(doSNOW)
library(foreach)
library(abind)
library(parallelly)
#library(simsl)
library(sparsevb)
library(kernlab)
n_cores = max(availableCores()-5, 4)
cluster = makeCluster(n_cores, outfile="")
registerDoSNOW(cluster)
set.seed(2022)

source("/home/roysapta/Bayes_SLCB/R/bandit.R")
setwd("/home/roysapta/Bayes_SLCB/R")
#save_file = 0 # 0 = do not save final result; 1 = save final result ; saving fromat: .csv file


X = as.matrix(gravier$x)
X = scale(X, center = colMeans(X), scale = apply(X, 2, sd))
y = gravier$y; y = as.factor(1*(y == "poor"))

mod = cv.glmnet(x = X, y = y, family = "binomial", intercept = F, nlambda = 100)
lambda = mod$lambda.min
beta = as.numeric(coef(mod))[-1]; #beta = beta
p0 = ncol(X)
supp = which(beta!=0)
spurious_id = sample(which(beta==0), p0 - length(supp)) 
ID = c(supp, spurious_id); ID = ID[sample(1:length(ID), length(ID))]


acc_train = mean(predict(mod, X, type = "class", s = "lambda.min") == y)
acc_train;  sum(coef(mod)!=0)


df = data.frame(X[, ID]); df$type = as.factor(1*(y==1))

D = as.matrix(df); D = D[sample(nrow(D), nrow(D)),]
#D = cbind(scale(D[,-ncol(D)]), D[,ncol(D)])
#D = D[D[,ncol(D)]<2,]
# 
sample_contexts = function(D){
  d = c()
  for (l in 0:1){
    id = sample(which(D[,ncol(D)]==l),1)
    d = rbind(d, D[id,])
  }
  k =  length(unique(D[,ncol(D)]))
  d = d[sample(k,k),]
  return(d)
}
#X = D[,-ncol(D)]#; X = 1+ log(1+X)
#y = D[,ncol(D)]
D1 = cbind(X[, ID],1*(y==1)); 


beta = as.numeric(coef(mod))[-1]
beta = beta[ID]
#beta = as.matrix(as.numeric(beta))
#X = D[,-ncol(D)]; y = D[,ncol(D)]
#pred = predict(mod, X)

### Simulation setup: Parameters ####
nrep = 10 # number of repetation
K =  length(unique(D[,ncol(D)])) # 10 different numbers
p = ncol(D)-1
s = sum(beta != 0)
sigma = 0.1 #sqrt(sum((y- X%*%beta)^2/length(y)))
Th = 400

Target_class = 1

# ESTC params ##
zeta = 0.1/2
n1  = as.integer(Th^(2/3) * (s^2 * log(2*p))^(1/3) * zeta)

regret_list  = foreach(i = 1:nrep, .packages = c("MASS", "sparsevb", "ebreg", "R6", "monomvn", "glmnet")) %dopar% {
  seed = i #i*I(i!=2 && i!= 4 && i<=5) + .10*i*I(i==2) + 100*i*I(i>5) + .3*I(i==4)
  #if(i<=6 && i>=2){seed = i*2}
  #if(i==8){seed = i^3}
  set.seed(seed)
  .GlobalEnv$p <- p
  .GlobalEnv$K <- K
  .GlobalEnv$sigma <- sigma
  #rndBandit = randomBandit$new(p=p, K=K)
  .GlobalEnv$Th = Th
  .GlobalEnv$n1 = n1
  #permutation = sample(Th, Th)
  linucb = LinUCB$new(lambda=2, T0=1, p=p, K=K, error.var=sigma^2)
  drlasso = DR.Lasso$new(lambda1=1, lambda2=0.5, p=p, K=K, tc=1, tr=TRUE, T0=10)
  lasso.opt = Lasso.optimism$new(lambda= 0.5, tau=1, p=p, K=K, T0=10)
  estc = ESTC$new(lambda=0.5, p=p, K=K, T0=n1)
  lints = LinTS$new(delta=0.01, epsilon=1/log(Th), p=p, K=K, error.var=sigma^2)
  vbts = VB.complexity.TS$new(p=p, K=K, u=1.001, lambda = 1, T0=10, T1=1, max_iter = 1000,err.sd=sigma , tol=1e-2)
  #ebregts = EBREG.TS$new(p=p, K=K, T0= 20)
  #blassots = BLasso.TS$new(p=p, K=K, T0 = 10, T1 = 50)
  
  # rewards
  #rndbandit.mean_rewards = numeric(Th)
  linucb.mean_rewards = numeric(Th)
  linucb.incorrect = numeric(Th)
  
  
  drlasso.mean_rewards = numeric(Th)
  drlasso.incorrect = numeric(Th)
  
  lasso_opt.mean_rewards = numeric(Th)
  lasso_opt.incorrect = numeric(Th)
  
  estc.mean_rewards = numeric(Th)
  estc.incorrect = numeric(Th)
  
  lints.mean_rewards = numeric(Th)
  lints.incorrect = numeric(Th)
  
  vbts.mean_rewards = numeric(Th)
  vbts.incorrect = numeric(Th)
  
  #ebregts.mean_rewards = numeric(Th)
  #blassots.mean_rewards = numeric(Th)
  
  opt.mean_rewards = numeric(Th)
  
  
  # times
  #rndbandit.time = 0
  linucb.time = 0
  drlasso.time = 0
  lasso.opt.time = 0
  estc.time = 0
  lints.time = 0
  vbts.time = 0
  #blassots.time = 0
  #ebregts.time = 0
  
  for (t in 1:Th){
    # generic
    M = sample_contexts(D1)
    X = M[,-ncol(M)]
    label = M[,ncol(M)]
    error = 0
    #cat("optimal reward setting","\n")
    opt.mean_rewards[t] = 1
    
    # start = Sys.time()
    # # random
    # a0 = rndBandit$choose.action(t, t(X))
    # chosen.mean = X[a0,] %*% beta
    # rndbandit.mean_rewards[t] = chosen.mean
    # cat("nrep", i,"round",t, "random complete","\n")
    # rndbandit.time = rndbandit.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    # 
    
    # linucb
    # start = Sys.time()
    # #tic()
    # a1 = linucb$choose.action(t, t(X))
    # chosen.mean = X[a1,] %*% beta
    # linucb.mean_rewards[t] = chosen.mean
    # linucb$update.beta(chosen.mean+error, X[a1,], t)
    # linucb.incorrect[t] = 1*(label[a1] != Target_class)
    # cat("nrep", i,"round",t, "Linucb complete","\n")
    # #linucb.time = linucb.time + toc()
    # linucb.time = linucb.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # drlasso
    start = Sys.time()
    a2 = drlasso$choose.action(t, t(X))
    chosen.mean = 2*(label[a2] != Target_class) - 1
    drlasso.mean_rewards[t] = chosen.mean
    drlasso$update.beta(chosen.mean+error, X[a2,], t)
    drlasso.incorrect[t] = 1*(label[a2] != Target_class)
    cat("nrep", i,"round",t, "DRlasso complete","\n")
    drlasso.time = drlasso.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # lasso_opt
    start = Sys.time()
    a3 = lasso.opt$choose.action(t, t(X))
    chosen.mean =  2*(label[a3] != Target_class) - 1
    lasso_opt.mean_rewards[t] = chosen.mean
    lasso.opt$update.beta(chosen.mean+error, X[a3,], t)
    lasso_opt.incorrect[t] = 1*(label[a3] != Target_class)
    cat("nrep", i,"round",t, "Lasso_opt complete","\n")
    lasso.opt.time = lasso.opt.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    
    # ESTC
    start = Sys.time()
    a4 = estc$choose.action(t, t(X))
    chosen.mean = 2*(label[a4] != Target_class) - 1
    estc.mean_rewards[t] = chosen.mean
    estc$update.beta(chosen.mean + error, X[a4,], t)
    estc.incorrect[t] = 1*(label[a4] != Target_class)
    cat("nrep", i,"round",t, "ESTC complete","\n")
    estc.time = estc.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    # 
    
    # # LinTS
    # start = Sys.time()
    # a5 = lints$choose.action(t, t(X))
    # chosen.mean = X[a5,] %*% beta
    # lints.mean_rewards[t] = chosen.mean
    # lints$update.beta(chosen.mean+error, X[a5,], t)
    # cat("nrep", i,"round",t, "LinTS complete","\n")
    # lints.time = lints.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    # lints.incorrect[t] = 1*(label[a5] != Target_class)
    
    # VB_TS
    start = Sys.time()
    a6 = vbts$choose.action(t, t(X))
    chosen.mean = 2*(label[a6] != Target_class) - 1
    vbts.mean_rewards[t] = chosen.mean
    vbts$update.beta((chosen.mean+error), X[a6,], t)
    cat("nrep", i,"round",t, "VB TS complete","\n")
    vbts.time = vbts.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    vbts.incorrect[t] = 1*(label[a6] != Target_class)
    
    # # EBREG_TS
    # start = Sys.time()
    # a7 = ebregts$choose.action(t, t(X))
    # chosen.mean = X[a7,] %*% beta
    # ebregts.mean_rewards[t] = chosen.mean
    # ebregts$update.beta(chosen.mean + error, X[a7,], t)
    # cat("nrep", i,"round",t, "EBREG TS complete","\n")
    # ebregts.time = ebregts.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
    # 
    # # BLASSO_TS
    # start = Sys.time()
    # a8 = blassots$choose.action(t, t(X))
    # chosen.mean = X[a8,] %*% beta
    # blassots.mean_rewards[t] = chosen.mean
    # blassots$update.beta(chosen.mean + error, X[a8,], t)
    # cat("nrep", i,"round",t, "BLASSO TS complete","\n")
    # blassots.time = blassots.time + as.numeric(difftime(time1 = Sys.time(), time2 = start, units = "secs"))
  }
  
  #regret.random = cumsum(opt.mean_rewards) - cumsum(rndbandit.mean_rewards)
  #regret.linucb = cumsum(opt.mean_rewards) - cumsum(linucb.mean_rewards)
  regret.drlasso = cumsum(opt.mean_rewards) - cumsum(drlasso.mean_rewards)
  regret.lasso_opt = cumsum(opt.mean_rewards) - cumsum(lasso_opt.mean_rewards)
  regret.estc = cumsum(opt.mean_rewards) - cumsum(estc.mean_rewards)
  #regret.lints = cumsum(opt.mean_rewards) - cumsum(lints.mean_rewards)
  regret.vbts = cumsum(opt.mean_rewards) - cumsum(vbts.mean_rewards)
  #regret.ebregts = cumsum(opt.mean_rewards) - cumsum(ebregts.mean_rewards)
  #regret.blassots = cumsum(opt.mean_rewards) - cumsum(blassots.mean_rewards)
  
  return(list(regret.drlasso = regret.drlasso, drlasso.time = drlasso.time, drlasso.incorrect = drlasso.incorrect,
              regret.lasso_opt = regret.lasso_opt, lasso_opt.time = lasso.opt.time, lasso_opt.incorrect = lasso_opt.incorrect,
              regret.estc = regret.estc, estc.time = estc.time, estc.incorrect = estc.incorrect,
              regret.vbts = regret.vbts, vbts.time = vbts.time, vbts.incorrect = vbts.incorrect))
}
stopCluster(cluster)
##### Building dataframe ######

methods = c( "DRLasso", "Lasso_L1", "ESTC", "VBTS")
m = length(methods)
regret_list2 = lapply(regret_list, function(x){x[3*(1:m)-2]})
time_list = lapply(regret_list, function(x){x[3*(1:m) - 1]})
incorrect_list = lapply(regret_list, function(x){x[3*(1:m)]})
df = data.frame(Rounds = 1:Th)

w = 1.96/sqrt(nrep)

for (method  in methods) {
  id = which(methods == method)
  l1 = apply(do.call(rbind, lapply(regret_list2, function(x){x[[id]]})), 2, mean)
  assign(paste("meanregret.", method, sep = ""), l1)
  
  l2 = l1 + w * apply(do.call(rbind, lapply(regret_list2, function(x){x[[id]]})), 2, sd)
  #assign(paste("sdregret.", method, sep = ""), l2)
  
  l3 = l1 - w* apply(do.call(rbind, lapply(regret_list2, function(x){x[[id]]})), 2, sd)
  #assign(paste("meantime.", method, sep = ""), l3)
  
  l4 = apply(do.call(rbind, lapply(time_list, function(x){x[[id]]})), 2, mean)
  #assign(paste("meantime.", method, sep = ""), l4)
  
  l5 = apply(do.call(rbind, lapply(incorrect_list, function(x){x[[id]]})), 2, mean)
  
  df_mid = data.frame(l1=l1, l2=l2, l3=l3, l4=l4, l5 = l5)
  colnames(df_mid) = c(paste("meanregret.", method, sep = ""), paste("meanregret_high.", method, sep = ""),
                       paste("meanregret_low.", method, sep = ""), paste("meantime.", method, sep = ""), paste("misclassification_rate.", method, sep = ""))
  df = cbind(df, df_mid)
}



sp = df 
plot(1:Th, sp$meanregret.VBTS, type = "l")
#lines(1:Th,sp$meanregret.Lasso_L1, col = "blue")
lines(1:Th,sp$meanregret.ESTC, col = "green")
lines(1:Th,sp$meanregret.Lasso_L1, col = "blue")
lines(1:Th,sp$meanregret.DRLasso, col = "red")
#lines(1:Th,sp$meanregret.LinUCB, col = "green")
#lines(1:Th,sp$meanregret.LinTS, col = "orange")


