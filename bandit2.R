# code by: SunritC, SaptarshiR (R version:2022.02.3 Build 492)
# implementation of various algorithms for contextual linear bandits in high-dim
# methods used:
# 1. Linear UCB
# 2. doubly robust LASSO
# 3. ESTC
# 4. LASSO_optimism
# 5. IDS
# 6. TS - Bayesian LASSO
# 7. TS - Linear bandit (Agrawal Gowal)
# 8. TS - EBREG
# 9. TS - VB with complexity prior

# each method has a R6 class
# final simulation function connects them
# follows the identical ipynb file
# usual notations: Th (horizon), p (dimension), K (no of arms)
# T0: random actions

# path
#setwd("/Volumes/GoogleDrive/Shared drives/Bandit/Sunrit_Vola/HD_Bandits/Code/R")

# packages required
require(R6)
require(monomvn)
require(sparsevb)
#require(ebreg)
#source("ebreg_new.R")
require(glmnet)
require(functClust)
require(tictoc)

###### 0 Random Bandit #######
randomBandit <- R6Class("randomBandit",
                        public = list(
                          p = NA,
                          K = NA,
                          
                          initialize = function(p, K){
                            self$K = K
                            self$p = p
                          },
                          
                          choose.action = function(t, contexts){
                            action = sample(self$K, 1)
                            return (action)
                          }
                        )
)

###### 1 LinUCB #######
LinUCB <- R6Class("LinUCB",
                  public = list(
                    chosen.contexts = NA,
                    obs.rewards = NA,
                    actions = NA,
                    logdet.V = NA,
                    V.inv = NA,
                    K = NA,
                    p = NA,
                    error.var = NA,
                    beta = NA,
                    T0 = NA,
                    lambda = NA,
                    Xt.y = NA,
                    
                    initialize = function(lambda, T0, p, K, error.var){
                      self$T0 = T0
                      self$p = p
                      self$K = K
                      self$error.var = error.var
                      self$lambda = lambda
                      
                      self$beta = numeric(p)
                      self$logdet.V = p * log(lambda)
                      self$V.inv = diag(1, nrow=p) / lambda
                      self$Xt.y = numeric(p)
                      self$chosen.contexts = c()
                      self$obs.rewards = c()
                    },
                    
                    choose.action = function(t, contexts){
                      # contexts has dim (d, K)
                      if (t <= self$T0){
                        action = sample(self$K, 1)
                      }else{
                        delta = 1 / (1+t)
                        a = self$logdet.V - self$p * log(self$lambda) - 2*log(delta)
                        #print(a)
                        ft = self$error.var * sqrt(a) + sqrt(self$lambda)
                        UCB = numeric(self$K)
                        score_est = t(contexts) %*% self$beta
                        for (k in 1:K){
                          UCB[k] = score_est[k] + ft * ((t(contexts[,k]) %*% self$V.inv %*% contexts[,k])^0.5)
                        }
                        action = which.max(UCB)
                      }
                      return (action) 
                    },
                    
                    update.beta = function(reward, chosen.context, t){
                      
                      self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                      self$obs.rewards = append(self$obs.rewards, reward)
                      self$Xt.y = self$Xt.y + as.numeric(reward) * chosen.context
                      
                      x = chosen.context
                      u = self$V.inv %*% x
                      self$logdet.V = self$logdet.V + log(1 + as.numeric(t(x)%*% u))
                      den = as.numeric(1 + t(x)%*%u)
                      self$V.inv = self$V.inv - (u %*% t(u)) / den
                      X = self$chosen.contexts; y = self$obs.rewards
                      self$beta = self$V.inv %*% self$Xt.y
                    }
                  )
)

###### 2 DR LASSO #######
DR.Lasso <- R6Class("DR.Lasso",
                    public = list(
                      p = NA,
                      K = NA,
                      beta = NA,
                      tc = NA,
                      tr = NA,
                      T0 = NA,
                      lambda1 = NA,
                      lambda2 = NA,
                      chosen.contexts = NA,
                      obs.rewards = NA,
                      pi = NA,
                      rhat = NA,
                      current.action = NA,
                      r = NA,
                      
                      initialize = function(lambda1,lambda2, p, K, tc, tr, T0){
                        self$p = p
                        self$K = K
                        self$tc = tc
                        self$tr = tr
                        self$T0 = T0
                        self$lambda1 = lambda1
                        self$lambda2 = lambda2
                        self$beta = numeric(p)
                        self$pi = 1 / K
                        self$r = c()
                        self$obs.rewards = c()
                      },
                      
                      choose.action = function(t, contexts){
                        # contexts has dim (p, K)
                        if (t <= self$T0){
                          action = sample(self$K, 1)
                        }else{
                          uniformp = self$lambda1 * sqrt((log(t) + log(self$p))/t)
                          uniformp = min(c(1, max(c(0, uniformp)) ))
                          choice = sample( c(0,1), 1,  prob=c(1-uniformp, uniformp))
                          est = t(contexts) %*% self$beta
                          #print(est)
                          if (choice==1){
                            action = sample(self$K, 1)
                            if (action == which.max(est)){
                              self$pi = uniformp/self$K + (1 - uniformp)
                            }else{
                              self$pi = uniformp/self$K
                            }
                          }else{
                            action = which.max(est)
                            self$pi =  uniformp/self$K + (1 - uniformp)
                          }
                          
                        }
                        x = rowMeans(contexts)
                        if(t==1){
                          self$chosen.contexts = matrix(x, nrow=1)
                        }else{
                          self$chosen.contexts = rbind(self$chosen.contexts, x)
                        }
                        self$rhat = t(contexts) %*% self$beta
                        self$current.action = action
                        return (action)
                      },
                      
                      update.beta = function(reward, chosen.context, t){
                        self$obs.rewards = append(self$obs.rewards, reward)
                        
                        pseudo_r = mean(self$rhat) + (reward - self$rhat[self$current.action]) / (self$pi * self$K)
                        if (self$tr == TRUE){
                          pseudo_r = min(c(3, max(c(-3, pseudo_r))))
                        }
                        self$r = append(self$r, pseudo_r)
                        
                        if (t>5){
                          if (t>self$tc){
                            lambda2_t = self$lambda2 * sqrt((log(t) + log(self$p))/t)
                          }
                          lasso = glmnet(self$chosen.contexts, self$r, lambda=lambda2_t, 
                                         family="gaussian", intercept = F, standardize = F)
                          self$beta = as.numeric(lasso$beta)
                        }
                      }
                    )
)

###### 3 ESTC #######
ESTC = R6Class("ESTC",
               public = list(
                 chosen.contexts = NA,
                 obs.rewards = NA,
                 p = NA,
                 K = NA,
                 lambda = NA,
                 T0 = NA, # this has a form based on actual time horizon and zeta
                 beta = NA,
                 
                 initialize = function(lambda, p, K, T0){
                   self$lambda = lambda * sqrt(log(p)/T0)
                   self$p = p
                   self$K = K
                   self$T0 = T0
                   self$beta = numeric(p)
                   self$obs.rewards = c()
                   self$chosen.contexts = c()
                 },
                 
                 choose.action = function(t, contexts){
                   if (t <= self$T0){
                     action = sample(self$K, 1)
                   }else{
                     est = t(contexts) %*% self$beta
                     action = which.max(est)
                   }
                   return (action)
                 },
                 
                 update.beta = function(reward, chosen.context, t){
                   self$obs.rewards = append(self$obs.rewards, reward)
                   self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                   
                   if (t == self$T0){
                     lasso = glmnet(self$chosen.contexts, self$obs.rewards, family='gaussian', intercept=F, standardize = F, lambda=self$lambda)
                     self$beta = as.numeric(lasso$beta)
                   }
                 }
               )
)

###### 4 LASSO Optimism #######
Lasso.optimism = R6Class("Lasso.optimism",
                         public = list(
                           p = NA,
                           K = NA,
                           chosen.contexts = NA,
                           obs.rewards = NA,
                           lambda = NA,
                           tau = NA,
                           beta = NA,
                           T0 = NA,
                           
                           initialize = function(lambda, tau, p, K, T0){
                             self$p = p
                             self$K = K
                             self$lambda= lambda
                             self$tau = tau
                             self$beta = numeric(p)
                             self$obs.rewards = c()
                             self$chosen.contexts = c()
                             self$T0 = T0
                           },
                           
                           choose.action = function(t, contexts){
                             # contexts has dim (p, K)
                             if (t <= self$T0){
                               action = sample(self$K, 1)
                             }else{
                               est = t(contexts) %*% self$beta
                               UCB = numeric(self$K)
                               for(k in 1:self$K){
                                 UCB[k] = est[k] + self$tau * max(abs(contexts[,k]))
                               }
                               action = which.max(UCB)
                             }
                             self$lambda = self$lambda * sqrt((log(t) + log(self$p))/t)
                             self$tau = self$tau * sqrt((log(t) + log(self$p))/t)
                             return (action)
                           },
                           
                           update.beta = function(reward, chosen.context, t){
                             
                             self$obs.rewards = append(self$obs.rewards, reward)
                             self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                             if (t > self$T0){
                               lasso = glmnet(self$chosen.contexts, self$obs.rewards, family='gaussian', intercept=F, standardize = F, lambda=self$lambda)
                               self$beta = as.numeric(lasso$beta)
                             }
                           }
                         )
)

###### 5 IDS #######


###### 6 TS - Bayesian LASSO #######
BLasso.TS = R6Class("BLasso.TS",
                    public = list(
                      p = NA,
                      K = NA,
                      beta = NA,
                      obs.rewards = NA,
                      chosen.contexts = NA,
                      T0 = NA,
                      T1 = NA,
                      
                      initialize = function(p, K, T0, T1){
                        self$p = p
                        self$K = K
                        self$beta = numeric(p)
                        self$obs.rewards = c()
                        self$chosen.contexts = c()
                        self$T0 = T0
                        self$T1 = T1
                      },
                      
                      choose.action = function(t, contexts){
                        if (t < self$T0){
                          action = sample(self$K, 1)
                        }else{
                          est = t(contexts) %*% self$beta
                          action = which.max(est)
                        }
                        return (action)
                      },
                      
                      update.beta = function(reward, chosen.context, t){
                        self$obs.rewards = append(self$obs.rewards, reward)
                        self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                        if (t >= self$T0){
                          if (t %% self$T1 == self$T0+1){
                            reg.lasso = blasso(X=self$chosen.contexts,
                                               y=self$obs.rewards,
                                               T = 500,
                                               thin = NULL,
                                               RJ = TRUE,
                                               M = NULL,
                                               beta = NULL, 
                                               lambda2 = 1,
                                               s2 = var(self$obs.rewards-mean(self$obs.rewards)),
                                               case = c("default"),
                                               mprior = c(1, p^1.01), rd = NULL, ab = NULL,
                                               theta = 0, rao.s2 = TRUE, icept = F,
                                               normalize = F, verb = 0)
                            self$beta = reg.lasso$beta[500,]
                          }
                          
                        }
                      }
                    ))

###### 7 Lin TS #######
LinTS = R6Class("LinTS",
                public = list(
                  chosen.contexts = NA,
                  obs.rewards = NA,
                  v2 = NA,
                  p = NA,
                  K = NA,
                  beta = NA,
                  mu.hat = NA,
                  f = NA,
                  B.inv = NA,
                  
                  initialize = function(delta, epsilon, p, K, error.var){
                    self$v2 = sqrt(24 * error.var * p * log(1/delta)/ epsilon)
                    self$p = p
                    self$K = K
                    self$chosen.contexts = c()
                    self$obs.rewards = c()
                    self$beta = numeric(p)
                    self$f = numeric(p)
                    self$mu.hat = numeric(p)
                    self$B.inv = diag(1, nrow=p)
                  },
                  
                  choose.action = function(t, contexts){
                    mu.TS = self$mu.hat #mvrnorm(1, mu=self$mu.hat, Sigma=self$v2 * self$B.inv)
                    est = t(contexts) %*% mu.TS
                    action = which.max(est)
                    return (action)
                  },
                  
                  update.beta = function(reward, chosen.context, t){
                    self$obs.rewards = append(self$ob.rewards, reward)
                    self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                    u = self$B.inv %*% chosen.context
                    self$B.inv = self$B.inv - (u %*% t(u)) / (1 + as.numeric(t(chosen.context) %*% u))
                    self$f = self$f + as.numeric(reward) * chosen.context
                    self$mu.hat = self$B.inv %*% self$f
                  }
                )
)

###### 8 TS - EBREG #######
EBREG.TS = R6Class("EBREG.TS",
                   public = list(
                     p = NA,
                     K = NA,
                     chosen.contexts = NA,
                     obs.rewards = NA,
                     beta = NA,
                     T0 = NA,
                     T1 = NA,
                     err.sd = NA,
                     
                     log.f = function(x, n){
                       a = -x * (log(1) + 0.05*log(self$p)) + log(x <= n)
                       return (a)
                     },
                     
                     initialize = function(p, K, T0, T1){
                       self$p = p
                       self$K = K
                       self$beta = numeric(p)
                       self$chosen.contexts = c()
                       self$obs.rewards = c()
                       self$T0 = T0
                       self$T1 = T1
                       self$err.sd = 1
                     },
                     
                     choose.action = function(t, contexts){
                       if (t < self$T0){
                         action = sample(self$K, 1)
                       }else{
                         est = t(contexts) %*% self$beta
                         action = which.max(est)
                       }
                       return (action)
                     },
                     
                     update.beta = function(reward, chosen.context, t){
                       self$obs.rewards = append(self$obs.rewards, reward)
                       self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                       
                       log.f.curr = function(x){
                         return (self$log.f(x, t))
                       }
                       
                       if (t>= self$T0){
                         if (t%%self$T1 == self$T0){
                           model = ebreg_new(
                             self$obs.rewards,
                             self$chosen.contexts,
                             matrix(1, nrow=1,ncol= self$p),
                             b.init = NULL,
                             standardized = TRUE,
                             alpha = 0.99,
                             gam = 0.005,
                             sig2= self$err.sd^2,
                             prior = FALSE,
                             log.f = log.f.curr,
                             M = 1000,
                             sample.beta = TRUE,
                             pred = FALSE,
                             conf.level = 0.95
                           )
                         }else{
                           model = ebreg_new(
                             self$obs.rewards,
                             self$chosen.contexts,
                             matrix(1, nrow=1,ncol= self$p),
                             b.init = self$beta,
                             standardized = TRUE,
                             alpha = 0.99,
                             gam = 0.005,
                             sig2= self$err.sd^2,
                             prior = FALSE,
                             log.f = log.f.curr,
                             M = 1000,
                             sample.beta = TRUE,
                             pred = FALSE,
                             conf.level = 0.95
                           )
                         }
                         self$beta = model$beta[1000,]
                         beta.hat = colMeans(model$beta)
                         RSS = mean((self$obs.rewards - self$chosen.contexts %*% beta.hat)^2)
                         self$err.sd = sqrt(RSS)
                       }
                     }
                   )
)

###### 9 TS - VB complexity prior (OURS) #######
VB.complexity.TS = R6Class("VB.complexity.TS",
                           public = list(
                             p = NA,
                             K = NA,
                             u = NA,
                             beta = NA,
                             T0 = NA,
                             T1 = NA,
                             chosen.contexts = NA,
                             obs.rewards = NA,
                             sigma.hat = NA,
                             mu.hat = NA,
                             gamma.hat = NA,
                             lambda = NA,
                             err.sd = NA,
                             
                             initialize = function(p, K, u=1.001, lambda=1, T0, T1){
                               self$p = p
                               self$K = K
                               self$u = u
                               self$beta = numeric(p)
                               self$T0 = T0
                               self$T1 = T1
                               self$chosen.contexts = c()
                               self$obs.rewards = c()
                               self$sigma.hat = rep(1, p)
                               self$mu.hat = rep(0, p)
                               self$gamma.hat = rep(0.5, p)
                               self$err.sd = 1
                               self$lambda = lambda
                             },
                             
                             choose.action = function(t, contexts){
                               if (t <= self$T0){
                                 action = sample(self$K, 1)
                               }else{
                                 est = t(contexts) %*% self$beta
                                 action = which.max(est)
                               }
                               return (action)
                             },
                             
                             update.beta = function(reward, chosen.context, t, fixed=TRUE){
                               self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                               self$obs.rewards = append(self$obs.rewards, reward)
                               
                               if (fixed == TRUE){ # modified
                                 new_lambda = self$lambda
                               }else{
                                 new_lambda = self$lambda * sqrt(t)
                                }
                                
                               
                               if (t >= self$T0){
                                 if ((t - self$T1) %% self$T1 == (self$T1 - 1)){
                                   vb.sparse = svb.fit(
                                     self$chosen.contexts,
                                     self$obs.rewards,
                                     family = "linear",
                                     slab = "laplace",
                                     alpha = 1,
                                     beta = p^self$u,
                                     prior_scale = new_lambda,
                                     intercept = FALSE,
                                     max_iter = 1000,
                                     noise_sd = self$err.sd,
                                     tol = 1e-05
                                   )
                                 }else{
                                   vb.sparse = svb.fit(
                                     self$chosen.contexts,
                                     self$obs.rewards,
                                     family = "linear",
                                     slab = "laplace",
                                     mu = self$mu.hat,
                                     sigma = self$sigma.hat,
                                     gamma = self$gamma.hat,
                                     alpha = 1,
                                     beta = p^self$u,
                                     prior_scale = new_lambda,
                                     intercept = FALSE,
                                     max_iter = 1000,
                                     noise_sd = self$err.sd,
                                     tol = 1e-05
                                   )
                                 }
                                 
                                 self$sigma.hat = vb.sparse$sigma
                                 self$mu.hat = vb.sparse$mu
                                 self$gamma.hat = vb.sparse$gamma
                                 
                                 RSS = 0
                                 beta.hat = numeric(self$p)
                                 m = 30
                                 for (j in 1:m){
                                   beta.hat = beta.hat + ((vb.sparse$mu + vb.sparse$sigma * rnorm(self$p) ) * rbinom(n=self$p, size=1, prob=vb.sparse$gamma))/m
                                 }
                                 RSS = mean((self$obs.rewards - self$chosen.contexts %*% beta.hat)^2)
                                 self$beta = (vb.sparse$mu + vb.sparse$sigma * rnorm(self$p) ) * rbinom(n=self$p, size=1, prob=vb.sparse$gamma)
                                 self$err.sd = sqrt(RSS)
                               }
                             }
                             
                           )
)

###### 10. Sparsity Agnostic LASSO / Can do Ridge with alpha (use as elastic net parameter) ####
SA.Lasso = R6Class("SA.Lasso",
                   public = list(
                     p = NA,
                     K = NA,
                     lambda0 = NA,
                     beta= NA,
                     chosen.contexts = NA,
                     obs.rewards = NA,
                     alpha = NA,
                     
                     initialize = function(p, K, lambda0=1, alpha=1){
                       self$p = p
                       self$K = K
                       self$lambda0 = lambda0
                       self$alpha = alpha
                       self$beta = numeric(p)
                       self$chosen.contexts = c()
                       self$obs.rewards = c()
                     },
                     
                     choose.action = function(t, contexts){
                       est = t(contexts) %*% self$beta
                       action = which.max(est)
                       return(action)
                     },
                     
                     update.beta = function(reward, chosen.context, t){
                       
                       self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                       self$obs.rewards = append(self$obs.rewards, reward)
                       
                       if (t>5){
                         lambda = self$lambda0 * sqrt((4*log(t)+2*log(self$p)) / t)
                         lasso = glmnet(self$chosen.contexts, self$obs.rewards, family='gaussian', intercept=F, standardize = F, lambda=lambda/2, alpha=self$alpha)
                         self$beta = as.numeric(lasso$beta)}
                     }
                   )
)

###### 11. Thresholded Lasso ####
Threshold.Lasso = R6Class("Threshold.Lasso",
                          public = list(
                            p = NA,
                            K = NA,
                            lambda0 = NA,
                            beta= NA,
                            chosen.contexts = NA,
                            obs.rewards = NA,
                            alpha = NA,
                            
                            initialize = function(p, K, lambda0=1, alpha=1){
                              self$p = p
                              self$K = K
                              self$lambda0 = lambda0
                              self$alpha = alpha
                              self$beta = numeric(p)
                              self$chosen.contexts = c()
                              self$obs.rewards = c()
                            },
                            
                            choose.action = function(t, contexts){
                              est = t(contexts) %*% self$beta
                              action = which.max(est)
                              return(action)
                            },
                            
                            update.beta = function(reward, chosen.context, t){
                              self$chosen.contexts = rbind(self$chosen.contexts, chosen.context)
                              self$obs.rewards = append(self$obs.rewards, reward)
                              
                              if (t>5){
                              lambda = self$lambda0 * sqrt(2*log(t)*log(self$p)/t)
                              lasso = glmnet(self$chosen.contexts, self$obs.rewards, family='gaussian', intercept=F, standardize = F, lambda=lambda/2, alpha=self$alpha)
                              beta = as.numeric(lasso$beta)
                              
                              abs.beta = abs(beta)
                              S0 = which(abs.beta > 4 * lambda)
                              n0 = length(S0)
                              if (n0 == 0){
                                S1 = which.max(abs.beta)
                              }else{
                                S1 = which(abs.beta > 4 * lambda * sqrt(n0))
                              }
                              
                              b = self$obs.rewards
                              A = self$chosen.contexts[,S1]
                              theta = solve(t(A)%*%A, t(A)%*%b)
                              beta = numeric(self$p)
                              beta[S1] = theta
                              self$beta = beta}
                            }
                          )
)
