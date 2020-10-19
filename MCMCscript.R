MCMC_noBlock <- function(train_win_X, train_win_Y, test_win_X, test_win_Y, p, total_itr=10000, kappa=0.02, remember=1000){
  num_avg <- 50000
  
  set.seed(1234)
  n_train <- length(train_win_X)
  n_test <- length(test_win_X)
  q <- dim(train_win_X[[1]])[2]

  
  ####################
  ### Dynamic Init ###
  ####################
  
  # Prior for balpha_0: Normal(m_0, C_0)
  m_0 <- rep(0, p)
  C_0 <- diag(p)
  balpha_0 <- mvrnorm(n = 1, m_0, C_0)
  
  ##########################
  ### Dynamic Model Init ###
  ##########################
  
  # Prior for phi (w^{-1}) : Gamma(a_0, b_0)
  a_0 <- 1/2
  b_0 <- 1/2
  phi <- rgamma(1, shape = a_0, rate = b_0)
  w <- 1/phi
  
  # Prior for btheta_{1:p}: Normal(theta_mu_0, theta_Sigma_0)
  theta_mu_0 <- rep(0, p)
  theta_Sigma_0 <- diag(p) *0.01
  btheta <- as.matrix(mvrnorm(n = 1, theta_mu_0, theta_Sigma_0/phi))
  
  ##############################
  ### Observation Model Init ###
  ##############################
  
  # Random initialize for tau (v): no density
  tau <- 1
  v <- 1/tau
  
  # Prior for beta_{1:q}: Normal(beta_mu_0, beta_Sigma_0)
  beta_mu_0 <- rep(0, q)
  beta_Sigma_0 <- diag(q) * ( kappa^-1) * (tau^-1)
  bbeta <- as.matrix(mvrnorm(1, beta_mu_0, beta_Sigma_0))
  
  
  ######################
  ### Container Init ###
  ######################
  
  record_alphas <- NULL
  record_thetas <- NULL
  record_betas <- NULL
  record_ws <- NULL
  record_vs <- NULL
  record_ms <- NULL
  record_Cs <- NULL
  record_res <- NULL
  record_mse <- NULL
  predictions <- as.list(rep(0, n_test))
  
  ####################
  #### MCMC Start ####
  # ####################
  
  
  for(itr in 1:total_itr)
  {
    if(itr %% 100 == 0){
      message(cat("finished iteration ", itr))
    }
    m <- list()
    C <- list()
    a <- list()
    R <- list()
    ms <- list()
    Cs <- list()
    balphas <- list()
    alphas <- c()
    residuals <- list()  
    a_pred <- list()
    R_pred <- list()
    mse <- c()
    
    # Make \bTHETA
    jordan_mtx <- cbind(diag(p-1), rep(0, p-1))
    bTHETA <- rbind(t(btheta), jordan_mtx)
    
    #############################
    ##### Forward Filtering #####
    #############################
    
    for(i in 1:n_train)
    {
      n_i <- dim(train_win_X[[i]])[1]
      # Make \bm{1}
      bm_1_row <- cbind(c(1), t(rep(0, p-1)))
      bm_1 <- matrix(rep( t(bm_1_row), n_i), nrow=n_i, byrow=TRUE)
      if(i==1)
      {
        a_i <- bTHETA %*% m_0
        R_i <- bTHETA %*% C_0 %*% t(bTHETA)+ w*diag(p)
        m_i <- solve(solve(R_i) + (1/v) * t(bm_1) %*% bm_1)  %*% 
          (solve(R_i)%*%a_i + 1/v * t(bm_1) %*%(train_win_Y[[i]] - train_win_X[[i]]%*%bbeta) )
        C_i <- solve(solve(R_i) + (1/v) * t(bm_1) %*% bm_1)
        a[[i]] <- a_i
        R[[i]] <- R_i
        m[[i]] <- m_i
        C[[i]] <- C_i
      }
      else{
        a_i <- bTHETA %*% m[[i-1]]
        R_i <- bTHETA %*% C[[i-1]] %*% t(bTHETA)+ w*diag(p)
        m_i <- solve(solve(R_i) + (1/v) * t(bm_1) %*% bm_1)  %*% 
          (solve(R_i)%*%a_i + 1/v * t(bm_1) %*%(train_win_Y[[i]] - train_win_X[[i]] %*% as.matrix(bbeta)) )
        C_i <- solve(solve(R_i) + (1/v) * t(bm_1) %*% bm_1)
        m[[i]] <- m_i
        C[[i]] <- C_i
        a[[i]] <- a_i
        R[[i]] <- R_i
      }
    }
    
    ### Make a prediction
    for(i in 1:n_test)
    {
      n_i <- dim(test_win_X[[i]])[1]
      # Make \bm{1}
      bm_1_row <- cbind(c(1), t(rep(0, p-1)))
      bm_1 <- matrix(rep( t(bm_1_row), n_i), nrow=n_i, byrow=TRUE)
      if(i==1)
      {
        a_pred_i <- bTHETA %*% m[[n_train]]
        R_pred_i <- bTHETA %*% C[[n_train]] %*% t(bTHETA)+ w*diag(p)
        a_pred[[i]] <- a_pred_i
        R_pred[[i]] <- R_pred_i
        y_pred <- bm_1 %*% a_pred_i + test_win_X[[i]] %*% as.matrix(bbeta)
        if(itr>total_itr-num_avg)
        {
          predictions[[i]] <- predictions[[i]] + as.matrix(y_pred)
        }
      }
      else{
        a_pred_i <- bTHETA %*% a_pred[[i-1]]
        R_pred_i <- bTHETA %*% R_pred[[i-1]] %*% t(bTHETA)+ w*diag(p)
        a_pred[[i]] <- a_pred_i
        R_pred[[i]] <- R_pred_i
        y_pred <- bm_1 %*% a_pred_i + test_win_X[[i]] %*% as.matrix(bbeta)
        if(itr>total_itr-num_avg)
        {
          predictions[[i]] <- predictions[[i]] + as.matrix(y_pred)
        }
      }
    }
    
    #############################
    ##### Backward Sampling #####
    #############################
    
    # n_train = T  
    ms[[n_train]] = m[[n_train]]
    Cs[[n_train]] = C[[n_train]]
    balpha_T <- mvrnorm(n = 1, ms[[n_train]], C[[n_train]])
    balphas[[n_train]] = balpha_T
    alphas[n_train] = balpha_T[1]
    for(i in (n_train-1):1)
    {
      # RTS smoother mean and variance
      J_i <- C[[i]] %*% t(bTHETA) %*% solve(R[[i+1]])
      ms_i <- m[[i]] + J_i %*% (ms[[i+1]] - bTHETA %*% m[[i]])
      Cs_i <- C[[i]] + J_i %*% (Cs[[i+1]] - R[[i+1]]) %*% t(J_i)
      
      # Backward sample Conditional balpha_t
      cond_ms_i <- m[[i]] + J_i %*% (balphas[[i+1]] - bTHETA %*% m[[i]])
      cond_Cs_i <- C[[i]] + J_i %*% R[[i+1]] %*% t(J_i)
      balpha_i <-  mvrnorm(n = 1, cond_ms_i, cond_Cs_i)
      
      # Record smoothed mean and variance
      ms[[i]] <- ms_i
      Cs[[i]] <- Cs_i
      
      # Record backward samples
      balphas[[i]] <- balpha_i
      alphas[i] <- balpha_i[1]   
    }
    
    # Deal with t=0 problem
    # RTS smoother mean and variance
    J_0 <- C_0 %*% t(bTHETA) %*% solve(R[[1]])
    ms_0 <- m_0 + J_0 %*% (ms[[1]] - bTHETA %*% m_0)
    Cs_0 <- C_0 + J_0 %*% (Cs[[1]] - R[[1]]) %*% t(J_0)
    
    # Backward sample Conditional balpha_t
    cond_ms_0 <- m_0 + J_0 %*% (balphas[[1]] - bTHETA %*% m_0)
    cond_Cs_0 <- C_0 + J_0 %*% R[[1]] %*% t(J_0)
    balpha_0 <-  mvrnorm(n = 1, cond_ms_0, cond_Cs_0)  
    
    
    ##################################
    ##### Dynamic Model Sampling #####
    ##################################
    
    Dyn_y <- as.matrix(alphas)
    Dyn_X <- t(cbind(balpha_0, t(do.call(rbind, balphas[c(1:n_train-1)]))) )
    
    
    theta_mu_n <- solve( t(Dyn_X) %*% Dyn_X + solve(theta_Sigma_0) ) %*% 
      (solve(theta_Sigma_0) %*% theta_mu_0 + t(Dyn_X) %*% Dyn_y)
    theta_Sigma_n <- solve(t(Dyn_X) %*% Dyn_X + solve(theta_Sigma_0))
    
    a_n <- a_0 + n_train/2
    b_n <- b_0 + 1/2 * (  t(Dyn_y) %*% Dyn_y + 
                            t(theta_mu_0) %*% solve(theta_Sigma_0) %*% theta_mu_0 -
                            t(theta_mu_n) %*% solve(theta_Sigma_n) %*% theta_mu_n )
    
    # Sample and update w = phi^{-1}
    phi <- rgamma(1, shape = a_n, rate = b_n )
    w <- 1 / phi
    # Sample and update btheta
    btheta <- mvrnorm(n=1, theta_mu_n, solve(theta_Sigma_n)/phi )
    
    
    ######################################
    ##### Observation Model Sampling #####
    ######################################
    
    # Make z_t
    
    train_win_Z <- lapply(c(1:n_train), function(idx) train_win_Y[[idx]] - alphas[idx])
    
    Obs_Z <- do.call(rbind, train_win_Z)
    Obs_X <- do.call(rbind, train_win_X)
    
    N <- dim(Obs_Z)[1]
    
    # Sample and update tau = v^{-1}
    tau <- rgamma(1, (N+q)/2, kappa/2 * t(bbeta) %*% bbeta)
    v <- 1/tau
    
    # Calculate posterior for beta | tau
    beta_mu_n <- solve( t(Obs_X) %*% Obs_X + kappa * diag(q)) %*% t(Obs_X) %*% Obs_Z 
    beta_Sigma_n <- 1/tau * solve( t(Obs_X) %*% Obs_X + kappa * diag(q))
    
    # Sample and update bbeta
    bbeta <- mvrnorm(1, beta_mu_n, beta_Sigma_n)
    
    # Calculate residuals
    res <- as.matrix(Obs_Z - Obs_X %*% bbeta)
    
    
    ##################################
    ##### record MCMC trajectory #####
    ##################################
    
    if(itr>total_itr-remember)
    {
      record_alphas <- rbind(record_alphas, alphas)
      record_thetas <- rbind(record_thetas, btheta)
      record_betas <- rbind(record_betas, bbeta)
      record_ws <- rbind(record_ws, w)
      record_vs <- rbind(record_vs, v)
      record_res <- rbind(record_res, t(res))
    }
  }
  
  #########################
  ##### Calculate MSE #####
  #########################
  record_mse <- c()
  average_y_pred <- list() 
  for(j in n_test)
  {
     y_pred_j <- predictions[[j]] / num_avg
     mse_j <- MSE(y_pred_j, test_win_Y[[j]])
     record_mse[j] <- mse_j
  }
    
    
  results <- list("record_alphas" = record_alphas,
                  "record_thetas" = record_thetas,
                  "record_betas" = record_betas,
                  "record_ws" = record_ws,
                  "record_vs" = record_vs,
                  "record_res" = record_res,
                  "record_mse" = record_mse)
  
  return(results)
}
