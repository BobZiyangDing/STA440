GodDamnItMCMC <- function(total_itr, p, q, n_train, train_win_X, train_win_Y)
{
  record_alphas <- NULL
  record_thetas <- NULL
  record_betas <- NULL
  record_ws <- NULL
  record_vs <- NULL
  
  for(i in 1:total_itr)
  {
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
        R_i <- t(bTHETA) %*% C[[i-1]] %*% bTHETA+ w*diag(p)
        m_i <- solve(solve(R_i) + (1/v) * t(bm_1) %*% bm_1)  %*% 
          (solve(R_i)%*%a_i + 1/v * t(bm_1) %*%(train_win_Y[[i]] - train_win_X[[i]] %*% as.matrix(bbeta)) )
        C_i <- solve(solve(R_i) + (1/v) * t(bm_1) %*% bm_1)
        m[[i]] <- m_i
        C[[i]] <- C_i
        a[[i]] <- a_i
        R[[i]] <- R_i
      }
    }
    
    
    
    #############################
    ##### Backward Sampling #####
    #############################
    
    # n_train = T  
    ms[[n_train]] = m[[n_train]]
    Cs[[n_train]] = C[[n_train]]
    balpha_T <- mvrnorm(n = 1, m_0, C_0)
    balphas[[n_train]] = balpha_T
    alphas[n_train] = balpha_T[1]
    for(i in (n_train-1):1)
    {
      # RTS smoother mean and variance
      J_i <- C[[i]] %*% t(bTHETA) %*% solve(R[[i]])
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
    J_0 <- C_0 %*% t(bTHETA) %*% solve(R[[i]])
    ms_0 <- m_0 + J_0 %*% (ms[[1]] - bTHETA %*% m_0)
    Cs_0 <- C_0 + J_0 %*% (Cs[[1]] - R[[1]]) %*% t(J_0)
    
    # Backward sample Conditional balpha_t
    cond_ms_0 <- m_0 + J_0 %*% (balphas[[1]] - bTHETA %*% m_0)
    cond_Cs_0 <- C_0 + J_0 %*% R[[1]] %*% t(J_0)
    balpha_0 <-  mvrnorm(n = 1, cond_ms_0, cond_Cs_0)  
    
    # # Update prior with posterior for future iterations
    # m_0 <- ms_0
    # C_0 <- Cs_0
    
    
    ##################################
    ##### Dynamic Model Sampling #####
    ##################################
    
    Dyn_y <- as.matrix(alphas)
    Dyn_X <- t(cbind(balpha_0, t(do.call(rbind, balphas[c(1:n_train-1)]))) )
    
    
    theta_mu_n <- solve( t(Dyn_X) %*% Dyn_X + theta_Sigma_0 ) %*% 
      (theta_Sigma_0 %*% theta_mu_0 + t(Dyn_X) %*% Dyn_y)
    theta_Sigma_n <- t(Dyn_X) %*% Dyn_X + theta_Sigma_0
    a_n <- a_0 + n_train/2
    b_n <- b_0 + 1/2 * (  t(Dyn_y) %*% Dyn_y + 
                            t(theta_mu_0) %*% theta_Sigma_0 %*% theta_mu_0 +
                            t(theta_mu_n) %*% theta_Sigma_n %*% theta_mu_n )
    
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
    
    ##################################
    ##### record MCMC trajectory #####
    ##################################
    
    record_alphas <- rbind(record_alphas, alphas)
    record_thetas <- rbind(record_thetas, btheta)
    record_betas <- rbind(record_betas, bbeta)
    record_ws <- rbind(record_ws, w)
    record_vs <- rbind(record_vs, v)
  }
  
  return( list(record_alphas,record_thetas,record_betas,record_ws,record_vs) )
}

