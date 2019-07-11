data {
  int<lower=1> T; // number of observations
  int<lower=1> J; // dimension of observations
  vector[J] Y[T]; // observations
  vector[J] Zero; // a vector of Zeros (fixed means of observations)
}
parameters {
  vector[T] Ytrue; // unobserved 'true' GDP series
  real mu; // Mean growth rate of 'latent 'true' GDP
  real<lower=0, upper=1> rho; // AR(1) Coef on 'true' GDP series
  real<lower = 0> tau; //variance of true GDP series   
  cholesky_factor_corr[J] L_Omega;  
  vector<lower=0>[J] L_sigma; 
}

model {
  #local variables
  matrix[J,J] L_Sigma;
  L_Sigma = diag_pre_multiply(L_sigma, L_Omega);

  #priors
  Ytrue[1] ~ normal(.7,1);            //inital value of 'true' GDP
  tau ~ cauchy(0,1);                  //variance of true GDP
  mu ~ normal(0.8, 1);                //mean growth rate of 'true' GDP
  rho ~ normal(0.5, 1);               //peristence of 'true' GDP
  L_Omega ~ lkj_corr_cholesky(4);
  L_sigma ~ cauchy(0, 2.5);
  
  #likelihood
  for(t in 2:T) {
    Ytrue[t] ~ normal(mu*(1-rho) + rho*Ytrue[t-1], tau);
    Y[t]-Ytrue[t] ~ multi_normal_cholesky(Zero, L_Sigma);
  }
}

generated quantities {
  matrix[J,J] Omega;
  matrix[J,J] Sigma;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, L_sigma);
}
