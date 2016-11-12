data {
  int T; // number of obs activity data
  matrix[T,3] Y; //dataset of generated series
  real phi; //scale parameter in unrestricted covariance
}

parameters {
  vector[T] GDPR; // unobserved 'true' GDP series
  real mu; // Mean growth rate of 'latent 'true' GDP
  real<lower=0, upper=1> rho; // AR(1) Coef on 'true' GDP series
  vector<lower = 0>[4] tau; //variance of GDP series   
  real<lower = 0> corr_ei; //correlation between expenditure, income
  real<lower = 0> corr_ep; //correlation between expenditure, prodn
  real<lower = 0> corr_ip; //correlation between income, production
  real<upper = 0> corr_gi; //correlation between 'true' and income GDP errors
  real<upper = 0> corr_gp; //correlation between 'true' and prodn GDP errors
}

transformed parameters {
  matrix[4,4] sigma;
  matrix[T,4] meanall;
  matrix[T,4] dat;

  sigma[1,1] = tau[1];
  sigma[1,2] = (tau[1]/(phi*(1-rho*rho)) - tau[1]/(1-rho*rho) - tau[2] )/2;
  sigma[1,3] = corr_gi;
  sigma[1,4] = corr_gp;  

  sigma[2,1] = sigma[1,2];
  sigma[2,2] = tau[2];
  sigma[2,3] = corr_ei;
  sigma[2,4] = corr_ep;  
  
  sigma[3,1] = sigma[1,3];
  sigma[3,2] = sigma[2,3];
  sigma[3,3] = tau[3];
  sigma[3,4] = corr_ip;
  
  sigma[4,1] = sigma[1,4];
  sigma[4,2] = sigma[2,4];
  sigma[4,3] = sigma[3,4];
  sigma[4,4] = tau[4];  
  
  for(t in 2:T) {
    meanall[t,1] = mu*(1-rho) + rho*GDPR[t-1];
    meanall[t,2] = GDPR[t];
    meanall[t,3] = GDPR[t];
    meanall[t,4] = GDPR[t];    
  }

  for(t in 1:T) {
    dat[t,1] = GDPR[t];
    dat[t,2] = Y[t,1];
    dat[t,3] = Y[t,2];
    dat[t,4] = Y[t,3];    
  }
}

model {
  // priors
  GDPR[1] ~ normal(0.4,1);
  tau ~ cauchy(0,5);
  corr_ei ~ cauchy(0,2);
  corr_ep ~ cauchy(0,2);
  corr_ip ~ cauchy(0,2);
  corr_gi ~ cauchy(0,2);
  corr_gp ~ cauchy(0,2);
  mu ~ normal(0.8, 1);
  rho ~ normal(0.5, 1);

  // likelihood
  for(t in 2:T) {
    target += multi_normal_lpdf(dat[t] | meanall[t], sigma);        
  }
}
