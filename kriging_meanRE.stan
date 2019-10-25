
data {
  
  int N; // locations
  vector[N] dist[N]; //distances
  int Nobs;
  vector[Nobs] yobs; //observed values
  int Nmiss; // number of missing locations
  
  matrix[N,2] Xmat;
  
}


parameters {

  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  
  vector[Nmiss] ymiss;
  
  vector[2] beta;
  
 }


model {
  
 matrix[N, N] cov= cov_exp_quad(dist, alpha, rho)+ diag_matrix(rep_vector(square(sigma), N)) ;
 matrix[N, N] L_cov= cholesky_decompose(cov);  
 vector[Nobs + Nmiss] ys;

  for(n in 1:Nobs){
    ys[n] = yobs[n];
  }
  
  for(n in (Nobs + 1):(Nobs + Nmiss)){
    ys[n] = ymiss[n - Nobs];
  }
  
 //priors
  rho ~ normal(0, 2);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
 
  beta[1] ~ normal(10, 1);
  beta[2] ~ normal(0, 1);
 
 //model
 ys ~ multi_normal_cholesky(Xmat*beta, L_cov); 

}



