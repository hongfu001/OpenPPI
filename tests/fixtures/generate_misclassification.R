# Regenerate with source R files from yiqunchen/debias-llm-as-a-judge,
# commit 632a89565883be25d94d38b7cd3ca78d62a491f4.
# Usage: Rscript tests/fixtures/generate_misclassification.R /path/to/repo
root <- commandArgs(trailingOnly=TRUE)[1]
source(file.path(root, 'R/sim_estimators.R'))
source(file.path(root, 'R/joint_mle.R'))
options(digits=17)
for (v in list(c(40,10,15,35,90,200), c(18,12,7,23,35,100), c(70,5,10,15,120,150))) {
  a<-v[1]; b<-v[2]; c<-v[3]; d<-v[4]; u<-v[5]; n<-v[6]
  y<-rep(c(0,0,1,1),c(a,b,c,d)); f<-rep(c(0,1,0,1),c(a,b,c,d))
  g<-c(rep(1,u),rep(0,n-u))
  print(rg_point_and_ci(mean(g),a/(a+b),d/(c+d),n,a+b,c+d,alpha=.05))
  out<-mle_point_and_ci(y,f,g,alpha=.05)
  print(unlist(out[c('theta','var','ci_lower','ci_upper','q0_hat','q1_hat')]))
}
