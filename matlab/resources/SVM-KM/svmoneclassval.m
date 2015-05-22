function ypred= svmoneclassval(x,xsup,alpha,rho,kernel,kerneloption);


K=normalizekernel(x,kernel,kerneloption,xsup);
ypred=K*alpha+rho;