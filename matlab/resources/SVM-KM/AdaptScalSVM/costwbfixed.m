function [cost] = costwbfixed(StepSigma,DirSigma,Sigma,indsup,Alpsup,w0,C,Sigmaold,Xapp,yapp,pow);
%COSTWBFIXED Computes an upper bound on SVM loss 
%  COST = COSTWBFIXED(STEPSIGMA,DIRSIGMA,SIGMA,INDSUP,ALPSUP,W0,C,XAPP,YAPP,POW) 
%  is the upper bound on the SVM loss for updated scale parameters  
%  (SIGMA.^POW + STEPSIGMA * DIRSIGMA)^(1/POW), when the weight and bias 
%  parameters are considered unaffected by the SIGMA update 
%  
%  STEPSIGMA is the stepsize of SIGMA update
%  DIRSIGMA is the direction of SIGMA update
%  SIGMA is the current SIGMA value
%  INDSUP is the (nsup,1) index of current support vectors 
%  ALPSUP is the (nsup,1) vector of non-zero Lagrange multipliers
%  W0 is the bias parameter
%  C is the error penalty hyper-aparameter
%  XAPP,YAPP are the learning examples

%  27/01/03 Y. Grandvalet

% initialization

nsup = length(indsup);
n    = size(Xapp,1);

% I) update bandwidths

SigmaP = Sigma.^pow + StepSigma * DirSigma;
Sigma  = abs(real(SigmaP.^(1/pow)));

% II) compute cost 

% II.1) slack variables, old Sigma old on support, new Sigma on data points

Xsup = Xapp(indsup,:).*repmat(Sigmaold,nsup,1);
Xapp = Xapp.*repmat(Sigma,n,1);
Dist = Xsup*Xapp';
Dist = Dist - repmat(0.5*sum(Xsup.^2,2),1,n) - repmat(0.5*sum(Xapp.^2,2)',nsup,1) ; % - 1/2 ||Sigma . xi - Sigmaold . xj||^2
Dist = exp(Dist) ;

xi = 1 - yapp.*((Dist'*Alpsup) + w0); 

costxi = C*sum(xi(xi>0));

% II.2) norm of classifier, w fixed (Sigma old)

Dist = Xsup*Xsup';
dist = 0.5*diag(Dist);
Dist = Dist - repmat(dist,1,nsup) - repmat(dist',nsup,1) ; % -1/2 (xi-xj)T Sigmaold^2 (xi-xj)
Dist = exp(Dist) ;

costw = 0.5 * (Alpsup' * Dist * Alpsup) ;

% II.3) end: total cost

cost = costw + costxi ;
