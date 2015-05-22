function [cost] = costlbfixed(StepSigma,DirSigma,Sigma,indsup,Alpsup,w0,C,Xapp,yapp,pow);
%COSTLBFIXED Computes an upper bound on SVM loss 
%  COST = COSTLBFIXED(STEPSIGMA,DIRSIGMA,SIGMA,INDSUP,ALPSUP,W0,C,XAPP,YAPP,POW) 
%  is the upper bound on the SVM loss for updated scale parameters  
%  (SIGMA.^POW + STEPSIGMA * DIRSIGMA)^(1/POW), when the Lagrange multipliers 
%  and the bias parameter are considered unaffected by the SIGMA update 
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

% II.1) distances

Xapp = Xapp.*repmat(Sigma,n,1);
Xsup = Xapp(indsup,:);
Dist = Xsup*Xapp';
dist = 0.5*sum(Xapp.^2,2) ;
Dist = Dist - repmat(dist(indsup),1,n) - repmat(dist',nsup,1) ; % -1/2 (xi-xj)T Sigma^2 (xi-xj)
Dist = exp(Dist) ;

% II.2) slack variables

xi = 1 - yapp.*((Dist'*Alpsup) + w0); 

costxi = C*sum(xi(xi>0));

% II.3) norm of classifier

costw = 0.5 * (Alpsup' * Dist(:,indsup) * Alpsup) ;

% II.4) end: total cost

cost = costw + costxi ;
