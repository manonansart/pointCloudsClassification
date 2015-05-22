function cost = costwfixed(StepSigma,DirSigma,Sigma,indsup,Alpsup,C,Sigmaold,Xapp,yapp,pow);
%COSTWFIXED Computes an upper bound on SVM loss 
%  COST = COSTWFIXED(STEPSIGMA,DIRSIGMA,SIGMA,INDSUP,ALPSUP,C,XAPP,YAPP,POW) 
%  is the upper bound on the SVM loss for updated scale parameters  
%  (SIGMA.^POW + STEPSIGMA * DIRSIGMA)^(1/POW), when the weight
%  parameters are considered unaffected by the SIGMA update 
%  
%  STEPSIGMA is the stepsize of SIGMA update
%  DIRSIGMA is the direction of SIGMA update
%  SIGMA is the current SIGMA value
%  INDSUP is the (nsup,1) index of current support vectors 
%  ALPSUP is the (nsup,1) vector of non-zero Lagrange multipliers
%  C is the error penalty hyper-aparameter
%  XAPP,YAPP are the learning examples

%  27/01/03 Y. Grandvalet

% initialization

nsup = length(indsup);

% I) update bandwidths

SigmaP = Sigma.^pow + StepSigma * DirSigma;
Sigma  = abs(real(SigmaP.^(1/pow)));

% II) compute cost 

% II.1) slack variables, old Sigma old on support, new Sigma on data points

indpos = find(yapp== 1);
indneg = find(yapp==-1);
npos = length(indpos) ;
nneg = length(indneg) ;
n    = npos + nneg ;
nmin = min(npos,nneg);

Xsup = Xapp(indsup,:).*repmat(Sigmaold,nsup,1);
Xapp = Xapp.*repmat(Sigma,n,1);
Dist = Xsup*Xapp';
Dist = Dist - repmat(0.5*sum(Xsup.^2,2),1,n) - repmat(0.5*sum(Xapp.^2,2)',nsup,1) ; % - 1/2 ||Sigma . xi - Sigmaold . xj||^2
Dist = exp(Dist) ;

xipos = -sort(-(1 - Alpsup'*Dist(:,indpos))); 
xineg = -sort(-(1 + Alpsup'*Dist(:,indneg))); 

xi = sum([xineg(1:nmin) ; xipos(1:nmin)],1);

costxi = C*sum(xi(xi>0));

% II.2) norm of classifier, w fixed (Sigma old)

Dist = Xsup*Xsup';
dist = 0.5*diag(Dist);
Dist = Dist - repmat(dist,1,nsup) - repmat(dist',nsup,1) ; % -1/2 (xi-xj)T Sigmaold^2 (xi-xj)
Dist = exp(Dist) ;

costw = 0.5 * (Alpsup' * Dist * Alpsup) ;

% II.3) end: total cost

cost = costw + costxi ;
