function [grad] = gradwfixed(Sigma,indsup,Alpsup,C,Xapp,yapp,Sigmaold,pow);
%GRADWFIXED Computes the gradient of an upper bound on SVM loss wrt SIGMA^POW  
%  GRAD = GRADWFIXED(SIGMA,INDSUP,ALPSUP,C,XAPP,YAPP,SIGMAOLD,POW) 
%  is the gradient of the upper bound on the SVM loss obtained when the 
%  weight parameters are considered to be unaffected by SIGMA. 
%  
%  SIGMA is the current SIGMA value
%  INDSUP is the (nsup,1) index of current support vectors 
%  ALPSUP is the (nsup,1) vector of non-zero Lagrange multipliers
%  C is the error penalty hyper-aparameter
%  XAPP,YAPP are the learning examples
%  SIGMAOLD is the SIGMA value corresponding to the current weights

%  27/01/03 Y. Grandvalet

% initialization

nsup  = length(indsup);
[n,d] = size(Xapp);

% I) compute slack variables, old Sigma on support, new sigma on data points

indpos = find(yapp== 1);
indneg = find(yapp==-1);
npos = length(indpos) ;
nneg = length(indneg) ;
nmin = min(npos,nneg);

% I.1)  distances

XsupS0  = Xapp(indsup,:).*repmat(Sigmaold,nsup,1);
XappS  = Xapp.*repmat(Sigma,n,1);
Dist   = XsupS0*XappS';
Dist   = Dist - repmat(0.5*sum(XsupS0.^2,2),1,n) - repmat(0.5*sum(XappS.^2,2)',nsup,1) ; % - 1/2 ||Sigma . xi - Sigmaold . xj||^2
Dist   = exp(Dist) ;

% I.2) slacks

xipos = -sort(-(1 - Alpsup'*Dist(:,indpos))); 
xineg = -sort(-(1 + Alpsup'*Dist(:,indneg))); 

xi = sum([xineg(1:nmin) ; xipos(1:nmin)],1);

% II) end: gradient

ind = find(xi>=0) ;

grad = zeros(1,d);
if ~isempty(ind);
   for k=1:d;
      Distk   = repmat(XappS(ind,k)',nsup,1) - repmat(XsupS0(:,k),1,length(ind)) ;
      grad(k) =  Alpsup' * (Distk.*Dist(:,ind)) * (Xapp(ind,k).*yapp(ind)) ;
   end;

   ind = find(Sigma~=0);
   grad(ind) = C*grad(ind).*(1/pow*abs(real(Sigma(ind).^(1-pow))));
end;
   

