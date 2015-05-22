function [grad] = gradlfixed(Sigma,indsup,Alpsup,C,Xapp,yapp,pow);
%GRADLFIXED Computes the gradient of an upper bound on SVM loss wrt SIGMA^POW  
%  GRAD = GRADLFIXED(SIGMA,INDSUP,ALPSUP,C,XAPP,YAPP,POW) 
%  is the gradient of the upper bound on the SVM loss obtained when the 
%  Lagrange multipliers are considered to be unaffected by SIGMA. 
%  
%  SIGMA is the current SIGMA value
%  INDSUP is the (nsup,1) index of current support vectors 
%  ALPSUP is the (nsup,1) vector of non-zero Lagrange multipliers
%  C is the error penalty hyper-aparameter
%  XAPP,YAPP are the learning examples

%  27/01/03 Y. Grandvalet

% initialization

nsup  = length(indsup);
[n,d] = size(Xapp);

% I) compute distances

XappS = Xapp.*repmat(Sigma,n,1);
XsupS = XappS(indsup,:);
Dist  = XsupS*XappS';
dist  = 0.5*sum(XappS.^2,2) ;
Dist  = Dist - repmat(dist(indsup),1,n) - repmat(dist',nsup,1) ; % -1/2 (xi-xj)T Sigma^2 (xi-xj)
Dist  = exp(Dist) ;

% II) compute gradient of error part

% II.1) slacks

indpos = find(yapp== 1);
indneg = find(yapp==-1);
npos = length(indpos) ;
nneg = length(indneg) ;
nmin = min(npos,nneg);

xipos = -sort(-(1 - Alpsup'*Dist(:,indpos))); 
xineg = -sort(-(1 + Alpsup'*Dist(:,indneg))); 
xi = sum([xineg(1:nmin) ; xipos(1:nmin)],1);

% II.2) gradient of error part

ind = find(xi>=0) ;

grad = zeros(1,d);
if ~isempty(ind);
   for k=1:d;
      Distk   = repmat(XappS(ind,k)',nsup,1) - repmat(XsupS(:,k),1,length(ind)) ;
      grad(k) =  Alpsup' * (Distk.*Dist(:,ind)) * (Xapp(ind,k).*yapp(ind)) ;
   end;
end;
grad = C*grad ;

% III) add gradient of norm part

Xsup  = Xapp(indsup,:);
for k=1:d;
   Distk = Xsup(:,k)*Xsup(:,k)';
   distk = diag(Distk);
   Distk = -2*Distk + repmat(distk,1,nsup) + repmat(distk',nsup,1) ; % (xik-xjk)^2
   grad(k) = grad(k) + (Alpsup' * (Distk.*Dist(:,indsup)) * Alpsup) * Sigma(k) ;
end;

% III) modify according to power

ind = find(Sigma~=0);
grad(ind) = grad(ind).*(1/pow*abs(real(Sigma(ind).^(1-pow))));

