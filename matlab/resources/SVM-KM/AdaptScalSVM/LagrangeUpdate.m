function [deltAlp,Stepmax] = LagrangeUpdate(DirSigma,Sigma,Xsup,Alpsup,C,pow);
%LAGRANGEUPDATE Compute gradient and maximal change in Lagrange multipliers
%       [DELTALP,STEPMAX] = LAGRANGEUPDATE(DIRSIGMA,SIGMA,XSUP,ALPSUP,C,POW)
%       provides the descent direction DELTALP and the maximal update STEPMAX
%       for Lagrange multipliers based on 
%       - the secent direction DIRSIGMA for Sigma;
%       - the current Sigma value SIGMA;
%       - the SVM parameters XSUP,ALPSUP;
%       - the SVM hyper-parameters C and POW

%       27/01/03 Y. Grandvalet

% initialization

[nsup d] = size(Xsup);
deltAlp  = zeros(nsup,1);
Stepmax  = realmax;

% support vectors of the first category

indsup1 = find(abs(Alpsup)<C);
nsup1   = length(indsup1);

if nsup1>0,
   % 1) compute distances
   
   Xsup = Xsup.*repmat(Sigma,nsup,1);
   Dist = Xsup(indsup1,:)*Xsup';
   dist = 0.5*sum(Xsup.^2,2) ;
   Dist = Dist - repmat(dist(indsup1),1,nsup) - repmat(dist',nsup1,1) ; % -1/2 (xi-xj)T Sigma^2 (xi-xj)
   Dist = exp(Dist) ;

   % 2) solve linear systems (d nsup1+1 Ax_k = b_k systems (A symmetric (not def. pos.): factorize ? ))

   A = [ [Dist(:,indsup1) ones(nsup1,1)] ; [ones(1,nsup1) 0] ];

   b = zeros(nsup1+1,d);
   for k=1:d;
      Distk  = Xsup(indsup1,k)*Xsup(:,k)';
      distk  = Xsup(:,k).^2;
      Distk  = -2*Distk + repmat(distk(indsup1),1,nsup) + repmat(dist',nsup1,1) ; % (xik-xjk)^2
      b(1:nsup1,k) = (Distk.*Dist) * Alpsup * Sigma(k) ;
   end;
   ind = find(Sigma~=0);
   b(1:nsup1,ind) = b(1:nsup1,ind).*repmat((1/pow*abs(real(Sigma(ind).^(1-pow)))),nsup1,1);

   x = (A+1e-10*eye(nsup1+1))\b;

   gradAlp          = x(1:nsup1,:);
   deltAlp(indsup1) = gradAlp*DirSigma';

   % 3) compute maximal Alpsup change

   % 3.1) no sign change

   ind = find((deltAlp(indsup1).*Alpsup(indsup1))<0);
   if ~isempty(ind)
      Stepmax = min(-Alpsup(indsup1(ind))./deltAlp(indsup1(ind))) ;
   end;
    
   % 3.2) max(Alpsup) = +-C;

   ind = find((deltAlp(indsup1).*Alpsup(indsup1))>0);
   if ~isempty(ind)
      Stepmax = min(Stepmax,min((C-abs(Alpsup(indsup1(ind))))./abs(deltAlp(indsup1(ind))))) ;
   end;
end;
