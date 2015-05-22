function [phi,psi,xval]=waveletfunction(Family,par,nbpoint);

% Usage
%
% [phi,psi,xval]=waveletfunction(Family,par,nbpoint);
%
%
%


n = 2^16;
J = log2(n);
j = 10;
k = 2^(J-j-1);

Gender='Father';
mphi = MakeWavelet(J-j,k,Family,par,Gender,n).*2^(j/2);

supp = find(abs(mphi) >1e-15);
mins1 = min(supp);
maxs1 = max(supp);
i = (((1:n)-n/2)./2^j)+1;
phi= mphi(mins1:maxs1);



Gender='Mother';
mpsi = MakeWavelet(J-j,k,Family,par,Gender,n).*2^(j/2);
mpsi=mpsi;

supp = find(abs(mpsi) >1e-15);
mins2 = min(supp);
maxs2 = max(supp);
psi= mpsi(mins2:maxs2);
xval= i(mins1:maxs1)- i(mins1);


dif=length(psi)-length(phi);
if dif>0;
    maxs2=maxs2-dif;
    psi= mpsi(mins2:maxs2);
    xval= i(mins1:maxs1)- i(mins1);

elseif   dif<0
    maxs1=maxs1+dif;
    phi= mphi(mins1:maxs1);
    xval= i(mins2:maxs2)- i(mins2); 
end;

longueur=length(xval);
pas=longueur/nbpoint;

warning off
xval=xval(1:pas:longueur-pas);
phi=phi(1:pas:longueur-pas);
psi=psi(1:pas:longueur-pas);
warning on
    xval=[xval round(max(xval))];
    phi=[phi 0];
    psi=[psi 0];
 