function [K,dist]= kernelset(A,kernelsetoption,oneclassoption,B)

% USAGE
% 
% K= kernelset(A,B,kerneloption,oneclassoption)
%
%
% A cell de matrice
% B cell de matrice
%
%



verbose=0;
samedata=0;
if nargin<4
    B=A;
    samedata=1;
end;
nbA=length(A);
nbB=length(B);
for i=1:nbA
    [xsup{i},alpha{i},rho{i},pos,Ksup{i}]=svmoneclass(A{i},oneclassoption.kernel,oneclassoption.kerneloption,oneclassoption.nu,verbose);
end;
if ~samedata
    for j=1:nbB
         [xsup2{j},alpha2{j},rho2{j},pos2,Ksup2{j}]=svmoneclass(B{j}, oneclassoption.kernel,oneclassoption.kerneloption,oneclassoption.nu,verbose);
     end;    
     else
         xsup2=xsup;
         alpha2=alpha;
         rho2=rho;
         Ksup2=Ksup;
     end;
for i=1:nbA
    
    for j=1:nbB  
        
        switch kernelsetoption.method
            case 'des'
                %%   calcul de l'angle entre les centres
                %%  c1,c2
                % formulation à la desobry, pas def. pos
                num=alpha{i}'*svmoneclassval(xsup{i},xsup2{j},alpha2{j},0,oneclassoption.kernel,oneclassoption.kerneloption);
                den=sqrt(alpha{i}'*Ksup{i}*alpha{i})*sqrt(alpha2{j}'*Ksup2{j}*alpha2{j});
                arcc1c2=acos(num/den);
                % angle
                % c2p2
                arcc2p2=acos(-rho2{j}/sqrt(alpha2{j}'*Ksup2{j}*alpha2{j}));
                % angle
                % c1p1
                arcc1p1=acos(-rho{i}/sqrt(alpha{i}'*Ksup{i}*alpha{i}));  
                dist(i,j)= arcc1c2^2/(arcc2p2^2+arcc1p1^2);
                K(i,j)=exp(-dist(i,j)./2/kernelsetoption.kerneloption^2);
                
            case 'norm'
                
                % ps entre hyperplan
                
                pshyper=alpha{i}'*svmoneclassval(xsup{i},xsup2{j},alpha2{j},0,oneclassoption.kernel,oneclassoption.kerneloption);
                normhp1=sqrt(alpha{i}'*Ksup{i}*alpha{i});
                normhp2=sqrt(alpha2{j}'*Ksup2{j}*alpha2{j});
                dist(i,j)= normhp1^2+normhp2^2-2*pshyper;
                K(i,j)=exp(-dist(i,j).^2/2/kernelsetoption.kerneloption^2)*exp(-(rho{i}-rho2{j}).^2/2/kernelsetoption.kerneloption^2);
                
                
    
            end;
    end;
end;


