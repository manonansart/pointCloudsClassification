% example of Kernel Basis Pursuit and LAR regression on
% real data and automatic selection of kernel parameters.
%
%

% Paper :
% V. Guigue, A. Rakotomamonjy, S. Canu, Kernel Basis Pursuit. European Conference on Machine Learning, Porto, 2005.

clear all;
close all;


load pyrim.mat;




nb_CV  = 5;
fact_ech=3;

kernel        = 'gaussian';
% borneLAR{1}.borne  = 10;
% borneLAR{1}.type = 'nbSV';

borneLAR{1}.borne  = 2;
borneLAR{1}.type = 'trapscale';
borneLAR{1}.indTS=1;

lambda        = 1e-10;
verbose       = 0;
Limites=[];

for k = 1:nb_CV    
    
    
    
    [xapp,yapp,xtest,ytest]=nfcvreg(x,y,nb_CV,k);
    nbapp = size(xapp,1);
    
    
    [sigma_trap]   = CalcTrapScale(xapp);
    kerneloption  = [sigma_trap sigma_trap*fact_ech sigma_trap*fact_ech^2 sigma_trap*fact_ech^3 sigma_trap*fact_ech^4];
    


    Kapp  = multiplekernel(xapp,kernel,kerneloption);
    [Kapp,meanK,stdK]=normalizekernelLAR(Kapp);
    [solution, solution_OLS] = LAR(Kapp,yapp, borneLAR, Limites, lambda, verbose);

    
    Ktest  = multiplekernel(xtest,kernel,kerneloption,xapp,solution{1});
    [Ktest]=normalizekernelLAR(Ktest,meanK,stdK,solution_OLS{1});
    ypredtest=LARval(Ktest,solution_OLS{1});    
    MSEtest(k) = (ypredtest-ytest)'*(ypredtest-ytest)/length(ytest);
    
    
end

MSEtest

[val,ind_min] = min(MSEtest);
[val,ind_max] = max(MSEtest);

indtokeep = setdiff(1:length(MSEtest),[ind_min ind_max]);
fprintf('mean MSEtest : %f std MSEtest : %f\n',mean(MSEtest),std(MSEtest)); 
fprintf('ROBUST mean MSEtest : %f std MSEtest : %f\n',mean(MSEtest(indtokeep)),std(MSEtest(indtokeep))); 