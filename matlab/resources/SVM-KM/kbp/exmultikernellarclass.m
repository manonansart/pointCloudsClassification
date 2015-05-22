%
%  Example of KBP applied on a classification problem
%
%  20/12/05 AR


clear all
close all

n = 500; 

sigma=0.4;
[xapp,yapp,xtest,ytest]=dataset('checkers',n,0,sigma);
[xapp]=normalizemeanstd(xapp);
[xtest,xtest1,xtest2,nn1,nn2]=DataGrid2D([-1:.05:1]*3.5,[-1:.05:1]*3.5);


lambda = 1e-7;  
kernel='gaussian';
kerneloptionvec=[0.1 0.2 0.5 1 2 5];
verbose=0;
LARMKpar{1}.type          = 'nbSV';
LARMKpar{1}.borne         = [50];
Limites=[];



Kapp  = multiplekernel(xapp,kernel,kerneloptionvec);
Ktest  = multiplekernel(xtest,kernel,kerneloptionvec,xapp);



[Kapp,Ktest]=normalizemeanstd(Kapp,Ktest);


[solution, solution_OLS] = LAR(Kapp,yapp, LARMKpar, Limites, lambda, verbose);
ypredapp =LARval(Kapp,solution{1});
ypred=LARval(Ktest,solution{1});
ypred = reshape(ypred,nn1,nn2); 


%--------------- plotting
figure(1); 
clf; 
hold on
[cc,hh]=contour(xtest1,xtest2,ypred,[0 0],'k');
clabel(cc,hh); 
set(hh,'LineWidth',2);
h1=plot(xapp(yapp==1,1),xapp(yapp==1,2),'+r'); 
set(h1,'LineWidth',2);

h2=plot(xapp(yapp==-1,1),xapp(yapp==-1,2),'db'); 
set(h2,'LineWidth',2);
axis([-3.5 3.5 -3 3]); 



