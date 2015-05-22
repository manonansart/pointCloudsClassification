% Example of wavelet discriminant basis signal classification
% 
% 

% 20/12/2005

clear all
close all


nbtrain=100;
noise=1;
nf=128;
localisation=100:105;
saut=1;
name={'HeaviSine' 'Doppler'};
nbsigperclass=1000;

%
%   Wavelet Settings
%


Type='Haar';
Par=10;
LARMKpar{1}.type          = 'nbSV';
LARMKpar{1}.borne         = [10];
Limites=[];
lambda=1e-9;
verbose=0;

nbsig=2*nbsigperclass;
nbtest=nbsig-nbtrain;

%
% create 2 class of signal
%

x1=zeros(nbsigperclass,nf);
x2=zeros(nbsigperclass,nf);

for iter=1:nbsigperclass
    [x1(iter,:)]=makesignal(name{1},nf) + noise*randn(1,nf);
  %  [x2(iter,:)]=makesignal(name{2},nf) + noise*randn(1,nf);
     [x2(iter,:)]=x1(iter,:) + noise*randn(1,nf); x2(iter,localisation)= x2(iter,localisation) +saut;
end;



x=[x1;x2];
y=[ones(nbsigperclass,1); -ones(nbsigperclass,1)];
classcode=[1 -1];
[xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x,y,nbtrain, classcode);
  [xapp,xtest] = normalizemeanstd(xapp,xtest);

wtapp=zeros(nbtrain,nf);
qmf = MakeONFilter(Type,Par);
L=0;
for iter=1:nbtrain
    wtapp(iter,:)=fwt_po(xapp(iter,:),L,qmf);
end;
% build multiple kernel
Kapp=[];
for iter=1:nf
    Kapp=[Kapp wtapp(:,iter)*wtapp(:,iter)'];    
end;
% 
wttest=zeros(nbtest,nf);
for iter=1:nbtest
    wttest(iter,:)=fwt_po(xtest(iter,:),L,qmf);
end;


[solution, solution_OLS] = LAR(Kapp,yapp, LARMKpar, Limites, lambda, verbose);
ypredapp =LARval(Kapp(:,solution{1}.indxsup),solution{1});

Ktest=[];
for iter=1:length(solution{1}.indxsup)
    IndexBasis=ceil(solution{1}.indxsup(iter)/nbtrain);
    IndexSignal=rem(solution{1}.indxsup(iter),nbtrain);
    if IndexSignal==0, IndexSignal=nbtrain; end
    Ktest=[Ktest wttest(:,IndexBasis)*wtapp(IndexSignal,IndexBasis)];    
    BasisList(iter,1)=IndexBasis;
    BasisList(iter,2)=floor(log2(IndexBasis-1));
    BasisList(iter,3)=IndexBasis-2^floor(log2(IndexBasis-1))-1;
end;
ypred=LARval(Ktest,solution{1});
mean(sign(ypred)==ytest);
[AUC]=svmroccurve(ypred,ytest)

C=10;
kernel='poly';
kerneloption=1;
verbose=0;span=1;
[xsup,w,b,pos,timeps,alpha,obj]=svmclass(xapp,yapp,C,lambda,kernel,kerneloption,verbose,span);
   [ypredsvm]=svmval(xtest,xsup,w,b,kernel,kerneloption,span);
   [AUCsvm]=svmroccurve(ypredsvm,ytest)
   
 figure
subplot(2,1,1)
h=plot(1:nf,x1(1,:),'b',1:nf,mean(x(y==1,:)),'r')
set(h,'LineWidth',2);
axis([0 nf -8 8]);
h=title('classe 1');
subplot(2,1,2)
h=plot(1:nf,x2(1,:),'b',1:nf,mean(x(y==-1,:)),'r')
set(h,'LineWidth',2);
axis([0 nf -8 8]);
set(gcf,'color','white');
h=title('classe -1');

figure
for iter=1:length(solution{1}.indxsup)
    wta=zeros(1,nf);wta(BasisList(iter,1))=1;
 
   h=plot(1:nf,iwt_po(wta,L,qmf)); hold on
    set(h,'LineWidth',2);
end;
%axis([0 nf -1 1]);
h=plot(1:nf,1+mean(x(y==-1,:))/4,'r')
set(h,'LineWidth',2);
set(gcf,'color','white');
