
clear all
close all


%
%  parameters of the model
%


load('../data/timeseries/basiron.mat');
nbtrain=100;
MaxIterBF=500;
nbitermax=1;



level=[-3 0;1 4;5  7];
kernelg='gaussian';

% lambda=[1 0.01  0.005]; % engines
% kerneloptiong=0.1; % Engines
% lambdag=0.1;

lambda=[0.01 2  5]; % 
kerneloptiong=0.05;  % basiron
lambdag=0.1;

%
%   Wavelet Settings
%

kernel='tensorwavkernel';
kerneloption.wname='Symmlet';
kerneloption.pow=10;
kerneloption.par=4;
kerneloption.crossterm='off';
kerneloption.coeffj=1; 
kerneloption.jmin=-3;
kerneloption.jmax=7;
kerneloption.check=0;  

x=x;
yt=y-mean(y);
nbdata=length(y);
for i=1:nbitermax
    fprintf('.');
    
    ind=randperm(nbdata);
    indapp=sort(ind(1:nbtrain));
    indtest=sort(ind(nbtrain+1:end));
    xapp=x(indapp,:);
    yapp=yt(indapp,:);
    xtest=x(indtest,:);
    ytest=yt(indtest,:);
    
    % process all the wavelet needed for the kernel
    [Kinit,KernelInfo]=tensorwavkernel(xapp,xapp,kerneloption);
    kerneloption.vect=KernelInfo.vect;
    % then decomposes
    [K1,Kt]=CreateMultiLevelKernel(xapp,xtest,kerneloption,level);
    C=backfitting(yapp,K1,lambda,MaxIterBF);
    [ypredapp,ysapp]=backfittingval(K1,C);
    [ypred,ys]=backfittingval(Kt,C);
    msel(i)=mean((ytest-ypred).^2)/mean(ytest.^2)

    %
    % Gaussian
    % 

    [c,d]=rncalc(xapp,yapp,kernelg,kerneloptiong,lambdag);
    ypredgapp=rnval(xapp,xapp,kernelg,kerneloptiong,c,d);
    ypredg=rnval(xapp,xtest,kernelg,kerneloptiong,c,d);
    nmseg(i)=mean((ytest-ypredg).^2)/mean(ytest.^2)
    
end;


[xd,ind]=sort([xapp;xtest]);
yaux=[ypredapp;ypred];
ydw=yaux(ind);
yaux=[ypredgapp;ypredg];
ydg=yaux(ind);
ypix=ones(size(xapp))*-400;
figure
h=plot(xtest,ytest,'k',xtest,ypred,'b--',xtest,ypredg,'r:');
set(h,'LineWidth',2)
set(gcf,'color','white');
taillefont=16;
h=ylabel('y');
set(h,'Fonts',taillefont);
h=xlabel('x');
set(h,'Fonts',taillefont);
figure
h=plot(x,yt,'k',xd,ydw,'k--',xd,ydg,'k:',xapp,ypix,'k+');
set(h,'LineWidth',2)
set(gcf,'color','white');
taillefont=16;
h=ylabel('y');
set(h,'Fonts',taillefont);
h=xlabel('x');
set(h,'Fonts',taillefont);