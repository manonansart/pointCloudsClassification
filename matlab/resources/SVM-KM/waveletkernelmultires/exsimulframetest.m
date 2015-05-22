
clear all
close all


%
%  parameters of the model
%



nbtrain=100;
MaxIterBF=500;
lambda=[1 1 5];
level=[0 1;2 3;4  5];

noise=0.1;
nf=100;
name='Doppler';
[xapp,yapp,xtest,ytest]=makesignalnus(name,nf,nbtrain,noise);

%
%   Wavelet Settings
%

kernel='tensorwavkernel';
kerneloption.wname='Symmlet';
kerneloption.pow=10;
kerneloption.par=6;
kerneloption.crossterm='off';
kerneloption.coeffj=1; 
kerneloption.jmin=-3;
kerneloption.jmax=3;
kerneloption.check=1;  






[K1,Kt]=CreateMultiLevelKernel(xapp,xtest,kerneloption,level);
C=backfitting(yapp,K1,lambda,MaxIterBF);
[ypredapp,ysapp]=backfittingval(K1,C);
[ypred,ys]=backfittingval(Kt,C);
msel=mean((ytest-ypred).^2)
figure
plot(xtest,ytest,'b',xtest,ypred,'r');
% hold on
% plot(xapp,yapp,'b+',xapp,yapp,'b',x,yt,'g',xtest,ytest,'kx')