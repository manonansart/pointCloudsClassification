function [x,y,xf,yf]=makesignalnus(name,nf,n,noise);

% Usage [x,y,xf,yf]=makesignalnus(name,nf,n,noise);
%
% name String
% nf length of original fucntion
% n  length of noisy non uniformly sampled signal
% noise gaussian noise variance
%
% non uniform sampling of Donoho's signal
%
xf=linspace(0,1,nf)';
yf=makesignal(name,nf)';
x=sort(floor(nf*rand(n,1))+1);

y=yf(x);
x=x./nf;
y=y+noise*randn(size(x));

