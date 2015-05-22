%
%  this is a script for testing and looking into wavelet kernel
%  and different decompositions. 
%  
%
clear all
close all

% this is the data
N=50;
x=linspace(0,1,N)';

% these are the parameters of the wavelet kernel 
kernel='tensorwavkernel';
kerneloption.wname='Symmlet';
kerneloption.pow=14;
kerneloption.par=4;
kerneloption.crossterm='off';
kerneloption.coeffj=1; 
kerneloption.jmin=0;
kerneloption.jmax=3;
kerneloption.check=0; 

% testing scaling span : on and off

% this is the full wavelet kernel with scaling function
kerneloption.father='on';
[K,KernelInfo]=tensorwavkernel(x,x,kerneloption);
kerneloption.vect=KernelInfo.vect;

%--------------------------------------------------------------------------
% without scaling function
kerneloption.father='off';
[K2,KernelInfo2]=tensorwavkernel(x,x,kerneloption);
mesh(K-K2);
title('Kernel Difference with and  without scaling function');


%--------------------------------------------------------------------------
% testing scaling span : on multiscale
%
kerneloption.father='on';
level=[0 1;2 3];
[K3,Kt]=CreateMultiLevelKernel(x,[],kerneloption,level);
figure
mesh(K-sum(K3,3));
title('Kernel Difference on all span and decomposition');

%--------------------------------------------------------------------------
% another example of level decomposition
%
kerneloption.father='on';
level=[0 2;3 3];
[K4,Kt]=CreateMultiLevelKernel(x,[],kerneloption,level);
figure
mesh(K3(:,:,1)-K4(:,:,1))
title('Kernel Difference on a level  of two different decomposition');

%--------------------------------------------------------------------------
% suppress  some level and recalculate the kernel. then compares the new kernel
% and the last decomposition of a multilevel kernel
%

% this is important so that only the corresponding levels are taken into account
kerneloption=rmfield(kerneloption,'vect'); 
kerneloption.father='off';
kerneloption.jmin=2;
[K5,KernelInfo2]=tensorwavkernel(x,x,kerneloption);
figure
mesh(K5-K3(:,:,2));
title('Kernel Difference on last decomposition');