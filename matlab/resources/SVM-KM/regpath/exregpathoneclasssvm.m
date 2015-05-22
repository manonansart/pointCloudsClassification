
clear all
close all
clc
n= 100;
sigma=0.3;
[xapp,yapp]=dataset('gaussian',n,0,sigma);
xapp=xapp(find(yapp==-1),:);
nbtrain=size(xapp,1);
[xtest1 xtest2]  = meshgrid([-1:.01:1]*3.5,[-1:0.01:1]*3); 
nn = length(xtest1); 
xtest = [reshape(xtest1 ,nn*nn,1) reshape(xtest2 ,nn*nn,1)]; 

kernel='gaussian';
kerneloption=1;
verbose=1;

%--------------------------------------------------------
%           One-Class SVM Reg Path
%--------------------------------------------------------

[alphamat,alpha0vec,lambdavec,event]=regpathsvmoneclass(xapp,kernel,kerneloption,verbose);
Nbapp=size(xapp,1);
N=30;
nuvec=fliplr(linspace(min(lambdavec)/Nbapp,max(lambdavec)/Nbapp,N));
nuvec=fliplr(linspace(0.05,0.95,N));
[alphamat,alpha0vec,lambdavec,nuvec]=TransformPathFromNu(alphamat,alpha0vec,lambdavec,nuvec,Nbapp);


%--------------------------------------------------------
%           Plotting the Reg Path results
%--------------------------------------------------------
xsup=xapp;
for i=1:8:length(lambdavec);
    
    w=alphamat(:,i);
    w0=-alpha0vec(:,i);
    ypred = svmval(xtest,xsup,w,w0,kernel,kerneloption,1)/lambdavec(i);
    ypred=reshape(ypred,nn,nn);
    figure(1);  
    hold on
    [cc,hh]=contour(xtest1,xtest2,ypred,[0 0],'k');
    set(hh,'LineWidth',2);
    h1=plot(xapp(:,1),xapp(:,2),'+r'); 
    axis([-1.5 0.5 -1.5 1]);
    
  %  pause
end;

figure
plot(lambdavec,alphamat','LineWidth',2)
axis([min(lambdavec) max(lambdavec) 0 1])
%set(gca,'Xscale','log');

set(gcf,'color','white');
xlabel('\lambda','Fonts',16);
ylabel('\alpha','Fonts',16)


%----------------------------------------------------
%                   Movie Section
%----------------------------------------------------

movalpha = avifile('exampleoneclassalpha.avi','FPS',2)
for i=1:length(lambdavec)-2
    figure(2);clf;
    plot(1./lambdavec(1:i),alphamat(:,1:i)','Linewidth',2);
    xlabel('1/(\nu . n)');
    set(gca,'Xscale','log');
    set(gcf,'color','white');
    axis([0 0.3 0 1]); 
    F = getframe(gca);
    movalpha = addframe(movalpha,F);
    
end
movalpha=close(movalpha);

mov = avifile('exampleoneclass.avi','FPS',2)
xsup=xapp;
for i=1:length(lambdavec);
    
    w=alphamat(:,i);
    w0=-alpha0vec(:,i);
    ypred = svmval(xtest,xsup,w,w0,kernel,kerneloption,1)/lambdavec(i);
    ypred=reshape(ypred,nn,nn);
    figure(1);  clf
    %subplot(2,1,1)
    %contourf(xtest1,xtest2,ypred,50);shading flat;
    
    [cc,hh]=contour(xtest1,xtest2,ypred,[0 0],'k');hold
    % clabel(cc,hh)
    set(hh,'LineWidth',2);
    set(gcf,'color','white');
    h1=plot(xapp(:,1),xapp(:,2),'+r'); 
    axis([-3 3 -3 3])
    F = getframe(gca);
    mov = addframe(mov,F);
    
end;

mov = close(mov);