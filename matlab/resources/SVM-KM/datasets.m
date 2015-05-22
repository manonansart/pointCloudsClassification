function [xapp,yapp,xtest,ytest,xtest1,xtest2]=dataset(n,nbapp,nbtest,sigma)

% Usage
% [xapp,yapp,xtest,ytest,xtest1,xtest2]=dataset(String,nbapp,nbtest,sigma)
%
%
%  
%    'Mixture'  : 2 classes         
%          First class is a a mixture of gaussian
%          which takes in sandwiches the second class 
%
%           
%  'Gaussian' : is a classical dataset with two gaussianly distributed
%         classes with similar variances but different means
%
%  'Checkers': Checkers
%
%  'MultiClassCheckers'
%          
%  'Clowns' : The Clowns
%
%   'CosExp': sign( 0.95*cos(0.5*(exp(x)-1)))); for x in [0...4]
%       xtest1,xtest2 meshgrid of xtest

xapp=[];
yapp=[];
xtest=[];
ytest=[];
xtest1=[];
xtest2=[];
switch lower(n)
    
    case 'mixture'
        
        %------------------Data Learn Generation -------------------------
        if nargin <4
            sigma=0.4;
        end;
        nbapp=round(nbapp/3);
        x1=sigma*randn(1,nbapp)+0.3;
        x2=sigma*randn(1,nbapp)-0.3;
        x3=sigma*randn(1,nbapp)-1;
        y1=sigma*randn(1,nbapp)+0.5;
        y2=sigma*randn(1,nbapp)-0.5;
        y3=sigma*randn(1,nbapp)-1;
        
        xapp=[x1 x2 x3; y1 y2 y3]';
        yapp=[ones(1,nbapp) -ones(1,nbapp) ones(1,nbapp)]';
        
        
        %----------------Data Test Generation ---------------------------
        nbtest=round(nbtest/3);
        xt=sigma*randn(1,nbtest)+0.3;
        xt2=sigma*randn(1,nbtest)-0.3;
        xt3=sigma*randn(1,nbtest)-1;
        yt=sigma*randn(1,nbtest)+0.5+0.7*xt;
        yt2=sigma*randn(1,nbtest)-0.5;
        yt3=sigma*randn(1,nbtest)-1-0.7*xt;
        xtest=[xt xt2 xt3; yt yt2 yt3]';
        
        
        ytest=[ones(1,nbtest) -ones(1,nbtest) ones(1,nbtest)]';
        return
    case 'gaussian'
        nbapp=round(nbapp/2);
        nbtest=round(nbtest/2);
        if nargin <4
            sigma=0.2;
        end;
        x1=sigma*randn(1,nbapp)+0.3;
        x2=sigma*randn(1,nbapp)-0.3;
        y1=sigma*randn(1,nbapp)+0.5;
        y2=sigma*randn(1,nbapp)-0.5;
        xapp=[x1 x2; y1 y2]';
        yapp=[ones(1,nbapp) -ones(1,nbapp)]';
        x1=sigma*randn(1,nbtest)+0.3;
        x2=sigma*randn(1,nbtest)-0.3;
        y1=sigma*randn(1,nbtest)+0.5;
        y2=sigma*randn(1,nbtest)-0.5;
        xtest=[x1 x2; y1 y2]';
        ytest=[ones(1,nbtest) -ones(1,nbtest)]';
        
    case 'checkers'
        xapp=[];
        yapp=[];
        nb=floor(nbapp/16);
        for i=-2:1;
            for j=-2:1;
                xapp=[xapp; [i+rand(nb,1) j+rand(nb,1)]];
                %(2*rem((i+j+4),2)-1)
                yapp=[yapp; (2*rem((i+j+4),2)-1)*ones(nb,1)];
            end;
        end;
        xtest=[];
        ytest=[];
        nb=floor(nbtest/16);
        for i=-2:1;
            for j=-2:1;
                xtest=[xtest; [i+rand(nb,1) j+rand(nb,1)]];
                %(2*rem((i+j+4),2)-1)
                ytest=[ytest; (2*rem((i+j+4),2)-1)*ones(nb,1)];
            end;
        end;
        %     [xt1,xt2]=meshgrid(linspace(-2,2,sqrt(nbtest)));
        %     [na,nb]=size(xt1);
        %     xtest1=reshape(xt1,na*nb,1);
        %     xtest2=reshape(xt2,na*nb,1);
        %     xtest=[xtest1 xtest2];
        %     ytest=zeros(length(xtest),1);
        %     for i=-2:1
        %         for j=-2:1
        %             pos=find( xtest(:,1)>=i & xtest(:,1)< i+1 & xtest(:,2)>=j & xtest(:,2)< j+1 );
        %             ytest(pos)=(2*rem((i+j+4),2)-1)*ones(length(pos),1);
        %         end;
        %     end;
        %     xtest1=reshape(xt1,na,nb);
        %     xtest2=reshape(xt2,na,nb);
        
        
        
    case 'clowns'
        nbapp=round(nbapp/2);
        nbtest=round(nbtest/2);
        
        x1 = (6*rand(nbapp,1)-3); 
        x2 = x1.^2 + randn(nbapp,1); 
        x0 = sigma*randn(nbapp,2)+(ones(nbapp,1)*[0 6]); 
        xapp = [x0;[x1 x2]]; 
        xapp = (xapp-ones(2*nbapp,1)*mean(xapp))*diag(1./std(xapp)); 
        yapp = [ones(nbapp,1); -ones(nbapp,1)]; 
        
        if nbtest>0
            x1 = (6*rand(nbtest,1)-3); 
            x2 = x1.^2 + randn(nbtest,1); 
            x0 = sigma*randn(nbtest,2)+(ones(nbtest,1)*[0 6]); 
            xtest = [x0;[x1 x2]]; 
            xtest = (xtest-ones(2*nbtest,1)*mean(xtest))*diag(1./std(xtest)); 
            ytest = [ones(nbtest,1); -ones(nbtest,1)]; 
        end;
        
        
        
        
    case 'cosexp'
        
            maxx=5;
            freq=0.7;
            nbtest=floor(sqrt(nbtest));
            [xi]=rand(nbapp,2);
            yapp= sign(cos(0.5*(exp(freq*maxx*xi(:,1))))-(2*xi(:,2)-1));
            xapp = [maxx*xi(:,1) (2*xi(:,2)-1)];
            [xtest1 xtest2]  = meshgrid(linspace(0,maxx,nbtest),linspace(-1,1,nbtest)); 
            nn = length(xtest1); 
            xtest = [reshape(xtest1 ,nn*nn,1) reshape(xtest2 ,nn*nn,1)]; 
            ytest = sign(cos(0.5*(exp(freq*maxx*xtest(:,1))))-(2*xtest(:,2)-1));
            
        

    case 'multiclasscheckers'
        xapp=[];
        yapp=[];
        nb=floor(nbapp/16);
        % keyboard
        for i=-2:1;
            for j=-2:1;
                xapp=[xapp; [i+rand(nb,1) j+rand(nb,1)]];
                
                
                if rem(abs(i),2)==0 & rem(abs(j),2)==0
                    yapp=[yapp; 1*ones(nb,1)];
                elseif rem(abs(i),2)==1 & rem(abs(j),2)==0
                    yapp=[yapp; 2*ones(nb,1)];
                elseif rem(abs(i),2)==0 & rem(abs(j),2)==1
                    yapp=[yapp; 3*ones(nb,1)];
                elseif rem(abs(i),2)==1 & rem(abs(j),2)==1
                    yapp=[yapp; 4*ones(nb,1)];  
                end;
                
            end;
        end;
        xtest=[];
        
        [xt1,xt2]=meshgrid(linspace(-2,2,sqrt(nbtest)));
        [na,nb]=size(xt1);
        xtest1=reshape(xt1,na*nb,1);
        xtest2=reshape(xt2,na*nb,1);
        xtest=[xtest1 xtest2];
        ytest=zeros(length(xtest),1);
        for i=-2:1
            for j=-2:1
                pos=find( xtest(:,1)>=i & xtest(:,1)< i+1 & xtest(:,2)>=j & xtest(:,2)< j+1 );
                %ytest(pos)=(2*rem((i+j+4),2)-1)*ones(length(pos),1);
                if rem(abs(i),2)==0 & rem(abs(j),2)==0
                    ytest(pos)=1*ones(length(pos),1);
                elseif rem(abs(i),2)==1 & rem(abs(j),2)==0
                    ytest(pos)=2*ones(length(pos),1);
                elseif rem(abs(i),2)==0 & rem(abs(j),2)==1
                    ytest(pos)=3*ones(length(pos),1);
                elseif rem(abs(i),2)==1 & rem(abs(j),2)==1
                    ytest(pos)=4*ones(length(pos),1);
                end;
                
                
            end;
        end;
        xtest1=reshape(xt1,na,nb);
        xtest2=reshape(xt2,na,nb);
        
        
    case 'multiclassgaussian'
        
        
        
        mean1=[1 1];
        mean2=[-1 1];
        mean3= [0 -1];
        x1=sigma*randn(nbapp,2)+ones(nbapp,1)*mean1;
        y1= ones(nbapp,1);
        x2=sigma*randn(nbapp,2)+ones(nbapp,1)*mean2;
        y2=2*ones(nbapp,1);
        x3=sigma*randn(nbapp,2)+ones(nbapp,1)*mean3;
        y3=3*ones(nbapp,1);
        Ytarget=[1 2 3];
        xapp=[x1;x2;x3];
        yapp=[y1;y2;y3];
        nbapp=size(xapp,1);
        
        
        %----------------Data Test Generation ---------------------------
        x1=sigma*randn(nbtest,2)+ones(nbtest,1)*mean1;
        y1= ones(nbtest,1);
        x2=sigma*randn(nbtest,2)+ones(nbtest,1)*mean2;
        y2=2*ones(nbtest,1);
        x3=sigma*randn(nbtest,2)+ones(nbtest,1)*mean3;
        y3=3*ones(nbtest,1);
        Ytarget=[1 2 3];
        xtest=[x1;x2;x3];
        ytest=[y1;y2;y3];
        nbapp=size(xapp,1);
        
    case 'westonnonlinear' % dataset used in the Weston FeatSel Paper
        nbapp=round(nbapp/2);
        yapp = [ones(nbapp,1); -ones(nbapp,1)]; 
        
        A=2*(rand(2*nbapp,1)<0.5)-1;
        xapp=randn(2*nbapp,2) + [3*A A.*(yapp*1.875+1.125)];
        xapp = [xapp  20*randn(nbapp*2,8)];
        nbtest=round(nbtest/2);
        ytest = [ones(nbtest,1); -ones(nbtest,1)]; 
        A=2*(rand(2*nbtest,1)<0.5)-1;
        xtest=randn(2*nbtest,2) + [3*A A.*(ytest*1.875+1.125)];
        xtest = [xtest  20*randn(nbtest*2,8)];
        
        
    case 'gaussianmultires'
        n=floor(nbapp/12);
        
        xm1=1;
        % The Big One
        mean1=[-xm1 -xm1];
        mean2=[xm1 xm1];
        sigma1=0.8;
        sigma2=0.3;
        xapp1=ones(2*n,1)* mean1+sigma1*randn(2*n,2);
        xapp2=ones(2*n,1)* mean2+sigma1*randn(2*n,2);
        yapp1=ones(n,1);
        yapp2=-ones(n,1);
        % The small ones
        mean1=[-2*xm1 0];
        mean2=[0 -2*xm1];
        mean3=[-2*xm1 -2*xm1];
        mean4=[-2*xm1 -2*xm1];
        
        xapp11=ones(n,1)* mean1+sigma2*randn(n,2);
        xapp12=ones(n,1)* mean2+sigma2*randn(n,2);
        xapp13=ones(n,1)* mean3+sigma2*randn(n,2);
        xapp14=ones(n,1)* mean4+sigma2*randn(n,2);
        % The small ones
        mean1=[2*xm1 0];
        mean2=[0 2*xm1];
        mean3=[2*xm1 2*xm1];
        mean4=[2*xm1 2*xm1];
        
        xapp21=ones(n,1)* mean1+sigma2*randn(n,2);
        xapp22=ones(n,1)* mean2+sigma2*randn(n,2);
        xapp23=ones(n,1)* mean3+sigma2*randn(n,2);
        xapp24=ones(n,1)* mean4+sigma2*randn(n,2);
        xapp=[xapp1;xapp21;xapp22;xapp23;xapp24;xapp2;xapp11;xapp12;xapp13;xapp14];
        yapp=[ones(6*n,1);-ones(6*n,1)];
        
        
        %        mean1=[-1 1];
        %     mean2=[-0.25 0.25];
        %     mean3=[0.25 -0.25];
        %     mean4=[+1 -1];
        %     sigma3=sigma2;
        %     sigma4=sigma1;
        %     xapp1=ones(n,1)* mean1+sigma1*randn(n,2);
        %     xapp2=ones(n,1)* mean2+sigma2*randn(n,2);
        %     xapp3=ones(n,1)* mean3+sigma3*randn(n,2);
        %     xapp4=ones(n,1)* mean4+sigma4*randn(n,2);
        %     yapp1=-ones(n,1);
        %     yapp2=ones(n,1);
        %     yapp3=-ones(n,1);
        %     yapp4=ones(n,1);
        %         xapp=[xapp; xapp1;xapp2;xapp3;xapp4];
        %     yapp=[yapp;yapp1;yapp2;yapp3;yapp4];
        %     
        
        
        xtest=[];
        ytest=[];
        
    case 'carpette'
        xapp=2*rand(nbapp,2)-1;
        xtest=2*rand(nbtest,2)-1;
        xapp(find(sum(xapp.^2,2)>0.64),:)=[];
        xtest(find(sum(xtest.^2,2)>0.64),:)=[];
        
        yapp = carpette(xapp);
        ytest = carpette(xtest); 
        
    case 'gaussianmultires2'
        
        n=floor(nbapp/10);
        sigma=1.5;
        ind2=[];
        ind1=[];
        xappaux1=sigma*randn(10*nbapp,2);
        rayon2=sum(xappaux1.^2,2);
        ind=find(rayon2 > 0.14 & rayon2 < 0.64); 
        xappaux1=xappaux1(ind(1:n),:); 
        
        
        xappaux2=sigma*randn(10*nbapp,2);
        rayon2=sum(xappaux2.^2,2);
        ind=find(rayon2 < 0.14); 
        xappaux2=xappaux2(ind(1:n),:);    
        
        xappaux3=sigma*randn(10*nbapp,2);
        rayon2=sum(xappaux3.^2,2);
        ind=find( rayon2 > 0.64); 
        xappaux3=xappaux3(ind(1:4*n),:);     
        xapp=[xappaux2;xappaux3;xappaux1];
        yapp=[ones(5*n,1);-ones(n,1)];
        
        
        mean1=[-4 0];
        mean2=[-0 4];
        mean3=[4 0];
        mean4=[0 -4];
        sigma1=0.5;
        xapp1=ones(n,1)* mean1+sigma1*randn(n,2);
        xapp2=ones(n,1)* mean2+sigma1*randn(n,2);
        xapp3=ones(n,1)* mean3+sigma1*randn(n,2);
        xapp4=ones(n,1)* mean4+sigma1*randn(n,2);
        xapp=[xapp;xapp1;xapp2;xapp3;xapp4];
        yapp=[yapp;-ones(4*n,1)];
        
        
        
end;
% ======================================
function color = carpette(x);

color = ones(length(x),1);
r2 = x(:,1).*x(:,1) +x(:,2).*x(:,2) ;
r2(find(r2<0.04)) = 10;
c2 = find(r2<0.25);
color(c2(find(x(c2,1)>=0))) = -1;
r2(c2) = 10;
c2 = find(r2<0.64);
color(c2(find(x(c2,1)<0))) = -1;
r2(c2) = 10;
c2 = find(r2<10);
%color(c2(find(x(c2,2)<0))) = 3;
%color(c2(find(x(c2,2)>=0))) = 4;

