function [Sigma,Xsup,Alpsup,w0,pos,Time,Crit,SigmaH] = svmfit(Xapp,yapp,Sigma,C,option,pow,verbose)

[n,d] = size(Xapp);

%  SVM fixed parameters 

lambdaReg    = 1e-10; 
kernel       = 'gaussian';
kerneloption = 1 ;
span         = ones(n,1); 

% Display parameters

ngrid = 41;

% initializations

nloop = 0;
nloopmax = 100;
coruntitour = 1;
Time = zeros(1,3);
SigmaH = zeros(nloopmax,d);

t = cputime ; 
[xsup,Alpsup,w0,pos] = svmclass(Xapp.*repmat(Sigma,n,1),yapp,C,lambdaReg,kernel,kerneloption,max(verbose-2,0),span);   
Time(1) = max(cputime - t,0) ;
Xsup = Xapp(pos,:);

Sigmaold  = Sigma ;
Xsupold   = Xsup ;
Alpsupold = Alpsup ;
w0old     = w0;
posold    = pos ;


while coruntitour ; nloop = nloop+1; 
	if verbose > 0,
		if (d==2) & ( ( (verbose == 2) & (nloop==1) ) | (verbose > 2) ),
			[xtest1 xtest2]  = meshgrid(linspace(min(Xapp(:,1)),max(Xapp(:,1)),ngrid),linspace(min(Xapp(:,2)),max(Xapp(:,2)),ngrid));
			ygrid = svmval([xtest1(:) xtest2(:)].*repmat(Sigma,ngrid*ngrid,1),xsup,Alpsup,w0,kernel,kerneloption,[ones(ngrid*ngrid,1)]);
         nsup = length(pos);
			ysup  = yapp(pos);

			figure(nloop);clf;hold on;
			plot(Xapp(yapp==1,1),Xapp(yapp==1,2),'+r');
			plot(Xapp(yapp==-1,1),Xapp(yapp==-1,2),'xb');
			h=plot(Xsup(ysup==1,1),Xsup(ysup==1,2),'or');
			set(h,'LineWidth',2);
			h=plot(Xsup(ysup==-1,1),Xsup(ysup==-1,2),'ob');
			set(h,'LineWidth',2);
			[cc, hh]=contour(xtest1,xtest2,reshape(ygrid,ngrid,ngrid),[-1 -0.5 0 0.5 1]);
			clabel(cc,hh);
			XsupG = Xsup.*repmat(Sigma,nsup,1);
			XsupD = [xtest1(:) xtest2(:)].*repmat(Sigma,ngrid*ngrid,1);
			Dist = XsupG*XsupD';
			Dist = - Dist + repmat(0.5*sum(XsupG.^2,2),1,ngrid*ngrid) + repmat(0.5*sum(XsupD.^2,2)',nsup,1) ; % 1/2 ||Sigma . xi - Sigmaold . xj||^2
			Dist = exp(-Dist) ;
			cc = exp(-Sigma*Sigma'/64);
			for i=1:nsup;
				  [c,h] = contour(xtest1,xtest2,reshape(Dist(i,:),ngrid,ngrid),cc*[1 1],':');
				  set(h,'Color',0.75*[1 1 1])
			end;
			axis('equal')
		 end;
	end;
	t = cputime ; 
	[Sigma] = SigmaupdateConj2(Sigma,pos,Alpsup,w0,C,Xapp,yapp,option,pow,verbose) ;
	Time(2) = Time(2) + max(cputime - t,0) ;
		 
	 if (d==2) & ( ( (verbose == 2) & (nloop==1) ) | (verbose > 2) ),
       nsup = length(pos);
		 figure(nloop);
		 XsupG = Xsup.*repmat(Sigma,nsup,1);
		 XsupD = [xtest1(:) xtest2(:)].*repmat(Sigma,ngrid*ngrid,1);
		 Dist = XsupG*XsupD';
		 Dist = - Dist + repmat(0.5*sum(XsupG.^2,2),1,ngrid*ngrid) + repmat(0.5*sum(XsupD.^2,2)',nsup,1) ; % 1/2 ||Sigma . xi - Sigmaold . xj||^2
		 Dist = exp(-Dist) ;
		 cc = exp(-Sigma*Sigma'/64);fprintf(1,'Distance new %9.2e \n',cc); 
		 for i=1:nsup;
				[c,h] = contour(xtest1,xtest2,reshape(Dist(i,:),ngrid,ngrid),cc*[1 1],'--');
				set(h,'Color',0.75*[1 1 1])
		 end;
		 axis('equal');
		 drawnow;
	 end;
	  
	 if max(abs(Sigma - Sigmaold)./(mean(Sigmaold)+Sigmaold))>1e-3 ,
        t = cputime ; 
        if isempty(xsup) | isnan(w0) | (min(abs(Alpsup))==C)
           [xsup,Alpsup,w0,pos] = svmclass(Xapp.*repmat(Sigma,n,1),yapp,C,lambdaReg,kernel,kerneloption,0,span);
        else
           [xsup,Alpsup,w0,pos] = svmclass(Xapp.*repmat(Sigma,n,1),yapp,C,lambdaReg,kernel,kerneloption,0,span,Alpsup.*yapp(pos),w0,pos);
        end
        Time(3) = Time(3) + max(cputime - t,0) ;
% 	     Time(nloop,2) = max(cputime - t,0) ;
	     Xsup = Xapp(pos,:);
	 end;
	    
	 ind = find(Sigma<1e-6*mean(Sigma));
	 if ~isempty(ind)
        Sigma(ind) = 0 ;
        Sigma      = Sigma*real(abs( (sum(real(abs(Sigma.^pow)))/sum(real(abs(Sigmaold.^pow))))^(1/pow) ));
    end;
       
	 fprintf(1,'Iteration %3d, ',nloop)
	 if size(Xsup,1) == size(Xsupold,1) ,
		 if max(abs(Xsup(:) - Xsupold(:))./(1+abs(Xsupold(:))))<1e-3 ,
			 if max(abs(Alpsup - Alpsupold)./(1+abs(Alpsupold)))<1e-3 ,
				 if max(abs(Sigma - Sigmaold)./(mean(Sigmaold)+Sigmaold))<1e-3 ,
					 coruntitour = 0;
					 fprintf(1,'convergence criteria reached \n');
					 Sigma  = Sigmaold;
					 Xsup   = Xsupold;
					 Alpsup = Alpsupold;
					 w0     = w0old;
				    pos    = posold;
				 else
					fprintf(1,'Sigma varies %9.2e\n', max(abs(Sigma - Sigmaold)./(mean(Sigmaold)+Sigmaold)));
				 end;
			 else
				 fprintf(1,'Alpha varies %9.2e\n',max(abs(Alpsup - Alpsupold)./(1+abs(Alpsupold))))
			 end;
		 else
			 fprintf(1,'support vectors vary \n')
		 end;
	 else
		 fprintf(1,'number of support vectors varies \n')
	 end;
	 if nloop>=nloopmax ,
	 	 coruntitour = 0;
	 	 fprintf(1,'maximum number of iterations reached\n')
	 end;
	 Sigmaold  = Sigma ;
    SigmaH(nloop,:) = Sigma;
	 Xsupold   = Xsup ;
	 Alpsupold = Alpsup ;
	 w0old     = w0;
    posold    = pos ;
end;

if (d==2) & (verbose > 1) ,
	figure(nloop);
	[xtest1 xtest2]  = meshgrid(linspace(min(Xapp(:,1)),max(Xapp(:,1)),ngrid),linspace(min(Xapp(:,2)),max(Xapp(:,2)),ngrid));
	ygrid = svmval([xtest1(:) xtest2(:)].*repmat(Sigma,ngrid*ngrid,1),xsup,Alpsup,w0,kernel,kerneloption,[ones(ngrid*ngrid,1)]);
   nsup = length(pos);
	ysup = yapp(pos);

	figure(nloop);clf;hold on;
	plot(Xapp(yapp==1,1),Xapp(yapp==1,2),'+r');
	plot(Xapp(yapp==-1,1),Xapp(yapp==-1,2),'xb');
	h=plot(Xsup(ysup==1,1),Xsup(ysup==1,2),'or');
	set(h,'LineWidth',2);
	h=plot(Xsup(ysup==-1,1),Xsup(ysup==-1,2),'ob');
	set(h,'LineWidth',2);
	[cc, hh]=contour(xtest1,xtest2,reshape(ygrid,ngrid,ngrid),[-1 -0.5 0 0.5 1]);
	clabel(cc,hh);
	XsupG = Xsup.*repmat(Sigma,nsup,1);
	XsupD = [xtest1(:) xtest2(:)].*repmat(Sigma,ngrid*ngrid,1);
	Dist = XsupG*XsupD';
	Dist = - Dist + repmat(0.5*sum(XsupG.^2,2),1,ngrid*ngrid) + repmat(0.5*sum(XsupD.^2,2)',nsup,1) ; % 1/2 ||Sigma . xi - Sigmaold . xj||^2
	Dist = exp(-Dist) ;
	cc = exp(-Sigma*Sigma'/64);
	for i=1:nsup;
		  [c,h] = contour(xtest1,xtest2,reshape(Dist(i,:),ngrid,ngrid),cc*[1 1],':');
		  set(h,'Color',0.75*[1 1 1])
	end;
	axis('equal')
end;

SigmaH = SigmaH(nloop+1:nloopmax,:) ;

if nargout>=7,
	Crit = costlbfixed(0,zeros(1,d),Sigma,pos,Alpsup,w0,C,Xapp,yapp,1);
end;

