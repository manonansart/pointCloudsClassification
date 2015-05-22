function [Sigma,Xsup,Alpsup,w0,pos,Time,Crit,SigmaH] = svmfit(Xapp,yapp,Sigma,C,option,pow,verbose)
%SVMFIT Fit SVM Gaussian classifier with adaptive scaling
%  [SIGMA,XSUP,ALPSUP,W0] = SVMFIT(XAPP,YAPP,SIGMA,C,OPTION,POW,VERBOSE)
%  returns the Gaussian scaling matrix SIGMA, the support vectors XSUP, 
%  the Lagrangian ALPSUP and the bias parameter W0 defining a Gaussian SVM
%  trained with data XAPP,YAPP,
%  initialized with scaling matrix SIGMA,
%  optimized with error penalty term C,
%  and optimization options OPTION:['wbfixed','wfixed','lbfixed','lfixed','lupdate].
%  
%  VERBOSE = [0,1,2] sets the verbosity level during the learning process
%  
%  [SIGMA,XSUP,ALPSUP,W0,POS,TIME,CRIT,SIGMAH] = SVMFIT(XAPP,YAPP,SIGMA,C,OPTION,POW,VERBOSE)
%  also returns
%  POS the indicator of support vectors in XAPP
%  TIME a (1,3) vector recording the time spent to: 
%       - TIME(1) learn the original SVM classifier
%       - TIME(2) update SIGMA 
%       - TIME(3) update the SVM parameters
%  CRIT the SVM criterion
%  SIGMAH the history of SIGMA updates

%  uses SVMCLASS, SIGMAUPDATE, COSTLBFIXED
%  27/01/03 Y. Grandvalet

%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%

[n,d] = size(Xapp);

% SVM fixed parameters (to be parameters ?)

lambdaReg    = 1e-10; 
kernel       = 'gaussian';
kerneloption = 1 ;
span         = ones(n,1); 

% Loop parameters

nloop = 0;
nloopmax = 100;
loop = 1;

% Monitoring

Time = zeros(1,3);
if nargout>=8,
   SigmaH = zeros(nloopmax,d);
end;

% Init SVM 

t = cputime ; 
[xsup,Alpsup,w0,pos] = svmclass(Xapp.*repmat(Sigma,n,1),yapp,C,lambdaReg,kernel,kerneloption,max(verbose-1,0),span);   
Time(1) = max(cputime - t,0) ;

Xsup = Xapp(pos,:);
Sigmaold  = Sigma ;
Xsupold   = Xsup ;
Alpsupold = Alpsup ;
w0old     = w0;
posold    = pos ;

%------------------------------------------------------------------------------%
% Update loop
%------------------------------------------------------------------------------%

while loop ; nloop = nloop+1; 
    if nargout>=8,
       SigmaH(nloop,:) = Sigma;
    end;
   % Update Sigma
   t = cputime ; 
   [Sigma] = SigmaUpdate(Sigma,pos,Alpsup,w0,C,Xapp,yapp,option,pow,verbose) ;
   Time(2) = Time(2) + max(cputime - t,0) ; 
   % Zero neglictible scale updates
   ind = find(Sigma<1e-6*mean(Sigma));
   if ~isempty(ind)
      Sigma(ind) = 0 ;
      Sigma      = Sigma*real(abs( (sum(real(abs(Sigma.^pow)))/sum(real(abs(Sigmaold.^pow))))^(1/pow) ));
   end;
   % If there is a significant Sigma update, update SVM parameters     
   if max(abs(Sigma - Sigmaold)./(mean(Sigmaold)+Sigmaold))>1e-3 ,
      t = cputime ; 
      if isempty(xsup) | isnan(w0) | (min(abs(Alpsup))==C)
         [xsup,Alpsup,w0,pos] = svmclass(Xapp.*repmat(Sigma,n,1),yapp,C,lambdaReg,kernel,kerneloption,0,span);
      else
        
         alphainit=zeros(size(yapp));
         alphainit(pos)=Alpsup.*yapp(pos);
         [xsup,Alpsup,w0,pos] = svmclass(Xapp.*repmat(Sigma,n,1),yapp,C,lambdaReg,kernel,kerneloption,0,span,alphainit);
     end
      Time(3) = Time(3) + max(cputime - t,0) ;
      Xsup = Xapp(pos,:);
   end;
   % Test for termination
   fprintf(1,'Iteration %3d, ',nloop)
   if size(Xsup,1) == size(Xsupold,1) ,
      if pos == posold ,
         if max(abs(Alpsup - Alpsupold)./(1+abs(Alpsupold)))<1e-3 ,
            if max(abs(Sigma - Sigmaold)./(mean(Sigmaold)+Sigmaold))<1e-3 ,
               loop = 0;
               fprintf(1,'convergence criteria reached \n');
               Sigma  = Sigmaold;
               Xsup   = Xsupold;
               Alpsup = Alpsupold;
               w0     = w0old;
               pos    = posold;
             else
               fprintf(1,'Sigma modified %9.2e\n', max(abs(Sigma - Sigmaold)./(mean(Sigmaold)+Sigmaold)));
             end;
          else
             fprintf(1,'multipliers modified %9.2e\n',max(abs(Alpsup - Alpsupold)./(1+abs(Alpsupold))))
          end;
       else
          fprintf(1,'support vectors modified \n')
       end;
    else
       fprintf(1,'number of support vectors modified \n')
    end;
    if nloop>=nloopmax ,
       loop = 0;
       fprintf(1,'maximum number of iterations reached\n')
    end;
    Sigmaold  = Sigma ;
    Xsupold   = Xsup ;
    Alpsupold = Alpsup ;
    w0old     = w0;
    posold    = pos ;
end;

%------------------------------------------------------------------------------%
% Post processing
%------------------------------------------------------------------------------%

if nargout>=7,
   Crit = costlbfixed(0,zeros(1,d),Sigma,pos,Alpsup,w0,C,Xapp,yapp,1);
   if nargout>=8,
      SigmaH = SigmaH(1:nloop,:) ;
   end;
end;
