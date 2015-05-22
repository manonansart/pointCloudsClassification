function [Sigma] = SigmaUpdate(Sigma,pos,Alpsup,w0,C,Xapp,yapp,option,pow,verbose)
%SIGMAUPDATE Update Sigma value
%     [SIGMA] = SIGMAUPDATE(SIGMA,POS,ALPSUP,W0,C,XAPP,YAPP,OPTION,POW,VERBOSE)
%     updates the kernel scale parameters SIGMA
%     based on 
%     - the current Sigma value SIGMA;
%     - the SVM parameters POS,ALPSUP,W0;
%     - the SVM hyper-parameters C and POW
%     - the learning data set XAPP,YAPP
%     - the update option OPTION: ['wbfixed','wfixed','lbfixed','lfixed','lupdate]
%  
%     VERBOSE = [0,1] sets the verbosity level during the update
%  

%       uses functions COST*FIXED, GRAD*FIXED
%       27/01/03 Y. Grandvalet


%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%

d = length(Sigma);
gold = (sqrt(5)+1)/2 ;

SigmaInit = Sigma ;
SigmaNew  = SigmaInit ; 
descold = zeros(1,d);
        
% Initial cost and gradient

switch option
case 'wbfixed'
        CostNew = costwbfixed(0,descold,SigmaNew,pos,Alpsup,w0,C,SigmaInit,Xapp,yapp,pow) ;
case 'wfixed'
        CostNew = costwfixed(0,descold,SigmaNew,pos,Alpsup,C,SigmaInit,Xapp,yapp,pow) ;
case 'lbfixed'
        CostNew = costlbfixed(0,descold,SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,pow) ;
case {'lfixed','lupdate'}
        CostNew = costlfixed(0,descold,SigmaNew,pos,Alpsup,C,Xapp,yapp,pow) ;
otherwise 
        error('Unkwown option in SigmaUpdate')
end;

switch option
case 'wbfixed'
        GradNew = gradwbfixed(SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,SigmaInit,pow) ;
case 'wfixed'
        GradNew = gradwfixed(SigmaNew,pos,Alpsup,C,Xapp,yapp,SigmaInit,pow) ;
case 'lbfixed'
        GradNew = gradlbfixed(SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,pow) ;
case {'lfixed','lupdate'}
        GradNew = gradlfixed(SigmaNew,pos,Alpsup,C,Xapp,yapp,pow) ;
end;
% reduced gradient, descent direction 
[val,coord] = max(SigmaNew) ;
GradNew = GradNew - GradNew(coord) ;
desc = - GradNew.* ( (SigmaNew>0) | (GradNew<0) ) ;
desc(coord) = - sum(desc);  % NB:  GradNew(coord) = 0

if verbose > 0,
  fprintf(1,'Sigma init, Cost init '); 
  fprintf(1,'%9.2e ',[SigmaNew  CostNew]); 
  fprintf(1,'\n');
end;

loop = any(desc~=0);
while loop, 
   SigmaOld = SigmaNew;
   CostOld  = CostNew;
   GradOld  = GradNew;
   NormGrad = GradNew*GradNew';
   if sqrt(NormGrad)<eps;
      loop = 0;
      fprintf(1,'No more descent direction. Sigma update terminated.\n')
   else
      % Compute optimal stepsize
      stepmin  = 0;
      costmin  = CostOld ;
      costmax  = 0 ;
      % maximum stepsize
      ind = find(desc<0);
      stepmax = min(-(SigmaNew(ind).^pow)./desc(ind));
      deltmax = stepmax;
      while costmax<costmin;
         switch option
         case 'wbfixed'
               costmax = costwbfixed(stepmax,desc,SigmaNew,pos,Alpsup,w0,C,SigmaInit,Xapp,yapp,pow) ;
         case 'wfixed'
               costmax = costwfixed(stepmax,desc,SigmaNew,pos,Alpsup,C,SigmaInit,Xapp,yapp,pow) ;
         case 'lbfixed'
               costmax = costlbfixed(stepmax,desc,SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,pow) ;
         case {'lfixed','lupdate'}
               costmax = costlfixed(stepmax,desc,SigmaNew,pos,Alpsup,C,Xapp,yapp,pow) ;
         end;
         if costmax<costmin
            costmin = costmax;
            SigmaP  = SigmaNew.^pow + stepmax * desc;
            SigmaNew  = abs(real(SigmaP.^(1/pow)));
	    % project descent direction in the new admissible cone
            desc = desc .* ( (SigmaNew>0) | (desc>0) ) ;
            desc(coord) = - sum(desc([[1:coord-1] [coord+1:end]]));  
            ind = find(desc<0);
            if ~isempty(ind)
               stepmax = min(-(SigmaNew(ind).^pow)./desc(ind));
               deltmax = stepmax;
               costmax = 0;
            else
               stepmax = 0;
               deltmax = 0;
            end;
         end;
      end;
      % update Lagrange multipliers ?  
      if strcmp(option,'lupdate')
         [deltAlp,stepAlp] = LagrangeUpdate(desc,SigmaNew,Xapp(pos,:),Alpsup,C,pow);
         costmax = costlfixed(stepmax,desc,SigmaNew,pos,Alpsup+min(stepmax,stepAlp)*deltAlp,C,Xapp,yapp,pow) ;
      end;
      Step = [stepmin stepmax];
      Cost = [costmin costmax];
      [val,coord] = min(Cost);
      % optimization of stepsize by golden search
      while (stepmax-stepmin)>1e-3*(deltmax);
         stepmedr = stepmin+(stepmax-stepmin)/gold;
         stepmedl = stepmin+(stepmedr-stepmin)/gold;                     
         switch option
         case 'wbfixed'
              costmedr = costwbfixed(stepmedr,desc,SigmaNew,pos,Alpsup,w0,C,SigmaInit,Xapp,yapp,pow) ;
	      costmedl = costwbfixed(stepmedl,desc,SigmaNew,pos,Alpsup,w0,C,SigmaInit,Xapp,yapp,pow) ;
         case 'wfixed'
              costmedr = costwfixed(stepmedr,desc,SigmaNew,pos,Alpsup,C,SigmaInit,Xapp,yapp,pow) ;
              costmedl = costwfixed(stepmedl,desc,SigmaNew,pos,Alpsup,C,SigmaInit,Xapp,yapp,pow) ;
         case 'lbfixed'
               costmedr = costlbfixed(stepmedr,desc,SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,pow) ;
               costmedl = costlbfixed(stepmedl,desc,SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,pow) ;
         case 'lfixed'
               costmedr = costlfixed(stepmedr,desc,SigmaNew,pos,Alpsup,C,Xapp,yapp,pow) ;
               costmedl = costlfixed(stepmedl,desc,SigmaNew,pos,Alpsup,C,Xapp,yapp,pow) ;
         case 'lupdate'
              costmedr = costlfixed(stepmedr,desc,SigmaNew,pos,Alpsup+min(stepmedr,stepAlp)*deltAlp,C,Xapp,yapp,pow) ;
              costmedl = costlfixed(stepmedl,desc,SigmaNew,pos,Alpsup+min(stepmedl,stepAlp)*deltAlp,C,Xapp,yapp,pow) ;
         end;
         Step = [stepmin stepmedl stepmedr stepmax];
         Cost = [costmin costmedl costmedr costmax];
         [val,coord] = min(Cost);
         switch coord
         case 1
              stepmax = stepmedl;
              costmax = costmedl;
         case 2
              stepmax = stepmedr;
              costmax = costmedr;
         case 3
              stepmin = stepmedl;
              costmin = costmedl;
         case 4
              stepmin = stepmedr;
              costmin = costmedr;
         end;
      end;
      CostNew = Cost(coord) ;
      step = Step(coord) ;
      % Sigma update
      if CostNew < CostOld ;
         SigmaP = SigmaNew.^pow + step * desc;
         SigmaNew  = abs(real(SigmaP.^(1/pow)));
         if verbose > 0,
            fprintf(1,'Sigma new, cost new '); 
            fprintf(1,'%9.2e ',[SigmaNew CostNew]); 
            fprintf(1,'\n');
         end;
         switch option
         case 'wbfixed'
              GradNew = gradwbfixed(SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,SigmaInit,pow) ;
         case 'wfixed'
              GradNew = gradwfixed(SigmaNew,pos,Alpsup,C,Xapp,yapp,SigmaInit,pow) ;
         case 'lbfixed'
              GradNew = gradlbfixed(SigmaNew,pos,Alpsup,w0,C,Xapp,yapp,pow) ;
         case {'lfixed','lupdate'}
              GradNew = gradlfixed(SigmaNew,pos,Alpsup,C,Xapp,yapp,pow) ;
         end;
         [val,coord] = max(SigmaNew) ;
         GradNew = GradNew - GradNew(coord) ;
         desc = ( ((GradNew - GradOld)*(GradNew)')/NormGrad )*desc - GradNew ;
         desc = desc .* ( (SigmaNew>0) | (desc>0) ) ;
         desc(coord) = - sum(desc([[1:coord-1] [coord+1:d]]));
         if all(desc==0),
            % restart from Gradient if no admissible descent direction
            desc = -GradNew .* ( (SigmaNew>0) | (GradNew<0) ) ;
            desc(coord) = - sum(desc([[1:coord-1] [coord+1:d]]));
            if all(desc==0)
               SigmaOld = SigmaNew;
            end;
         end;
      elseif (verbose > 0) & all(SigmaNew==SigmaInit),
         disp('No sigma update, step opt. = 0 ')
      end;       
      loop = max(abs(SigmaNew - SigmaOld)./(mean(SigmaOld)+SigmaOld))>1e-3;
   end;
end;
Sigma = SigmaNew ;
