%--------------------------------------%
% init MATLAB
%--------------------------------------%

clear all ; 

%--------------------------------------%
% data
%--------------------------------------%

Napp = 100;
d    = 2;

angle = 0*pi/4;
ovlap = 0.10 % Bayes Err.

Xapp = rand(Napp,d)*diag([1 2])+[ [-(1-2*ovlap)*ones(Napp/4,1) ; (1-2*ovlap)*ones(Napp/4,1) ; zeros(Napp/2,1)] zeros(Napp,1)];
mXapp = mean(Xapp);
Xapp = Xapp - repmat(mXapp,Napp,1) ;
Xapp = Xapp*[cos(angle) -sin(angle) ; sin(angle) cos(angle)];
Yapp = [ones(Napp/2,1) ; -ones(Napp/2,1) ];

%--------------------------------------%
% SVM parameters
%--------------------------------------%

lambdaReg    = 1e-10; 
kernel       ='gaussian';
kerneloption = 1;
verbose = 3 ; 
Sigma = 1e0*ones(1,d);
C = 1e0;

% sigma tuning

option  = 'lupdate'  ;
pow = 2 ;

%--------------------------------------%
% display parameters
%--------------------------------------%

ngrid = 31;
minX1 = min(Xapp(:,1));
maxX1 = max(Xapp(:,1));
DltX1 = (maxX1-minX1)/20;
minX2 = min(Xapp(:,2));
maxX2 = max(Xapp(:,2));
DltX2 = (maxX2-minX2)/20;

[XGrid1,XGrid2] = meshgrid(linspace(minX1-DltX1,maxX1+DltX1,ngrid),linspace(minX2-DltX2,maxX2+DltX2,ngrid));

%--------------------------------------%
% SVM learning
%--------------------------------------%

[Sigma,Xsup,Alpsup,w0,pos,nflops,crit,SigmaH] = svmfit(Xapp,Yapp,Sigma,C,option,pow,verbose);

ysup  = Yapp(pos);
nsup = length(pos);
ypred = svmval([XGrid1(:) XGrid2(:)].*repmat(Sigma,ngrid*ngrid,1),Xsup.*repmat(Sigma,nsup,1),Alpsup,w0,kernel,kerneloption,ones(ngrid*ngrid,1));
ypred = reshape(ypred,ngrid,ngrid);

%--------------------------------------%
% Graphical output
%--------------------------------------%

figure(1);clf;
h=plot(Xapp(Yapp==1,1),Xapp(Yapp==1,2),'+r');
set(h,'MarkerSize',9);
hold on;
h=plot(Xapp(Yapp==-1,1),Xapp(Yapp==-1,2),'xb');
set(h,'MarkerSize',9);
h=plot(Xsup(ysup==1,1),Xsup(ysup==1,2),'or');
set(h,'MarkerSize',9,'LineWidth',2);
h=plot(Xsup(ysup==-1,1),Xsup(ysup==-1,2),'ob'); 
set(h,'MarkerSize',9,'LineWidth',2);
[cc, hh]=contour(XGrid1,XGrid2,ypred,[-1 0 1]);
if ~isempty(cc);
   set(hh,'LineWidth',1)
   clabel(cc,hh);
end;		
if all(Sigma~=0),
   XsupG = Xsup.*repmat(Sigma,nsup,1);
   XsupD = [XGrid1(:) XGrid2(:)].*repmat(Sigma,ngrid*ngrid,1);
   Dist = XsupG*XsupD';
   Dist = - Dist + repmat(0.5*sum(XsupG.^2,2),1,ngrid*ngrid) + repmat(0.5*sum(XsupD.^2,2)',nsup,1) ; 
   Dist = exp(-Dist) ;
   cc = exp(-Sigma*Sigma'/64);
   for i=1:nsup;
      [c,h] = contour(XGrid1,XGrid2,reshape(Dist(i,:),ngrid,ngrid),cc*[1 1],'-');
      set(h,'Color',0.75*[1 1 1])
   end;
end;
set(gca,'XTick',[],'YTick',[]);
title('final solution');
axis('equal');
drawnow;

% H = input('History? (Y/N)','s');
H='y';
if lower(H)=='y';
   figure(2);clf;
   nh = size(SigmaH,1);
   indh = unique(round(linspace(1,nh,4)));
   for i=1:length(indh);
       [xi,ai,wi,posi] = svmclass(Xapp.*repmat(SigmaH(indh(i),:),Napp,1),Yapp,C,lambdaReg,kernel,kerneloption,0,ones(Napp,1));
       Xi = Xapp(posi,:);
       yi = Yapp(posi);
       ni = length(posi);
       ypred = svmval([XGrid1(:) XGrid2(:)].*repmat(SigmaH(indh(i),:),ngrid*ngrid,1),Xi.*repmat(SigmaH(indh(i),:),ni,1),ai,wi,kernel,kerneloption,ones(ngrid*ngrid,1));
       ypred = reshape(ypred,ngrid,ngrid);
       subplot(2,2,i);
		plot(Xapp(Yapp==1,1),Xapp(Yapp==1,2),'+r');
		hold on;
		plot(Xapp(Yapp==-1,1),Xapp(Yapp==-1,2),'xb');
		h=plot(Xsup(ysup==1,1),Xsup(ysup==1,2),'or');
		set(h,'LineWidth',1);
		h=plot(Xsup(ysup==-1,1),Xsup(ysup==-1,2),'ob'); 
		set(h,'LineWidth',1);
		[cc, hh]=contour(XGrid1,XGrid2,ypred,[-1 0 1]);
		if ~isempty(cc);
            set(hh,'LineWidth',1)
            clabel(cc,hh);
		end;		
		if all(SigmaH(indh(i),:)~=0),
            XsupG = Xi.*repmat(SigmaH(indh(i),:),ni,1);
            XsupD = [XGrid1(:) XGrid2(:)].*repmat(SigmaH(indh(i),:),ngrid*ngrid,1);
            Dist = XsupG*XsupD';
            Dist = - Dist + repmat(0.5*sum(XsupG.^2,2),1,ngrid*ngrid) + repmat(0.5*sum(XsupD.^2,2)',ni,1) ; 
            Dist = exp(-Dist) ;
            cc = exp(-SigmaH(indh(i),:)*SigmaH(indh(i),:)'/64);
            for j=1:nsup;
               [c,h] = contour(XGrid1,XGrid2,reshape(Dist(j,:),ngrid,ngrid),cc*[1 1],'-');
               set(h,'Color',0.75*[1 1 1])
            end;
		end;
		set(gca,'XTick',[],'YTick',[]);
		title(['Solution step' num2str(indh(i)) '/' num2str(nh)]);
		axis('equal')
   end;
end;
   
