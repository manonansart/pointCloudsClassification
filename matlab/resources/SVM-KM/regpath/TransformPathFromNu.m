function [alphamat,alpha0vec,lambdavec,nuvec]=TransformPathFromNu(alphamat,alpha0vec,lambdavec,nuvec,Nbapp);

N=length(nuvec);
%Nbapp=size(xapp,1);
%nuvec=fliplr(linspace(min(lambdavec)/Nbapp,max(lambdavec)/Nbapp,N));
lambdatrue=nuvec*Nbapp;
seuil=1e-6;
for i=1:N
    lambdacurrent=lambdatrue(i);
    ind=find( abs(lambdavec-lambdacurrent) <= seuil*Nbapp );
    if length(ind)>1       
    ind=ind(1);
    end;
    if isempty(ind)
        [mini,ind]=min(abs(lambdavec-lambdacurrent)) ; 
        if  lambdavec(ind)<lambdacurrent;
            lambdamoins=lambdavec(ind);
            lambdaplus=lambdavec(ind-1);
            plus=ind-1;
            moins=ind;
        else
            
            if ind+1 <= length(lambdavec)
                lambdamoins=lambdavec(ind+1);
                lambdaplus=lambdavec(ind);
                plus=ind;
                moins=ind+1;
            else
                moins=ind;
                plus=ind;
                lambdamoins=lambdacurrent;
                lambdaplus=0;
            end;
        end; 
        newalphamat(:,i)=alphamat(:,moins)+ (lambdacurrent - lambdamoins)*(alphamat(:,plus)-alphamat(:,moins))/(lambdaplus-lambdamoins);
        newalpha0vec(i)=alpha0vec(moins)+    (lambdacurrent - lambdamoins)*(alpha0vec(:,plus)-alpha0vec(:,moins))/(lambdaplus-lambdamoins);
    else
        newalphamat(:,i)=alphamat(:,ind);
        newalpha0vec(:,i)=alpha0vec(ind);
    end;
    
end;

alphamat=newalphamat;
alpha0vec=newalpha0vec;
lambdavec=lambdatrue;