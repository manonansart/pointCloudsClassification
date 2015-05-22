function  [xapp,yapp,xtest,ytest,indice]=CreateDataAppTestReg(x,y,nbtrain)

% [xapp,yapp,xtest,ytest,indice]=CreateDataAppTestReg(x,y,nbtrain)
%

    xapp=[];
    yapp=[];
    xtest=[];
    ytest=[];
    indice=[];
    indice.app=[];
    indice.test=[];

    nbdata=length(y);
        ind=1:nbdata;
        aux=randperm(nbdata);

        indapp=ind(aux(1:nbtrain));
        indtest=ind(aux(nbtrain+1:end));
        xapp=[x(indapp,:)];
        yapp=[y(indapp,:)];
        xtest=[x(indtest,:)];
        ytest=[;y(indtest,:)];
        indice.app=[indapp];
        indice.test=[indtest];
