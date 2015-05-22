function CreateCVtoy(file,nbtrain,nbCV);



%clear all
% nbCV=100;
% nbtrain=50;
% file='toygaussian';



load(file);
nbdata=size(x,1);
classcode=[1 -1];


for i=1:nbCV
        
        [xapp,yapp,xtest,ytest,indice(i)]=CreateDataAppTest(x,y,nbtrain, classcode);

end

save(['CV-' file '-nbapp' int2str(nbtrain) '-nbdata' int2str(nbdata) '.mat']);
