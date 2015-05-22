clear all
close all
clc

nbitermax=100;
ratio=0.6;  %need to add a 

classcode=[1 -1];


datafile={'colon'};
%datafile={'wpbc' 'sonar'};
for i=1:length(datafile);
    data=datafile{i};
    
    
    file=['../../data/' data '/' data '.mat'];
    load(file);
    nbtrain = round (size(x,1)*ratio); 
    
    
    for iter=1:nbitermax
        [xapp,yapp,xtest,ytest,indice(iter)]=CreateDataAppTest(x,y,nbtrain, classcode);
    end;
    
    filesave=['CV-' data '-ratio-' int2str(ratio*100) '.mat'];
    save(filesave, 'indice')
end;