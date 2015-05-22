function [indice,val]=featselcorrcoeff(x,y)


[nsize,nvar]=size(x);

meany=mean(y);
meanx=mean(x);

rxy=zeros(1,nvar);
for i=1:nvar
    rxy(i)= sum( (x(:,i)-meanx(i)).*(y(:)-meany))/ sqrt ( sum( (x(:,i)-meanx(i)).^2) * sum ((y(:)-meany).^2));
end;
[val,indice]=sort(-abs(rxy));

