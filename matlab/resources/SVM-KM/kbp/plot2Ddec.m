function plot2Ddec(xforplot,y,A,xgrid_1, xgrid_2, ft, nfig, hinge )

if nargin < 8
    hinge=0;
end
if ~isempty(nfig)
    if length(nfig)==1
        figure(nfig);
    else %if length(nfig)>1
        subplot(nfig(1), nfig(2), nfig(3));
    end
else
    figure
end



couleur = {'b*','r*','g*','c*','k*','m*'};

[val,ind_max] = max(A);
if val > length(y)
    A(ind_max) = []; % coef b, sans consequence pour le retour dans PP
end

ngrid = length(xgrid_1);

Classes = unique(sort(y));

for i=1:length(Classes)
    h=plot(xforplot(find(y==Classes(i)),1),xforplot(find(y==Classes(i)),2),couleur{i}); hold on;
    set(h,'markersize',10);
    set(h,'linewidth',2);
end
if length(find(A > size(xforplot,1))) == 0
    h=plot(xforplot(A,1),xforplot(A,2),'sk'); hold on
    set(h,'markersize',10);
    set(h,'linewidth',2);
end

% text(xforplot(A,1),xforplot(A,2)+0.05,num2str(b));
%    pp = plot(x,yp,'c');
if hinge==1
    [cs,h] = contour(xgrid_1, xgrid_2, reshape(ft, ngrid,ngrid),[-1 0 1]);
    set(h,'linewidth',2);

%     clabel(h,cs);
else
    [cs,h] = contour(xgrid_1, xgrid_2, reshape(ft, ngrid,ngrid),[0 0]);
    set(h,'linewidth',2);
end
% title(['move backward : ' num2str(move_backward)]);
%     surf(xgrid_1, xgrid_2, reshape(ft, ngrid,ngrid));


%     plot(xi(indnextpt),yi(indnextpt),'pm');

%     plot(x,ft,'r');
%     plot(xi,K(:,A)*v,'--m');
hold off