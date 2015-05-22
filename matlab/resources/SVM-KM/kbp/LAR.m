function [solution, solution_OLS] = LAR(x,y, AllBorne, Limites, lambda, verbose, xappforplot)
%
% USAGE :
%  [solution, solution_OLS] = LAR(x,y, AllBorne, Limites, lambda, verbose, xappforplot)
%
%  Looks for a regression solution : y_hat = x Beta +b
%  by solving : min ||y_hat-y||^2 
%               sum(abs(Beta)) < t
%
%  The algorithm is based on the LARS of Efron et al.
%  Stopping criterion and multiple kernel formulation are based on
%  V. Guigue's paper untitled Kernel Basis Pursuit (google it!).
%
% ARGUMENTS :
%   verbose    : plot (>0) or no (==0) lfigures.
%               figures are numbered from verbose
%   x          : input data
%   y          : output data
%   AllBorne :  This paraameter is a structure that deals with the stopping
%               criterion. One can combine several kinds of stopping
%               criterion for the processing of the regularization path.
%
%                    Allborne{i}.type : type of stopping criterion 
%                    Allborne{i}.borne : bound value on this stopping
%                                       criterion
%                                     
%                   Here we give the possible type of stopping criterion
%
%                   'sumBeta'  ->  sum(abs(Beta)) < borne_Beta
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.
%                       
%                       (usual bound as defined in the LARS algorithm)
%
%                   'pcSV'     ->  borne_Beta = percent of selected support
%                               vectors
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.

%                   'nbSV'     ->  borne_Beta = nb of support vector
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.

%                   'trapscale'-> scale or kernel representing noise 
%                       * structure.indTS : index of the kernel trapscale
%                       in the multiple kernel setting. It should be one of
%                       the last.                   
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds. (counts
%                       the number of time the trapscale has been
%                       selected.)
%
%                   'RVtrap'   ->  Automatically add some random noise as
%                   information sources.
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds. (counts
%                       the number of time the RV has been selected).
%
%                   'countback' -> allowed backward movement
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.
%
%   Limites         * structure.nbmaxMoveBack : maximal number of backward
%                     movement for the LASSO.
%                   * structure.nbmaxSV : maximal number of SV allowed.
%                       this is useful in a Large Scale problem in order to
%                       avoid memory crash.
%
%   lambda : regularisation term
%
% OUTPUT:
%   solution  : all the solutions in a structures, the beta are only the
%                   non-zeros one
%                   solution{i}.b   
%                   solution{i}.Beta
%                   solution{i}.indxsup
%                   solution{i}.type % de solution !
%                   solution{i}.borne
%
%   solution_OLS : same solution  but the last step is based on the OLS not
%   on LARS.
%
%





% V. Guigue, A. Rakotomamonjy 12/2004


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRE TEST

if nargin<6
    verbose = 0;
end

nbvar    =  size(x,2);
nbptsapp =  size(x,1);
% if sum(std(x)) > nbvar+0.5 | sum(std(x)) < nbvar-0.5
%     %error('LARS method should be used on normalized data\n');   
%     fprintf('LARS method should be used on normalized data\n');
% end

 if sum(mean(x))>1e-3;
     fprintf('LARS method should be used on normalized data\n');
 end

nbexptot = 0;
nbborne  = length(AllBorne);
for i=1:nbborne,
    nbexptot = nbexptot + length(AllBorne{i}.borne);
    
    if strcmp(AllBorne{i}.type,'trapscale') % transformation de l'indice de debut TS
        AllBorne{i}.indTS = (AllBorne{i}.indTS-1)*nbptsapp;   
    end
    
end

% fabrication d'un ensemble de point aleatoire :
RVset = randn(nbptsapp,15)';


nbexpcur = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AJOUT DE B NORMALISE

x = [x ones(size(x,1),1)/sqrt(nbptsapp)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMTRES PAR DEFAUT


test_OLS = 1;
eps = 1e-12;

if isfield(Limites,'nbmaxMoveBack') == 0
    Limites.nbmaxMoveBack = 1e8;
end
if isfield(Limites,'nbmaxSV') == 0
    Limites.nbmaxSV = 1e8;
end

%lambda
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALISATION
y_hat     = zeros(size(y));

% points courrants
indxsup      = [];
gamma_sup    = [];
touslesind   = 1:size(x,2);
nbdim        = size(x,2);
% ind_nonxsup  = touslesind;
Beta         = zeros(1,nbdim);
arret         = 0;
move_backward = 0;


compt_trap     = 0;
compt_backward = 0;
compt_borne    = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALISATION MEMOIRES
c_mem           = [];
Beta_mem       = [zeros(1,nbdim)];
cout_tot_mem   = [];
sum_Beta_mem   = [];

iter           = 0;
ptsadd         = [];
ptsret         = [];
scale_mem      = [];
stdR_mem       = [];
R_im1          = 100000;

if verbose ~=0
    gcf=figure(verbose);
    set(gcf, 'Position', [100 500  1000 400]);
    %keyboard
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BOUCLES

while arret == 0,
        
    ind_nonxsup          = setdiff(touslesind,indxsup);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CORRELATION VARIABLE RESIDU
     R                    = (y - y_hat);
    % R                       = max(0,1-(y.*y_hat));
    % R                       = R.*sign(y);
    
    %keyboard
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % VERIFICATION DE CVG 
%     if sum(abs(R)) > R_im1 + lambda %cvg non assure
%         fprintf('Cvg non assuree !\n');
%         break;
%     end
%     R_im1                = sum(abs(R));
    
    c                    = x'*R;
    
    [max_c,indmax_c]     = max(abs(c(ind_nonxsup))); % point de plus fort cout
    indmax_c             = ind_nonxsup(indmax_c);
    
    if move_backward == 0
        indxsup              = [indxsup; indmax_c];   
        ptsadd               = [ptsadd; iter indmax_c];
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Existe-t-il une VA qui est plus correlee ?
    c_RV                 = abs(RVset*R); % std = 1, mean = 0
    Compt_RVpluscor      = length(find(c_RV > max_c));

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MEMOIRE ET TRACE
    
    cout_tot_mem = [cout_tot_mem sum(R.^2)];
    sum_Beta_mem = [sum_Beta_mem sum(abs(Beta))];
    
    if verbose ~=0
        figure(verbose)  
        nbfig = 3;
        subplot(1,nbfig,1);
        
        h= plot(cout_tot_mem); 
        set(h,'linewidth',2);
        h= title('Cost (MSE)');
        set(h,'fontsize',16);
        
        subplot(1,nbfig,2);
        %figure(verbose+1)  
        h= plot(sum_Beta_mem,'r');
        set(h,'linewidth',2);
        h= title('\Sigma_i |\beta_i|');
        set(h,'fontsize',16);
        subplot(1,nbfig,3);
        %figure(verbose+2)  
        if length(ptsadd) > 0
            h= plot(ceil(ptsadd(:,2)/size(x,1)));
            set(h,'linewidth',2);
        end
        h= title('Scales of chosen points');
        set(h,'fontsize',16);
        
%         figure(verbose+1)
%         subplot(3,1,1)
%         plot(xappforplot,y,'b');
%         h= title('cos(exp(\omega t))');
%         set(h,'fontsize',16);
%         subplot(3,1,2)
%         plot(xappforplot,y_hat,'b');
%         h= title('\hat{y}');
%         set(h,'fontsize',16);
%         subplot(3,1,3)
%         plot(xappforplot,R,'b');
%         h= title('Residu');
%         set(h,'fontsize',16);
% 
%         figure(verbose+2)
%         imagesc(reshape(c(1:end-1,:),nbptsapp,nbvar/nbptsapp));
%         h= title('Correlation residu/K_i(x_j,\cdot)');
%         set(h,'fontsize',16);
        
        drawnow;
%         pause
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CALCUL DU PAS
    
    S        = sign(c);   
    
    % METHODE CHOL
    
        Xsup     = x(:,indxsup) * diag(S(indxsup));
        

            regterm    = lambda;
    
        Gsup     = Xsup'*Xsup + eye(length(indxsup))*regterm;           
        
%   CHOL        
%         Rchol    = chol(Gsup);
%         bchol    = Rchol'\ones(size(Gsup,1),1);
%         ychol    = Rchol\bchol;
%         Asup     = (sum(ychol))^(-0.5);
%         Wsup     = Asup*ychol;
    %     
% INVERSION BRUTALE !
    
    invGsup  = inv(Gsup);    % INVERSION MAT !
    Asup     = (sum(sum(invGsup)))^(-0.5);
    Wsup     = sum(Asup*invGsup,2);
    
% METHODE SANS INVERSION -> uniquement pour les bases orthogonales
% 
%     Wsup       = [R'*Xsup]'; % projection du residu sur les variables actives
%     %Usup       = Xsup*Wsup;
%     %Usup       = Usup/sqrt(Usup'*Usup); % normalisation    
%     b          = x'*x*W;

    % FIN COMMUNE
%     
    Usup     = Xsup*Wsup;
    b        = x'*Usup;
    W          = zeros(nbdim,1);
    W(indxsup) = Wsup;
        
    ind_nonxsup          = setdiff(touslesind,indxsup);
    
    
    if length(ind_nonxsup) ~= 0
        gamma_out_xsup_moins = (max_c - c(ind_nonxsup))./(Asup - b(ind_nonxsup));
        gamma_out_xsup_plus  = (max_c + c(ind_nonxsup))./(Asup + b(ind_nonxsup));
        all_gamma            = [gamma_out_xsup_moins; gamma_out_xsup_plus];
        all_gamma            = all_gamma(find(all_gamma>0)); % uniquement les valeurs positives des gammas
        gamma                = min(all_gamma);
        if isempty(gamma)
            gamma = eps;
        end
    else
        gamma                = max_c/Asup;
    end
    
    move_backward = 0;
    % calcul de beta

    Beta_tp1             = Beta + gamma*S'.*W';  
    
    % gamma_OLS            = max_c/Asup;
     gamma_OLS            = max(c(indxsup)./b(indxsup));
    % gamma_OLS            = (gamma + 3*min(c(indxsup)./b(indxsup)))/4;
    if isempty(gamma_OLS)
        gamma_OLS = eps;
    end
    Beta_OLS             = Beta + gamma_OLS*S'.*W';
    
    inddiffsign = [];
    inddiff0             = find(Beta > eps | Beta < -eps);
    inddiffsign          = find(sign(Beta(inddiff0)) ~= sign(Beta_tp1(inddiff0)));
    inddiffsign          = inddiff0(inddiffsign);
    
    if length(inddiffsign) == 0
        Beta          = Beta_tp1;
    elseif length(inddiffsign) == 1
        if verbose ~= 0
            fprintf('move backward (%d) \n', length(indxsup));
        end
        
        indtodel      = inddiffsign(1);        
        gamma         = Beta(indtodel)/(W(indtodel)'*-S(indtodel));
        
        Beta          = Beta + gamma*S'.*W';        
        indtodel      = find(indxsup == indtodel);
        move_backward = 1;
        compt_backward = compt_backward+1;
    else     
        if verbose ~= 0
            fprintf('move backward mult (%d) \n', length(indxsup));
        end
        
        gamma         = Beta(inddiffsign)./(W(inddiffsign)'.*-S(inddiffsign)');
        [gamma,ind]   = min(gamma);
        indtodel      = inddiffsign(ind);
        
        Beta          = Beta + gamma*S'.*W';       
        indtodel      = find(indxsup == indtodel);
        move_backward = 1;
        compt_backward = compt_backward+1;
    end
    
    
    % stockage gamma
    gamma_sup            = [gamma_sup gamma];
    
    % calcul de y_hat
    y_hat                = x*Beta';
    
    y_hat_OLS            = x*Beta_OLS';
    
    Beta_mem   = [Beta_mem; Beta];
    sumabsBeta = sum(abs(Beta_mem),2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % LASSO (SUITE)
    
    % enlever le vecteur support correspondant
    if move_backward == 1     
        ptsret                 = [ptsret; iter indxsup(indtodel)];
        indxsup(indtodel)      = [];
        gamma_sup(indtodel)    = [];
        sum_Beta_mem(indtodel) = [];
        cout_tot_mem(indtodel) = [];
        ind_nonxsup            = setdiff(touslesind,indxsup);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CRITERES ARRET
    
    if length(indxsup) > Limites.nbmaxSV | compt_backward >Limites.nbmaxMoveBack 
        % il faut tout arreter !
        fprintf('Bornes de securite depassees.\n');
        break;
    end
    
    if move_backward ~= 1  % ne pas prendre de solution sous-optimale
        for i=1:nbborne
            RETENIR = 0;    
            if strcmp(AllBorne{i}.type,'sumBeta' ) & length(AllBorne{i}.borne)>0
                if sum(abs(Beta)) > AllBorne{i}.borne(1)  
                    RETENIR = 1;
                end
                
            elseif strcmp(AllBorne{i}.type,'pcSV' ) & length(AllBorne{i}.borne)>0
                if length(indxsup)/nbdim > AllBorne{i}.borne(1)
                    RETENIR = 1;
                end
                
            elseif strcmp(AllBorne{i}.type,'nbSV') & length(AllBorne{i}.borne)>0
                if length(indxsup) >= AllBorne{i}.borne(1)
                    RETENIR = 1;            
                end
                
            elseif strcmp(AllBorne{i}.type,'trapscale')  & length(AllBorne{i}.borne)>0
                if (indxsup(end) > AllBorne{i}.indTS & indxsup(end) < size(x,2)) % le point appartient a une echelle interdite
                    compt_trap = length(find(indxsup > AllBorne{i}.indTS & indxsup < size(x,2))); % combien ?
                    
                    if compt_trap > AllBorne{i}.borne(1) 
                        RETENIR = 1;    
                    end
                end
                
            elseif strcmp(AllBorne{i}.type,'RVtrap') & length(AllBorne{i}.borne)>0
                if Compt_RVpluscor > AllBorne{i}.borne(1)
                    RETENIR = 1;   
                end
                
            elseif strcmp(AllBorne{i}.type,'countback') & length(AllBorne{i}.borne)>0
                if compt_backward > AllBorne{i}.borne(1) 
                    RETENIR = 1;    
                end    
                
            end
            
            
            if RETENIR == 1
                solution{nbexpcur}.indxsup = indxsup;
                solution{nbexpcur}.Beta    = Beta(indxsup);
                solution{nbexpcur}.type    = AllBorne{i}.type;
                if strcmp(AllBorne{i}.type,'trapscale'),
                    solution{nbexpcur}.borne   = AllBorne{i}.borne(1)*0.01 + AllBorne{i}.indTS;% recomposition !
                else
                    solution{nbexpcur}.borne   = AllBorne{i}.borne(1);
                end
                
                if test_OLS == 1
                    solution_OLS{nbexpcur}.indxsup = indxsup;
                    solution_OLS{nbexpcur}.Beta    = Beta_OLS(indxsup);
                    solution_OLS{nbexpcur}.type    = AllBorne{i}.type;% '_OLS'];
                    if strcmp(AllBorne{i}.type,'trapscale'),
                        solution_OLS{nbexpcur}.borne   = AllBorne{i}.borne(1)*0.01 + AllBorne{i}.indTS; % recomposition !
                    else
                        solution_OLS{nbexpcur}.borne   = AllBorne{i}.borne(1);
                    end
                        
                end
                
                
                %fprintf('%4d/%4d : %15s (%f) # %d\n',nbexpcur,nbexptot,solution{nbexpcur}.type,solution{nbexpcur}.borne,length(indxsup));
                fprintf('\b\b\b\b\b\b\b\b\b%4d/%4d',nbexpcur,nbexptot);
                
                AllBorne{i}.borne(1)     = []; % elimination !
                nbexpcur                 = nbexpcur +1;     
                %            size(indxsup)
            end
        end
    end
    if length(indxsup) == nbdim 
        fprintf('Regression totale !\n');
        break;
    end
    
    
    
    iter = iter+1;
    
    if nbexpcur-1==nbexptot, % le compteur est en avance !
        break;
    end
    % pause
end
fprintf('\n');

nbexpcur    = nbexpcur-1;

for i=1:nbexpcur
    
    indb = find(solution{i}.indxsup == size(x,2));
    if length(indb) ~= 0
        solution{i}.b = solution{i}.Beta(indb) / sqrt(nbptsapp); % "denormalisation"
        solution{i}.Beta(indb) = [];
        solution{i}.indxsup(indb) = [];    
    else
        solution{i}.b = 0;
    end
    
    if test_OLS == 1
        if length(indb) ~= 0
            solution_OLS{i}.b = solution_OLS{i}.Beta(indb) / sqrt(nbptsapp); % "denormalisation"
            solution_OLS{i}.Beta(indb) = [];
            solution_OLS{i}.indxsup(indb) = [];    
        else
            solution_OLS{i}.b = 0;
        end
    end
end