
% Created by Hamdi Eryilmaz on 9/14/23

% This script relates misclassification index to binary covariates such as
% Race and Sex. 
% Adapted from Greene et al. 2022 as cited in the paper. 

% Greene AS, Shen X, Noble S, Horien C, Hahn CA, Arora J, Tokoglu F, Spann MN,
% Carrión CI, Barron DS, Sanacora G, Srihari VH, Woods SW, Scheinost D, Constable RT (2022).
% Brain–phenotype models fail for individuals who defy sample stereotypes. Nature, 609(7925), 109-118.

%% Extract average misclassification index (MI) for a given outcome

% Set your outcome and the covariate
myanalysis = 'TOTAL';
mycovariate = 'Race'; 
subgroup = 'highscore'; % 'lowscore' or 'highscore'

switch myanalysis
    case 'TOTAL' 
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_MisIdx_9_14_total.mat
    case 'FLUID'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_MisIdx_9_14_fluid.mat
    case 'CRYSTAL'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_MisIdx_9_14_crystal.mat
end

% Compute Misclassification Index
mis_perc = sum(misclass_mtx,2);
%mis_idx_allsubs = (mis_perc/niter); 
mis_idx_allsubs = (mis_perc/niter); 

%% Get MI for high and low scorers

N=no_sub; % full sample
true_label_all = true_label(:,1); % participant's actual performance (-1 or 1)

misidx_low = nan(N,1); misidx_high = nan(N,1);
misidx_low(find(true_label_all==-1)) = mis_idx_allsubs(find(true_label_all==-1));
misidx_high(find(true_label_all==1)) = mis_idx_allsubs(find(true_label_all==1));

switch subgroup
    case 'lowscore'
        mis_idx = misidx_low;
    case 'highscore'
        mis_idx = misidx_high;
end

thenans = isnan(mis_idx);

%% Compare MI between the different groups of the binary covariate

covtable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_covariates.xlsx','Sheet','Sheet1');
eval(['mycov = covtable.' mycovariate ';'])

if strcmp(mycovariate,'Sex')  % for binary variables do a Wilcoxon test
    % Binarize the covariates
    mycov_bin = zeros(numel(mycov),1);
    for ic=1:numel(mycov)
        if strcmp(mycov{ic,1},'F') % assign 1 to female subjects
            mycov_bin(ic,1) = 1;
        end
    end
    % Remove the NaN subjects from covariate and MF
    % covnans = isnan(mycov_bin); % no missing Sex data
    cov_subset = mycov_bin(thenans==0);
    mis_idx_subset = mis_idx(thenans==0);

    mf_gr1 = mis_idx_subset(find(cov_subset==0)); mf_gr2 = mis_idx_subset(find(cov_subset==1));
    [pval,h,stats] = ranksum(mf_gr1,mf_gr2);

    g = [zeros(length(mf_gr1), 1); ones(length(mf_gr2), 1)];
    figure, boxplot([mf_gr1; mf_gr2], g); eval(['title(''MI for covariate ' mycovariate ' in ' subgroup ' group'');'])
    xticklabels({'Male','Female'}); ylabel('Misclassification Index');

elseif strcmp(mycovariate,'Race') 
    % Binarize the covariates
    mycov_bin = zeros(numel(mycov),1);
    for ic=1:numel(mycov)
        if strcmp(mycov{ic,1},'<undefined>') % assign NaN to <undefined> subjects
            mycov_bin(ic,1) = NaN;
        elseif ~strcmp(mycov{ic,1},'White') % assign 1 to non-White subjects
            mycov_bin(ic,1) = 1;
        end
    end
    % Remove the NaN subjects from covariate and MF
    covnans = isnan(mycov_bin);
    cov_subset = mycov_bin(thenans==0 & covnans==0);
    mis_idx_subset = mis_idx(thenans==0 & covnans==0);

    mf_gr1 = mis_idx_subset(find(cov_subset==0)); mf_gr2 = mis_idx_subset(find(cov_subset==1));
    [pval,h,stats] = ranksum(mf_gr1,mf_gr2);

    g = [zeros(length(mf_gr1), 1); ones(length(mf_gr2), 1)];
    figure, boxplot([mf_gr1; mf_gr2], g); eval(['title(''MI for covariate ' mycovariate ' in ' subgroup ' group'');'])
    xticklabels({'White','RG'}); ylabel('Misclassification Index');  

end

