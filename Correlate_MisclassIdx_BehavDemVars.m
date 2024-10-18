
% Created by Hamdi Eryilmaz on 9/14/23 

% This script computes the correlation between misclassification index
% (MI) and continuous clinical and sociodemographic covariates in the
% HCP-EP data.
% Adapted from Greene et al. 2022 as cited in the paper. 

% Greene AS, Shen X, Noble S, Horien C, Hahn CA, Arora J, Tokoglu F, Spann MN,
% Carrión CI, Barron DS, Sanacora G, Srihari VH, Woods SW, Scheinost D, Constable RT (2022).
% Brain–phenotype models fail for individuals who defy sample stereotypes. Nature, 609(7925), 109-118.

%% Extract average misclassification index (MI) for a given outcome

% Set your outcome and the covariate
myanalysis = 'TOTAL';
mycovariate = 'POSSX';

switch myanalysis
    case 'TOTAL' 
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_MisIdx_9_14_total.mat
    case 'FLUID'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_MisIdx_9_14_fluid.mat
    case 'CRYSTAL'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_MisIdx_9_14_crystal.mat
end

% Compute MI
mis_perc = sum(misclass_mtx,2);
mis_idx = (mis_perc/niter); 
mis_idx_noNan = mis_idx(~isnan(mis_idx));

%% Calculate the correlation between MI and the covariate

covtable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_covariates.xlsx','Sheet','Sheet1');
eval(['mycov = covtable.' mycovariate ';'])

% Get MI and covariate values for the subjects included in the MI analysis (low and high scorers)
thenans = isnan(mis_idx);
covnans = isnan(mycov);
cov_subset = mycov(thenans==0 & covnans==0);
mis_idx_subset = mis_idx(thenans==0 & covnans==0);

% Get frequently correctly and incorrectly classified participants 
lowMI_idx = find(mis_idx_subset<=0.4);
highMI_idx = find(mis_idx_subset>=0.6);

% Compare covariate in correctly vs. incorrectly classified participants
lowMI_cov = cov_subset(lowMI_idx);
highMI_cov = cov_subset(highMI_idx);
disp([mean(lowMI_cov) mean(highMI_cov)]);

% Get low and high scorers in the subset for which we have MI and covariate
behav_subset = all_behav(thenans==0 & covnans==0);
lowidx = find(behav_subset<=median(behav_subset));
lowscore = behav_subset(lowidx);
highidx = find(behav_subset>median(behav_subset));
highscore = behav_subset(highidx);
% Extract MI and covariate for low and high scorers
mis_idx_low = mis_idx_subset(lowidx);
mis_idx_high = mis_idx_subset(highidx);
cov_low = cov_subset(lowidx);
cov_high = cov_subset(highidx);
% Run correlation
[rho_low_scorer, pval_low_scorer] = corr(mis_idx_low,cov_low,'Type','Spearman');
[rho_high_scorer, pval_high_scorer] = corr(mis_idx_high,cov_high,'Type','Spearman');

% Get cognitive outcome for subjects with low and high MI 
behav_lowMI = behav_subset(lowMI_idx);
behav_highMI = behav_subset(highMI_idx);
% Run the correlation between covariate and cognitive outcome in low and high MI participants
cov_lowMI = cov_subset(lowMI_idx);
cov_highMI = cov_subset(highMI_idx); 
[rho_lowMI,pval_lowMI] = corr(cov_lowMI,behav_lowMI); 
[rho_highMI,pval_highMI] = corr(cov_highMI,behav_highMI);

