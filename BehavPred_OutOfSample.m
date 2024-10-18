
% Edited by Hamdi Eryilmaz on 11/1/23 

% This script builds a connectome-based model to predict NIH toolbox Fluid cognition
% scores using the full HCP-EP sample (N=92) and validates it externally 
% by predicting working memory performance (a subdomain of Fluid cognition) 
% in the MGH sample (N=18).

% The connectome-based predictive modeling framework used here was originally developed by 
% Finn et al (Copyright 2015 Xilin Shen and Emily Finn) as cited in the paper:

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.

% Edited by Hamdi Eryilmaz to implement external validation of the model
% using an independently acquired dataset. 

%%
clear;
clc;
myanalysis = 'all_fluid'; % all_cogcomp all_fluid all_crystal
testset_outcome = 'acc_AVG'; % average working memory accuracy (ratio of trials answered correctly by the participant)

%% Get connectivity and behavioral data for the behavioral outcome selected

switch myanalysis

    % Total cognition
    case 'all_cogcomp'

        mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
        subs = mytable.subject;
        Nreg=333;

        % ------------ INPUTS -------------------
        % Outcome variable
        all_behav=mytable.NIH_cogcomp;

        % Corr matrices
        all_mats=zeros(Nreg,Nreg,length(subs));
        for i=1:length(subs)
            eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/Correlation Matrices/Individual_CMs_afterqc/' num2str(subs(i)) '_CM.mat'')'])
            all_mats(:,:,i)=CM_indiv;
        end

    % Fluid cognition
    case 'all_fluid'

        mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
        subs = mytable.subject;
        Nreg=333;

        % ------------ INPUTS -------------------
        % Outcome variable
        all_behav=mytable.NIH_fluidcog;

        % Corr matrices
        all_mats=zeros(Nreg,Nreg,length(subs));
        for i=1:length(subs)
            eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/Correlation Matrices/Individual_CMs_afterqc/' num2str(subs(i)) '_CM.mat'')'])
            all_mats(:,:,i)=CM_indiv;
        end

    % Crystallized cognition
    case 'all_crystal'

        mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
        subs = mytable.subject;
        Nreg=333;

        % ------------ INPUTS -------------------
        % Outcome variable
        all_behav=mytable.NIH_crystalcog;

        % Corr matrices
        all_mats=zeros(Nreg,Nreg,length(subs));
        for i=1:length(subs)
            eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/Correlation Matrices/Individual_CMs_afterqc/' num2str(subs(i)) '_CM.mat'')'])
            all_mats(:,:,i)=CM_indiv;
        end

end

% Get connectivity and behavioral data
load /cluster/eryilmaz/users/NFE/Analyses_Connectivity/AllCMs_NFE_N20.mat myCMall
chance_level = [11 19]; % subjects who responded at chance level (thus excluded)
all_mats_NFE = myCMall;
all_mats_NFE(:,:,chance_level) = [];

eval(['load /cluster/eryilmaz/users/NFE/Analyses_Connectivity/BehavData_N20.mat ' testset_outcome ''])
eval(['all_behav_NFE = ' testset_outcome ';'])
all_behav_NFE(chance_level) = [];


no_sub = size(all_mats,3);
no_node = size(all_mats,1);
NTEST = size(all_mats_NFE,3);

%% Feature selection / Model building / Prediction

% Threshold for feature selection
THR = 0.01;

% Initialize predictions
behav_pred_pos = zeros(NTEST,1);
behav_pred_neg = zeros(NTEST,1);
behav_pred_comb = zeros(NTEST,1);

% HCP-EP full set
train_mats = all_mats;
train_vcts = reshape(train_mats,[],size(train_mats,3));
train_behav = all_behav;

% MGH validation set
test_mats = all_mats_NFE;
test_vcts = reshape(test_mats,[],size(test_mats,3));
test_behav = all_behav_NFE;

NTRAIN = size(all_mats,3); % full HCP-EP sample
NTEST = size(all_mats_NFE,3); % full NFE sample

% Correlate all edges with behavior using robust regression
edge_no = size(train_vcts,1);
r_mat = zeros(1, edge_no);
p_mat = zeros(1, edge_no);

for edge_i = 1: edge_no
    [~, stats] = robustfit(train_vcts(edge_i,:)', zscore(train_behav)); 
    cur_t = stats.t(2);
    r_mat(edge_i) = sign(cur_t)*sqrt(cur_t^2/(no_sub-1-2+cur_t^2));
    p_mat(edge_i) = 2*(1-tcdf(abs(cur_t), no_sub-1-2)); 
end

r_mat = reshape(r_mat,no_node,no_node);
p_mat = reshape(p_mat,no_node,no_node);

% Set threshold and define masks
pos_mask = zeros(no_node, no_node);
neg_mask = zeros(no_node, no_node);
comb_mask = zeros(no_node, no_node);

pos_edge = find( ~isnan(r_mat) & r_mat >0 & p_mat < THR);
neg_edge = find( ~isnan(r_mat) & r_mat <0 & p_mat < THR);
comb_edge = find( ~isnan(r_mat) & p_mat < THR );

pos_mask(pos_edge) = 1;
neg_mask(neg_edge) = 1;
comb_mask(comb_edge) = 1;

% Get sum of all edges in training participants (divide by 2 to control for the
% fact that matrices are symmetric)
train_sumpos = zeros(NTRAIN,1);
train_sumneg = zeros(NTRAIN,1);
train_sumcomb = zeros(NTRAIN,1);
for ss = 1:size(train_sumpos)
    train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
    train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    train_sumcomb(ss) = train_sumpos(ss) - train_sumneg(ss);
end

% Build model on training participants
fit_pos = polyfit(train_sumpos, train_behav, 1);
fit_neg = polyfit(train_sumneg, train_behav, 1);
fit_comb = polyfit(train_sumcomb, train_behav, 1);

% Run model on test participants
test_sumpos = zeros(NTEST,1);
test_sumneg = zeros(NTEST,1);
test_sumcomb = zeros(NTEST,1);
for tt=1:NTEST
    test_sumpos(tt) = sum(sum(test_mats(:,:,tt).*pos_mask))/2;
    test_sumneg(tt) = sum(sum(test_mats(:,:,tt).*neg_mask))/2;
    test_sumcomb(tt) = test_sumpos(tt) - test_sumneg(tt);
    behav_pred_pos(tt) = fit_pos(1)*test_sumpos(tt) + fit_pos(2);
    behav_pred_neg(tt) = fit_neg(1)*test_sumneg(tt) + fit_neg(2);
    behav_pred_comb(tt) = fit_comb(1)*test_sumcomb(tt) + fit_comb(2);
end

% Compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos,zscore(test_behav)); 
[R_neg, P_neg] = corr(behav_pred_neg,zscore(test_behav));
[R_comb, P_comb] = corr(behav_pred_comb,zscore(test_behav));

pred_acc_pos = R_pos;
pred_acc_neg = R_neg;
pred_acc_comb = R_comb;
pval_pos = P_pos;
pval_neg = P_neg;
pval_comb = P_comb;

% Print the results
fprintf('POS - Prediction accuracy: %.3f - Pval: %.3f\n', pred_acc_pos, P_pos);
fprintf('NEG - Prediction accuracy: %.3f - Pval: %.3f\n', pred_acc_neg, P_neg);
fprintf('COMB - Prediction accuracy: %.3f - Pval: %.3f\n', pred_acc_comb, P_comb);


