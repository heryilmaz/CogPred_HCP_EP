
% Edited by Hamdi Eryilmaz on 11/20/23

% This script computes the p-value for the significance of the predictive
% power of a connectome-based model using a permutation test  
% Adapted from Finn et al (Copyright 2015 Xilin Shen and Emily Finn)

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.

% Edited by Hamdi Eryilmaz to run the permutation test for prediction of
% cognitive performance in the independent MGH sample. 

clear;
clc;

myanalysis = 'fluidcog'; 
myNFEoutcome = 'avgACC'; % average accuracy (outcome variable in the external validation set)

% Get connectivity and behavioral data
eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_' myanalysis '_NFE_' myNFEoutcome '.mat'')'])
all_mats  = all_mats;
all_behav = all_behav;
no_sub = size(all_mats,3);
no_node = size(all_mats,1);

% Get the true prediction for positive and negative set
true_prediction_pos = pred_acc_pos;
true_prediction_neg = pred_acc_neg;
true_prediction_comb = pred_acc_comb;

% Number of iterations for permutation
no_iterations   = 1000;
prediction_r    = zeros(no_iterations,3);
prediction_r(1,1) = true_prediction_pos;
prediction_r(1,2) = true_prediction_neg;
prediction_r(1,3) = true_prediction_comb;

THR=0.01; NTEST=length(all_behav_NFE);
% Create a null distribution of the test statistic via random shuffling of data labels   
progressbar
for it=2:no_iterations
    fprintf('\n Performing iteration %d out of %d\n', it, no_iterations);
    new_behav  = all_behav(randperm(no_sub));

    thresh=THR;

    behav_pred_pos = zeros(NTEST,1);
    behav_pred_neg = zeros(NTEST,1);
    behav_pred_comb = zeros(NTEST,1);

    % TRAIN-TEST matrices and behavior
    % HCP-EP full set
    train_mats = all_mats;
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    train_behav = new_behav; % shuffled outcomes
    % NFE validation set
    test_mats = all_mats_NFE;
    test_vcts = reshape(test_mats,[],size(test_mats,3));
    test_behav = all_behav_NFE;

    NTRAIN = size(all_mats,3); % training is the full HCP-EP sample
    NTEST = size(all_mats_NFE,3); % test is the full NFE sample

    % Correlate all edges with behavior using robust regression
    edge_no = size(train_vcts,1);
    r_mat = zeros(1, edge_no);
    p_mat = zeros(1, edge_no);

    for edge_i = 1: edge_no;
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

    pos_edge = find( r_mat >0 & p_mat < thresh);
    neg_edge = find( r_mat <0 & p_mat < thresh);
    comb_edge = find( p_mat < thresh );

    pos_mask(pos_edge) = 1;
    neg_mask(neg_edge) = 1;
    comb_mask(comb_edge) = 1;

    % Get sum of all edges in training participants (divide by 2 to control for the
    % fact that matrices are symmetric)    
    train_sumpos = zeros(NTRAIN,1);
    train_sumneg = zeros(NTRAIN,1);
    train_sumcomb = zeros(NTRAIN,1);
    for ss = 1:size(train_sumpos);
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

    pred_acc_pos_TEMP = R_pos;
    pred_acc_neg_TEMP = R_neg;
    pred_acc_comb_TEMP = R_comb;

    prediction_r(it,1) = pred_acc_pos_TEMP;
    prediction_r(it,2) = pred_acc_neg_TEMP;
    prediction_r(it,3) = pred_acc_comb_TEMP;

    progressbar(it/no_iterations)
end

sorted_prediction_r_pos = sort(prediction_r(:,1),'descend');
position_pos            = find(sorted_prediction_r_pos==true_prediction_pos);
pval_pos                = position_pos(1)/no_iterations;

sorted_prediction_r_neg = sort(prediction_r(:,2),'descend');
position_neg            = find(sorted_prediction_r_neg==true_prediction_neg);
pval_neg                = position_neg(1)/no_iterations;

sorted_prediction_r_comb = sort(prediction_r(:,3),'descend');
position_comb            = find(sorted_prediction_r_comb==true_prediction_comb);
pval_comb                = position_comb(1)/no_iterations;

