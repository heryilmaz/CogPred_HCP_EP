
% Edited by Hamdi Eryilmaz on 7/10/23 

% This script computes the p-value for the significance of the predictive
% power of a connectome-based model using a permutation test  
% Adapted from Finn et al (Copyright 2015 Xilin Shen and Emily Finn)

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.

% Edited by Hamdi Eryilmaz to run the permutation test in a train-test
% split framework. 

clear;
clc;

myanalysis = 'TOTAL';

% Load corr matrices and behavioral outcomes
switch myanalysis
    case 'TOTAL' 
        load('/cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_output_7_12_all_cogcomp.mat')
    case 'FLUID'
        load('/cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_output_7_12_all_fluid.mat')
    case 'CRYSTAL'
        load('/cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_output_7_12_all_crystal.mat') 
end

all_mats  = all_mats;
all_behav = all_behav;
no_sub = size(all_mats,3);
no_node = size(all_mats,1);

% Get the true prediction for positive and negative set
true_prediction_pos = mean(pred_acc_pos);
true_prediction_neg = mean(pred_acc_neg);
true_prediction_comb = mean(pred_acc_comb);

% number of iterations for permutation testing
no_iterations   = 1000;
prediction_r    = zeros(no_iterations,3);
prediction_r(1,1) = true_prediction_pos;
prediction_r(1,2) = true_prediction_neg;
prediction_r(1,3) = true_prediction_comb;

%tic

% Create a null distribution of the test statistic via random shuffling of data labels   
progressbar
for it=2:no_iterations
    fprintf('\n Performing iteration %d out of %d\n', it, no_iterations);
    new_behav  = all_behav(randperm(no_sub));

    pred_acc_pos_TEMP = zeros(NSPLIT,1); pred_acc_neg_TEMP = zeros(NSPLIT,1); pred_acc_comb_TEMP = zeros(NSPLIT,1); all_pos_mask_TEMP = zeros(no_node,no_node,NSPLIT); all_neg_mask_TEMP = zeros(no_node,no_node,NSPLIT); all_comb_mask_TEMP = zeros(no_node,no_node,NSPLIT);
    parpool(15)
    parfor ii=1:NSPLIT
        [pred_acc_pos_TEMP(ii), pred_acc_neg_TEMP(ii), pred_acc_comb_TEMP(ii), all_pos_mask_TEMP(:,:,ii), all_neg_mask_TEMP(:,:,ii), all_comb_mask_TEMP(:,:,ii)] = BehavPred_100TestTrainSplit(split,ii,all_mats,new_behav,0.01,no_sub,no_node);
    end
    poolobj=gcp('nocreate'); delete(poolobj);

    prediction_r(it,1) = mean(pred_acc_pos_TEMP); 
    prediction_r(it,2) = mean(pred_acc_neg_TEMP);
    prediction_r(it,3) = mean(pred_acc_comb_TEMP);
    
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

