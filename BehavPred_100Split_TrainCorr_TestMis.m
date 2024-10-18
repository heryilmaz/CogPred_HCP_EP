function [R_pos, R_neg, R_comb, pos_mask, neg_mask, comb_mask] = BehavPred_100Split_TrainCorr_TestMis(split,sidx,all_mats,all_behav,THR,no_sub,no_node,no_cor,mymodel)

% Edited by Hamdi Eryilmaz on 8/25/23

% This script uses a part of the connectome-based predictive modeling framework initially developed by
% Finn et al (Copyright 2015 Xilin Shen and Emily Finn) as cited in the paper:

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.

% Edited by Hamdi Eryilmaz to implement a cross-validation framework
% involving training sets comprised of correctly SVM-classified participants and
% test sets comprised of SVM-misclassified participants

%% Get training and test participants for the current split

trainCUTOFF = round(0.7*no_cor); 

% Separate train and test for matrices and behavior
trainidx=split(sidx,1:trainCUTOFF); testidx=split(sidx,trainCUTOFF+1:end);
NTRAIN = length(trainidx); NTEST = length(testidx);

%% Feature selection / Model building / Prediction

% Initialize predictions for positive, negative and combined set
behav_pred_pos = zeros(NTEST,1);
behav_pred_neg = zeros(NTEST,1);
behav_pred_comb = zeros(NTEST,1);

% Get train-test matrices and behavior
train_mats = all_mats(:,:,trainidx);
train_vcts = reshape(train_mats,[],size(train_mats,3));
train_behav = all_behav(trainidx);
test_mats = all_mats(:,:,testidx);
test_vcts = reshape(test_mats,[],size(test_mats,3));
test_behav = all_behav(testidx);

% Correlate all edges with behavior using robust regressionon
edge_no = size(train_vcts,1);
r_mat = zeros(1, edge_no);
p_mat = zeros(1, edge_no);

for edge_i = 1: edge_no;
    [~, stats] = robustfit(train_vcts(edge_i,:)', train_behav);
    cur_t = stats.t(2);
    r_mat(edge_i) = sign(cur_t)*sqrt(cur_t^2/(no_sub-1-2+cur_t^2));
    p_mat(edge_i) = 2*(1-tcdf(abs(cur_t), no_sub-1-2));  %two tailed
end

r_mat = reshape(r_mat,no_node,no_node);
p_mat = reshape(p_mat,no_node,no_node);

% Set threshold and define masks
pos_mask = zeros(no_node, no_node);
neg_mask = zeros(no_node, no_node);
comb_mask = zeros(no_node, no_node);


pos_edge = find( r_mat >0 & p_mat < THR);
neg_edge = find( r_mat <0 & p_mat < THR);
comb_edge = find( p_mat < THR );

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
if strcmp(mymodel,'normal')

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

elseif strcmp(mymodel,'flipped')

    test_sumpos = zeros(NTEST,1);
    test_sumneg = zeros(NTEST,1);
    test_sumcomb = zeros(NTEST,1);
    for tt=1:NTEST
        test_sumpos(tt) = -1*sum(sum(test_mats(:,:,tt).*pos_mask))/2; % sign flipped
        test_sumneg(tt) = -1*sum(sum(test_mats(:,:,tt).*neg_mask))/2; % sign flipped
        test_sumcomb(tt) = test_sumpos(tt) - test_sumneg(tt);
        behav_pred_pos(tt) = fit_pos(1)*test_sumpos(tt) + fit_pos(2);
        behav_pred_neg(tt) = fit_neg(1)*test_sumneg(tt) + fit_neg(2);
        behav_pred_comb(tt) = fit_comb(1)*test_sumcomb(tt) + fit_comb(2);
    end

else
    error('Model must be either normal or flipped!');

end

% Compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos,test_behav);
[R_neg, P_neg] = corr(behav_pred_neg,test_behav);
[R_comb, P_comb] = corr(behav_pred_comb,test_behav);
