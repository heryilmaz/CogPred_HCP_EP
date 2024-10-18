function [R_pos, R_neg, R_comb] = BehavPred_100TestTrainSplit_Vec(split,sidx,all_vcts,all_behav,THR,no_sub);

% Edited by Hamdi Eryilmaz on 4/23/24

% This script uses the vectorized version of the correlation matrices as
% input in the connectome-based predictive models.
% The script utilizes the framework initially developed by Finn et al
% (Copyright 2015 Xilin Shen and Emily Finn) as cited in the paper:

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.


%% Get training and test participants for the current split

trainCUTOFF = round(0.7*no_sub); 

% Separate train and test for matrices and behavior
trainidx=split(sidx,1:trainCUTOFF); testidx=split(sidx,trainCUTOFF+1:end);
NTRAIN = length(trainidx); NTEST = length(testidx);

%% Feature selection / Model building / Prediction

behav_pred_pos = zeros(NTEST,1);
behav_pred_neg = zeros(NTEST,1);
behav_pred_comb = zeros(NTEST,1);

% Get train-test matrices and behavior
train_vcts = all_vcts(:,trainidx);
train_behav = all_behav(trainidx);
test_vcts = all_vcts(:,testidx);
test_behav = all_behav(testidx);

% Correlate all edges with behavior using robust regression
edge_no = size(train_vcts,1);
r_mat = zeros(1, edge_no);
p_mat = zeros(1, edge_no);

for edge_i = 1: edge_no;
    [~, stats] = robustfit(train_vcts(edge_i,:)', train_behav);
    cur_t = stats.t(2);
    r_mat(edge_i) = sign(cur_t)*sqrt(cur_t^2/(no_sub-1-2+cur_t^2));
    p_mat(edge_i) = 2*(1-tcdf(abs(cur_t), no_sub-1-2));  %two tailed
end

% Set threshold and define masks
pos_mask = zeros(1, length(r_mat));
neg_mask = zeros(1, length(r_mat));
comb_mask = zeros(1, length(r_mat));

pos_edge = find( r_mat >0 & p_mat < THR);
neg_edge = find( r_mat <0 & p_mat < THR);
comb_edge = find( p_mat < THR );

pos_mask(pos_edge) = 1;
neg_mask(neg_edge) = 1;
comb_mask(comb_edge) = 1;

% Get sum of all edges in training participants
train_sumpos = zeros(NTRAIN,1);
train_sumneg = zeros(NTRAIN,1);
train_sumcomb = zeros(NTRAIN,1);
for ss = 1:size(train_sumpos,1)
    train_sumpos(ss) = sum(train_vcts(:,ss).*pos_mask');
    train_sumneg(ss) = sum(train_vcts(:,ss).*neg_mask');
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
    test_sumpos(tt) = sum(test_vcts(:,tt).*pos_mask');
    test_sumneg(tt) = sum(test_vcts(:,tt).*neg_mask');
    test_sumcomb(tt) = test_sumpos(tt) - test_sumneg(tt);
    behav_pred_pos(tt) = fit_pos(1)*test_sumpos(tt) + fit_pos(2);
    behav_pred_neg(tt) = fit_neg(1)*test_sumneg(tt) + fit_neg(2);
    behav_pred_comb(tt) = fit_comb(1)*test_sumcomb(tt) + fit_comb(2);
end

% Compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos,test_behav);
[R_neg, P_neg] = corr(behav_pred_neg,test_behav);
[R_comb, P_comb] = corr(behav_pred_comb,test_behav);


