
% Created by Hamdi Eryilmaz on 8/8/23

% This script runs connectome-based predictive models cross-validated over
% 100 train/test splits (run on parallel pool) to predict cognitive outcomes in HCP-EP data.
% In this version, the models are trained in a subset of correctly SVM-classified participants 
% and tested in a subset of SVM-misclassified participants

mymodel = 'normal'; % 'normal' or 'flipped' 
% 'flipped' model inverts the sign of the positive set summary score and
% the negative set summary score. 

%% Load Misclassification output for a given outcome

myanalysis = 'TOTAL';

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
mis_idx = (mis_perc/niter); 
avg_misidx = nanmean(mis_idx); 

lowMF_idx = find(mis_idx<=0.4);
highMF_idx = find(mis_idx>=0.6);


%% TRAIN-TEST SPLITS

no_cor= numel(lowMF_idx); no_mis=numel(highMF_idx);

% Create 100 train-test splits 
NO_SUBSET=52; 
NO_TRAIN=34; NO_TEST=18;
NSPLIT=100;
split = zeros(NSPLIT, NO_SUBSET);
for permidx=1:100

    cor_split = randperm(no_cor); 
    mis_split = randperm(no_mis);

    split(permidx,:) = [lowMF_idx(cor_split(1:NO_TRAIN))' highMF_idx(mis_split(1:NO_TEST))'];

end

%% LOOP OVER ALL TRAIN-TEST SPLITS

% Initialize some variables
pred_acc_pos = zeros(NSPLIT,1);
pred_acc_neg = zeros(NSPLIT,1);
pred_acc_comb = zeros(NSPLIT,1);
all_pos_mask = zeros(no_node,no_node,NSPLIT);
all_neg_mask = zeros(no_node,no_node,NSPLIT);
all_comb_mask = zeros(no_node,no_node,NSPLIT);

parpool(12)
parfor i=1:NSPLIT
    disp(i)
    [pred_acc_pos(i), pred_acc_neg(i), pred_acc_comb(i), all_pos_mask(:,:,i), all_neg_mask(:,:,i), all_comb_mask(:,:,i)] = BehavPred_100Split_TrainCCP_TestMCP(split,i,all_mats,all_behav,0.01,no_sub,no_node,no_cor,mymodel);
end
poolobj=gcp('nocreate'); delete(poolobj);


%% FEATURE IMPORTANCE

% Calculate the number of times an edge appears in 100 cross-validation iterations
itercount_pos = zeros(no_node,no_node);
itercount_neg = zeros(no_node,no_node);
itercount_comb = zeros(no_node,no_node);

for jj=1:NSPLIT
pos_mask = reshape(all_pos_mask(:,:,jj),no_node,no_node);
neg_mask = reshape(all_neg_mask(:,:,jj),no_node,no_node);
comb_mask = reshape(all_comb_mask(:,:,jj),no_node,no_node);

    for idx1=1:no_node
        for idx2=1:no_node
            if pos_mask(idx1,idx2)==1
                itercount_pos(idx1,idx2)=itercount_pos(idx1,idx2)+1;
                itercount_comb(idx1,idx2)=itercount_comb(idx1,idx2)+1;
            elseif neg_mask(idx1,idx2)==1
                itercount_neg(idx1,idx2)=itercount_neg(idx1,idx2)+1;
                itercount_comb(idx1,idx2)=itercount_comb(idx1,idx2)+1;               
            end
        end
    end

end

