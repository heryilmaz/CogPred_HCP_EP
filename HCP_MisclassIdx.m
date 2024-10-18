function [mis_subs, corr_subs, misclass_mtx, all_edges_pos, all_edges_neg, pred_svm, true_label] = HCP_MisclassFreq(iter,all_mats,all_behav,no_sub,no_node,thresh)

% Edited by Hamdi Eryilmaz on 9/14/23

% This script runs a connectome-based predictive model using SVM to predict
% binary outcomes (classifying low vs. high scorer) in the HCP-EP data and
% calculate misclassification index for individual subjects.
% Adapted from Greene et al. 2022 as cited in the paper. 

% Greene AS, Shen X, Noble S, Horien C, Hahn CA, Arora J, Tokoglu F, Spann MN,
% Carrión CI, Barron DS, Sanacora G, Srihari VH, Woods SW, Scheinost D, Constable RT (2022).
% Brain–phenotype models fail for individuals who defy sample stereotypes. Nature, 609(7925), 109-118.


% Initialize variables
mis_subs = [];
corr_subs = [];
misclass_mtx = zeros(no_sub,1);
pred_svm = zeros(no_sub,1);
true_label = zeros(no_sub,1);

NSUBSMPL = 30; % number of selected subjects in low and high scorer groups
NBACKUP = 28; % number of selected subjects in case there are fewer than NSUBSMPL subjects in each class (rare case)

all_edges_pos = zeros(no_node,no_node);
all_edges_neg = zeros(no_node,no_node);

for mysub=1:no_sub

    % Get the behavioral score for the test participant
    test_behav = all_behav(mysub);
    temp_behav = all_behav;

    % Remove the test participant
    temp_behav(mysub) = [];

    % Select only low and high scorers
    myavg = mean(temp_behav);
    mystd = std(temp_behav);
    cutoff_low = myavg - (mystd/3);
    cutoff_high = myavg + (mystd/3);

    % Find low and high scorers
    low_idx = find(temp_behav<cutoff_low);
    high_idx = find(temp_behav>cutoff_high);

    % Randomly select equal number of subjects for both classes
    if(numel(low_idx)>NSUBSMPL && numel(high_idx)>NSUBSMPL)
        low_idx = low_idx(randperm(length(low_idx),NSUBSMPL));
        high_idx = high_idx(randperm(length(high_idx),NSUBSMPL));
    else
        low_idx = low_idx(randperm(length(low_idx),NBACKUP));
        high_idx = high_idx(randperm(length(high_idx),NBACKUP));
    end

    % Make sure that class sizes are equal
    if numel(low_idx)~=numel(high_idx)
        error('Class sizes must match!');
    end

    train_idx = sort([low_idx; high_idx],'ascend'); % indices of training subjects from each class

    % Label the test subject for prediction
    if test_behav < cutoff_low
        true_label(mysub,1) = -1;
    elseif test_behav > cutoff_high
        true_label(mysub,1) = 1;
    else % no prediction or selected edges for subjects with intermediate or missing scores
        true_label(mysub,1) = 0;
        disp('This test subject is not low or high scorer, skipping this iteration');
        pred_svm(mysub,1) = NaN;
        continue;
    end

    % Obtain Low/High binarization of behavior 
    train_behav = zeros(length(temp_behav),1);
    train_behav(low_idx) = -1;
    train_behav(high_idx) = 1;
    train_behav(train_behav==0) = []; % remove average scorers

    % Get training matrices
    train_mats_tmp = all_mats;
    train_vcts_tmp = reshape(train_mats_tmp,[],size(train_mats_tmp,3));
    train_vcts_tmp(:,mysub) = []; % remove test subject
    train_vcts = train_vcts_tmp(:,train_idx);
    train_mats = reshape(train_vcts,no_node,no_node,size(train_vcts,2));
    NTRAIN = size(train_vcts,2);

    % Correlate all edges with behavior
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
    % Select correlated positive and negative edges
    pos_edge = find( r_mat >0 & p_mat < thresh);
    neg_edge = find( r_mat <0 & p_mat < thresh);
    pos_mask(pos_edge) = 1;
    neg_mask(neg_edge) = 1;

    disp(['Number of positive edges = ' num2str(length(pos_edge)) '; Number of negative edges = ' num2str(length(neg_edge))]);

    % Summary statistic for training subset (network strength)
    train_sumpos = zeros(NTRAIN,1);
    train_sumneg = zeros(NTRAIN,1);
    for ss = 1:size(train_sumpos);
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end

    % Summary stat for test subject
    test_sumpos = sum(sum(all_mats(:,:,mysub).*pos_mask))/2;
    test_sumneg = sum(sum(all_mats(:,:,mysub).*neg_mask))/2;

    % Run SVM 
    train_mdl = fitcsvm(zscore([train_sumpos train_sumneg]),train_behav,'KernelFunction','linear');

    % Predict the left-out test subject's behavior using the SVM model
    pred_svm(mysub,1) = predict(train_mdl,([test_sumpos test_sumneg]-mean([train_sumpos train_sumneg]))./std([train_sumpos train_sumneg]));

    % Determine correctly and incorrectly classified subjects
    if (true_label(mysub,1)==1 && pred_svm(mysub,1)==-1) || (true_label(mysub,1)==-1 && pred_svm(mysub,1)==1)
        mis_subs = [mis_subs; mysub];
    elseif (true_label(mysub,1)==1 && pred_svm(mysub,1)==1) || (true_label(mysub,1)==-1 && pred_svm(mysub,1)==-1)
        corr_subs = [corr_subs; mysub];
    else
        disp('issue assigning misclassified and correctly assigned subjects');
    end

    % Selected edges
    all_edges_pos = all_edges_pos + double(pos_mask);
    all_edges_neg = all_edges_neg + double(neg_mask);

end

% Misclassification matrix
misclass_mtx(mis_subs) = 1; % misclassified subjects=1; correctly classified=0;
misclass_mtx(find(isnan(pred_svm(:,1)))) = NaN; % average scorers=NaN


