
% Created by Hamdi Eryilmaz on 4/22/24

% This script runs a permutation test to obtain a null distribution
% for the degree to which prediction accuracy changes after randomly removing a
% number of edges (equivalent to the number of edges when removing a given network)

% Steps:

% 1. Define target performance as performance after removing edges from a given network (m edges removed)
% 2. For each permutation k:
%   a. Randomly remove m edges from across the whole brain
%   b. Estimate new performance
%   c. Define null stat for permutation k as new performance
% 3. Compare target performance to the null distribution to calculate the p-value

%% Select outcome variable, the feature set and the network to lesion

myanalysis = 'TOTAL';
myset = 'positive';

% Number of iterations for permutation testing
no_iterations  = 500; 

netlabel = {'DMN','VIS','FPN','DAN','VAN','SAL','CON','SM','SML','AUD','CP','RSP'};

%% First load the CPM output for a given outcome variable

switch myanalysis
    case 'TOTAL'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_output_7_12_all_cogcomp.mat pred_acc_pos pred_acc_neg pred_acc_comb

    case 'FLUID'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_output_7_12_all_fluid.mat pred_acc_pos pred_acc_neg pred_acc_comb

    case 'CRYSTAL'
        load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/HCP_output_7_12_all_crystal.mat pred_acc_pos pred_acc_neg pred_acc_comb
end

% True prediction accuracy
true_pred_acc_pos = mean(pred_acc_pos);
true_pred_acc_neg = mean(pred_acc_neg);
true_pred_acc_comb = mean(pred_acc_comb);

clear pred_acc_pos pred_acc_neg pred_acc_comb

%% Get the true change in prediction accuracy after lesioning the given network
SurrDeltaPredAcc = zeros(numel(netlabel),no_iterations); % 12x500 matrix
TrueDeltaPredAcc = zeros(numel(netlabel),1);

progressbar
for nw=1:numel(netlabel) 
    network2lesion = netlabel{nw};

    mycmd = ['load /cluster/eryilmaz/users/HCP_Connectivity/CPM_Jun23/VirtualLesion_' myanalysis '_' network2lesion '.mat'];
    eval(mycmd);

    if strcmp(myset,'positive')
        TrueDeltaPredAcc(nw,1) = mean(pred_acc_pos) - true_pred_acc_pos;
    elseif strcmp(myset,'negative')
        TrueDeltaPredAcc(nw,1) = mean(pred_acc_neg) - true_pred_acc_neg;
    elseif strcmp(myset,'combined')
        TrueDeltaPredAcc(nw,1) = mean(pred_acc_comb) - true_pred_acc_comb;
    end

    clear pred_acc_pos pred_acc_neg pred_acc_comb

    %% Permutation

    % Change in prediction accuracy obtained after removing random edges
    SurrDeltaPredAcc(nw,1) = TrueDeltaPredAcc(nw,1); % the first one is the true delta prediction accuracy

    % Create estimate distribution of the test statistic by randomly removing features from the entire cortex
    for it=2:no_iterations
        fprintf('\n Performing iteration %d out of %d\n', it, no_iterations);

        % Get original connectivity and behavioral data for the specified cognitive outcome
        switch myanalysis

            % Total cognition
            case 'TOTAL'

                mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
                subs = mytable.subject;
                Nreg=333;

                % Outcome variable
                all_behav=mytable.NIH_cogcomp;

                % Corr matrices
                all_mats=zeros(Nreg,Nreg,length(subs));
                for i=1:length(subs)
                    eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/Correlation Matrices/Individual_CMs_afterqc/' num2str(subs(i)) '_CM.mat'')'])
                    all_mats(:,:,i)=CM_indiv;
                end

            % Fluid cognition
            case 'FLUID'

                mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
                subs = mytable.subject;
                Nreg=333;
-
                % Outcome variable
                all_behav=mytable.NIH_fluidcog;

                % Corr matrices
                all_mats=zeros(Nreg,Nreg,length(subs));
                for i=1:length(subs)
                    eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/Correlation Matrices/Individual_CMs_afterqc/' num2str(subs(i)) '_CM.mat'')'])
                    all_mats(:,:,i)=CM_indiv;
                end

            % Crystallized cognition
            case 'CRYSTAL'

                mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
                subs = mytable.subject;
                Nreg=333;

                % Outcome variable
                all_behav=mytable.NIH_crystalcog;

                % Corr matrices
                all_mats=zeros(Nreg,Nreg,length(subs));
                for i=1:length(subs)
                    eval(['load(''/cluster/eryilmaz/users/HCP_Connectivity/Correlation Matrices/Individual_CMs_afterqc/' num2str(subs(i)) '_CM.mat'')'])
                    all_mats(:,:,i)=CM_indiv;
                end

        end

        no_sub = size(all_mats,3);
        no_node = size(all_mats,1);

        %% Random Lesioning

        avg_mat = mean(all_mats,3);
        % Determine the total number of edges removed when virtually lesioning this network
        [test_les_mat, NRemEdges] = VirtualLesion(avg_mat,network2lesion);
        no_node_les = size(test_les_mat,1);

        % Now randomly remove NRemEdges edges from the raw matrices (do this in vectorized versions of the matrices)
        % First vectorize all matrices
        lowtrind = tril(true(size(avg_mat)),-1);
        all_vcts = zeros(nnz(lowtrind),size(all_mats,3));
        for vi=1:size(all_vcts,2)
            mymtx = all_mats(:,:,vi);
            all_vcts(:,vi) = VectorizeMatrix(mymtx,'one');
        end
        % Lesion random edges from all vectors (the same random edges for all subjects)
        new_vcts = all_vcts;
        rand_arr_idx = randperm(size(all_vcts,1)); % randomized indices of the vectorized version of the 333x333 matrix
        rand_feat_idx = rand_arr_idx(1:NRemEdges); % indices of the edges to be removed
        new_vcts(rand_feat_idx',:) = [];

        all_vcts = new_vcts; % lesioned matrix to be used in the CPM

        %% TRAIN-TEST SPLITS

        % Create 100 train-test splits 
        NSPLIT=100;
        split = zeros(NSPLIT, no_sub);
        for permidx=1:100
            split(permidx,:) = randperm(no_sub);
        end

        %% Run predictions for all cross-validation iterations

        % Initialize variables
        pred_acc_pos = zeros(NSPLIT,1);
        pred_acc_neg = zeros(NSPLIT,1);
        pred_acc_comb = zeros(NSPLIT,1);
   
        parpool(12)
        parfor i=1:NSPLIT
            disp(i)
            [pred_acc_pos(i), pred_acc_neg(i), pred_acc_comb(i)] = BehavPred_100TestTrainSplit_Vec(split,i,all_vcts,all_behav,0.01,no_sub);
        end
        poolobj=gcp('nocreate'); delete(poolobj);

        % Record the change in prediction accuracy for this permutation
        if strcmp(myset,'positive')
            MyDeltaPredAcc = mean(pred_acc_pos) - true_pred_acc_pos;
        elseif strcmp(myset,'negative')
            MyDeltaPredAcc = mean(pred_acc_neg) - true_pred_acc_neg;
        elseif strcmp(myset,'combined')
            MyDeltaPredAcc = mean(pred_acc_comb) - true_pred_acc_comb;
        end

        SurrDeltaPredAcc(nw,it) = MyDeltaPredAcc;

    end % of permutations

    progressbar(nw/numel(netlabel));
end % of networks


%% Significance testing using FDR

pval_virtles = zeros(size(SurrDeltaPredAcc,1),1);
for si=1:size(SurrDeltaPredAcc,1)
    sorted_SurrDeltaPredAcc= sort(reshape(SurrDeltaPredAcc(si,:),1,[]));
    position = find(sorted_SurrDeltaPredAcc==TrueDeltaPredAcc(si,1));
    pval_virtles(si,1) = position(1)/no_iterations;
end






