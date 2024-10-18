
% Created by Hamdi Eryilmaz on 4/12/24 

% This script runs connectome-based predictive models to predict cognitive outcomes
% in HCP-EP data after virtually lesioning a network (run on parallel pool)

%%
clear;
clc;
myanalysis = 'all_total'; % all_total all_fluid all_crystal 

%% Get connectivity and behavioral data for the behavioral outcome selected

switch myanalysis

    % Total cognition
    case 'all_total'

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
    case 'all_fluid'

        mytable = readtable('/cluster/eryilmaz/users/HCP_Connectivity/Behavioral Data Files/HCP_outcomes.xlsx','Sheet','Sheet1');
        subs = mytable.subject;
        Nreg=333;

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

%% Virtual lesioning

netlabel = {'DMN','VIS','FPN','DAN','VAN','SAL','CON','SM','SML','AUD','CP','RSP'};

% Loop over networks
for ni=1:numel(netlabel)

    network2lesion = netlabel{ni};
    [exmpl_mtx, NRemEdges] = VirtualLesion(all_mats(:,:,1),network2lesion);
    no_node = size(exmpl_mtx,1);

    new_all_mats = zeros(no_node,no_node,no_sub);
    for si=1:no_sub
        [new_all_mats(:,:,si), NRemEdges] = VirtualLesion(all_mats(:,:,si),network2lesion);
    end

    %% Create 100 train-test splits

    NSPLIT=100;
    split = zeros(NSPLIT, no_sub);
    for permidx=1:100
        split(permidx,:) = randperm(no_sub);
    end

    %% Run prediction analysis for each split

    % Initialize variables
    pred_acc_pos = zeros(NSPLIT,1);
    pred_acc_neg = zeros(NSPLIT,1);
    pred_acc_comb = zeros(NSPLIT,1);
    all_pos_mask = zeros(no_node,no_node,NSPLIT);
    all_neg_mask = zeros(no_node,no_node,NSPLIT);
    all_comb_mask = zeros(no_node,no_node,NSPLIT);

    tic
    parpool(12)
    parfor i=1:NSPLIT
        disp(i)
        [pred_acc_pos(i), pred_acc_neg(i), pred_acc_comb(i), all_pos_mask(:,:,i), all_neg_mask(:,:,i), all_comb_mask(:,:,i)] = BehavPred_100TestTrainSplit(split,i,new_all_mats,all_behav,0.01,no_sub,no_node);
    end
    poolobj=gcp('nocreate'); delete(poolobj);
    toc

    %% Feature importance (number of selected edges per network across all iterations)

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

end % of networks
