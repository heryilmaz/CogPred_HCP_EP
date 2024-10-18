
% Created by Hamdi Eryilmaz on 9/14/23

% This script uses SVM to predict binary cognitive outcomes in the HCP-EP data 
% done over 100 iterations (run on parallel pool) and calculates misclassification
% index for each subject.

clear;
clc;
myanalysis = 'TOTAL'; 

%% Get connectivity and behavioral data for the behavioral outcome selected

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
thresh = 0.01;

%% Prediction analysis in 100 iterations 

niter=100;
% Initialize predictions and true labels
pred_svm = zeros(length(subs),niter);
true_label = zeros(length(subs),niter);
% Initialize selected edges to store at each iteration
all_edges_pos = zeros(no_node,no_node,niter);
all_edges_neg = zeros(no_node,no_node,niter);
% Initialize misclassified subject matrix
misclass_mtx = zeros(length(subs),niter);
mis_subs = cell(niter,1);
corr_subs = cell(niter,1);

parpool(12)
parfor_progress(niter); 
parfor iter=1:niter
%for iter=1:niter
    disp(iter)
    [mis_subs{iter,1}, corr_subs{iter,1}, misclass_mtx(:,iter), all_edges_pos(:,:,iter), all_edges_neg(:,:,iter), pred_svm(:,iter), true_label(:,iter)] = ... 
    HCP_MisclassFreq(iter,all_mats,all_behav,no_sub,no_node,thresh);
    parfor_progress;
end
poolobj=gcp('nocreate'); delete(poolobj);
parfor_progress(0); % Clean up


%% Classification accuracy

misclassCount = zeros(numel(mis_subs),1);
correctCount = zeros(numel(corr_subs),1);
class_acc = zeros(numel(mis_subs),1);
for i=1:numel(mis_subs)
    misclassCount(i,1) = length(mis_subs{i});
    correctCount(i,1) = length(corr_subs{i});
    class_acc(i,1) = correctCount(i,1)/(correctCount(i,1)+misclassCount(i,1));
end

avg_class_acc = mean(class_acc);

