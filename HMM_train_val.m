%% load data

h5_path = 'data/data_latent.h5';
x_train = h5read(h5_path,'/xtrain')';
y = h5read(h5_path,'/ytrain');
x_unlabelled0 = h5read(h5_path,'/prediction_set0')';
x_unlabelled1 = h5read(h5_path,'/prediction_set1')';
x_unlabelled2 = h5read(h5_path,'/prediction_set2')';
x_unlabelled3 = h5read(h5_path,'/prediction_set3')';

x = cat(1,x_train,x_unlabelled0,x_unlabelled1,x_unlabelled2,x_unlabelled3);

%% K-means clustering
% since observations are continuous and there is practically an infinite
% number of possible observations, need to perform k-means clustering to
% discretize possible observations and make them finite

num_obsv_per_state = 2048;
[idx,C] = kmeans(x,num_obsv_per_state,...
                 'Display','final',...
                 'Distance','sqeuclidean',...
                 'Start','plus',...
                 'Replicates',3,...
                 'MaxIter',200);

%% obtain indices

idx_labelled = idx(1:42574,:);
idx_unlabelled0 = idx(42575:52072,:);
idx_unlabelled1 = idx(52073:64342,:);
idx_unlabelled2 = idx(64343:77282,:);
idx_unlabelled3 = idx(77283:88612,:);

save('data/codes.mat','idx_labelled','idx_unlabelled0','idx_unlabelled1','idx_unlabelled2','idx_unlabelled3');

%% obtain the vector-quantized latent vectors

% x_encoded = C(idx,:);

%% k-fold cross validation

% since the hmmestimate() function only allows integers as observations,
% then the cluster indices will instead be used as the observations

dataset_size = size(idx_labelled,1);
k = 5;
fold_p = 1/k; % fold proportion
fold_size = round(fold_p * dataset_size);
cv_f1_score = zeros(k,1);

for i = 0:k-1
    
    % get the indices for the testing fold
    
    start_x = fold_size * i + 1;
    end_x = min(fold_size * (i+1), dataset_size);
    
    % testing fold
    
    x_test = idx_labelled(start_x:end_x);
    y_test = y(start_x:end_x);
    
    % get the indices for the training fold
    
    train_idx = setdiff(1:dataset_size,start_x:end_x);
    
    % training fold
    
    x_train = idx_labelled(train_idx);
    y_train = y(train_idx);
    
    % train HMM
    
    [transition,emission] = hmmestimate(x_train,y_train,...
                                        'Symbols',1:num_obsv_per_state,...
                                        'Statenames',0:3);
                                    
    % need to account for zero emission probabilities
    
    emission = emission + 1e-10;
    
    % test HMM
    
    y_hat = hmmviterbi(x_test,transition,emission);
    
    % convert to range 0:3, transpose, and convert data type for the
    % confusionmat() function
    
    y_hat = y_hat' - 1;
    y_hat = cast(y_hat,'like',y_test);
    
    % compute confusion matrix and other stats
    
    CM = confusionmat(y_test,y_hat);
    precision = zeros(size(CM,1),1);
    recall = zeros(size(CM,1),1);
    f1_score = zeros(size(CM,1),1);
    for j = 1:size(CM,1)
        precision(j,1) = CM(j,j) / sum(CM(:,j));
        recall(j,1) = CM(j,j) / sum(CM(j,:));
        f1_score(j,1) = 2 * (precision(j,1) * recall(j,1)) / (precision(j,1) + recall(j,1));
    end
    
    % average F1 score for all classes
    
    cv_f1_score(i+1) = mean(f1_score);
end

% final f1 score after cross validation

mean_cv_f1_score = mean(cv_f1_score);