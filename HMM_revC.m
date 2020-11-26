%% load data

h5_path = '60LR.h5';
x_train = h5read(h5_path,'/train_data')';
y_train = h5read(h5_path,'/train_labels');
x_val = h5read(h5_path,'/val_data')';
y_val = h5read(h5_path,'/val_labels');

% concatenate datasets

x = cat(1,x_train,x_val);
y = cat(1,y_train,y_val);

%% K-means clustering
% since observations are continuous and there is practically an infinite
% number of possible observations, need to perform k-means clustering to
% discretize possible observations and make them finite

num_obsv_per_state = 512;
[idx,C] = kmedoids(x,num_obsv_per_state,...
                   'Options',statset('Display','final',...
                                     'MaxIter',1000),...
                    'Algorithm','clara',...
                    'Distance','cosine',...
                    'Start','cluster',...
                    'Replicates',5);

%% obtain the vector-quantized latent vectors

x_encoded = C(idx,:);

%% k-fold cross validation

% since the hmmestimate() function only allows integers as observations,
% then the cluster indices will instead be used as the observations

dataset_size = size(idx,1);
k = 5;
fold_p = 1/k; % fold proportion
fold_size = round(fold_p * dataset_size);
mean_f1_score = 0;

for i = 0:k-1
    
    % get the indices for the testing fold
    
    start_x = fold_size * i + 1;
    end_x = min(fold_size * (i+1), dataset_size);
    
    % testing fold
    
    x_test = idx(start_x:end_x);
    y_test = y(start_x:end_x);
    
    % get the indices for the training fold
    
    train_idx = setdiff(1:dataset_size,start_x:end_x);
    
    % training fold
    
    x_train = idx(train_idx);
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
    
    mean_f1_score = mean_f1_score + mean(f1_score);
end

% final f1 score after cross validation

cv_f1_score = mean_f1_score / k;