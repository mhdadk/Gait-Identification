% load data

h5_path = 'final_data.h5';
x_train = h5read(h5_path,'/xtrain')';
y_train = h5read(h5_path,'/ytrain');
x_test0 = h5read(h5_path,'/prediction_set0')';
x_test1 = h5read(h5_path,'/prediction_set1')';
x_test2 = h5read(h5_path,'/prediction_set2')';
x_test3 = h5read(h5_path,'/prediction_set3')';

x = cat(1,x_train,x_test0,x_test1,x_test2,x_test3);

start_of_x_same_as_x_train = all(x(1:size(x_train,1),:) == x_train,'all');

%% perform k-means clustering to discretize possible observations

num_obsv_per_state = 256;
[idx,C] = kmeans(x,num_obsv_per_state,...
                 'Display','final',...
                 'Distance','sqeuclidean',...
                 'Start','plus',...
                 'Replicates',3,...
                 'MaxIter',200);

%% assign indices 

x_train_idx = idx(1:size(x_train,1),:);
x_test_idx = idx(size(x_train,1) + 1:end,:);

%% obtain the vector-quantized latent vectors

% x_train_encoded
% x_encoded = C(idx,:);

%% split into training and testing sets

% note that it is the indices idx that will be split since
% the hmmestimate function only takes discrete inputs

m = size(x_train_idx,1);
train_split = 0.8;
% test_split = 1 - train_split
x_train2 = x_train_idx(1:round(m*train_split),:);
y_train2 = y_train(1:round(m*train_split),:);
x_test = x_train_idx(round(m*train_split)+1:end,:);
y_test = y_train(round(m*train_split)+1:end,:);

%% HMM

% initial guess

[transition_init,emission_init] = hmmestimate(x_train2,y_train2,...
                                    'Symbols',1:num_obsv_per_state,...
                                    'Statenames',0:3);
                
% test

transition = transition + 1e-10;

y_hat = hmmviterbi(x_test,transition,emission_init);

% convert to range 0:3, transpose, and convert data type

y_hat = y_hat' - 1;
y_hat = cast(y_hat,'like',y_test);

% compute confusion matrix and other stats

CM = confusionmat(y_test,y_hat);
precision = zeros(size(CM,1),1);
recall = zeros(size(CM,1),1);
f1_score = zeros(size(CM,1),1);

for i = 1:size(CM,1)
    precision(i,1) = CM(i,i) / sum(CM(:,i));
    recall(i,1) = CM(i,i) / sum(CM(i,:));
    f1_score(i,1) = 2 * (precision(i,1) * recall(i,1)) / (precision(i,1) + recall(i,1));
end

mean_f1_score = mean(f1_score);