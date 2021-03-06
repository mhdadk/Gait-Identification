% load HMM parameters and data

load('data/codes.mat');
h5_path = 'data/data_latent.h5';
y = h5read(h5_path,'/ytrain');
num_obsv_per_state = 2048;

% train HMM on entire labelled dataset

[transition,emission] = hmmestimate(idx_labelled,y,...
                                    'Symbols',1:num_obsv_per_state,...
                                    'Statenames',0:3);

% need to account for zero emission probabilities

emission = emission + 1e-10;

% test HMM on unlabelled dataset

final_pred0 = hmmviterbi(idx_unlabelled0,transition,emission);
final_pred1 = hmmviterbi(idx_unlabelled1,transition,emission);
final_pred2 = hmmviterbi(idx_unlabelled2,transition,emission);
final_pred3 = hmmviterbi(idx_unlabelled3,transition,emission);
