close all;
clear all;

% load and normalize data

data = csvread('train.csv',1,0);
norm_data = data(:,1:end-1);
norm_data = normalize(norm_data);
data = [norm_data(:,1:end) data(:,end)];

% Evaluation function

Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% create 2 arrays: the one containing the numbers of features that will be
% tested - used for training each time, the other containing the radius' values respectively

% choose sizes of the arrays - number of elements for each: how many different numbers of features do I
% want to test - train, and how many different radius' values for each number of
% features respectively

% choose values for every array element: number of features (0-81) and clusters' radius (0-1)

% number of combinations will be: size1*size2

features_number = [12 13 14];
r_a_values = [0.3 0.4 0.5];

metrics = zeros(3,3,4); % will be used to save metrics (4) for each combination 

% find the most useful features 

[idx,weights] = relieff(data(:,1:end-1),data(:,end),6); % ranking of features 

% need arrays for RMSE, number of rules, and selected number of features

rmse = zeros(3,3);
rules_number = zeros(3,3);
selected_features = zeros(3,3);

% grid search

for i = 1:3 % for each different number of features
    
    for j = 1:3 % for each different radius' value
        
        features = features_number(i);
        r_a = r_a_values(j);
       
        % split data - 80% train, 20% test
        
        partition_1 = cvpartition(data(:,end),'KFold',5,'Stratify',true); % 5 folds   
        
        metrics_cv = zeros(5,4); % will be used to save metrics (4) of cross validation, 5 because 5-fold 
        
        for k = 1:partition_1.NumTestSets
            
           % load data from partition_1 
           
           data_80 = data(training(partition_1,k),:);
           tstData = data(test(partition_1,k),:);
           
           % split 80% of data, 75% for training, 25% for evaluation
           
           partition_2 = cvpartition(data_80(:,end),'KFold',4,'Stratify',true);
           trnData = data_80(training(partition_2,2),:);
           chkData = data_80(test(partition_2,2),:);
           
           % save the most useful features
           
           trnData = [trnData(:, idx(1:features)) trnData(:,end)];
           chkData = [chkData(:, idx(1:features)) chkData(:,end)];
           tstData = [tstData(:, idx(1:features)) tstData(:,end)];
           
           % FIS with subtractive clustering 
           
           fis = genfis2(trnData(:,1:end-1), trnData(:,end), r_a);
           
           [trnFis,trnError,~,valFis,valError] = anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData); % 100 epochs each
           
           % calculate metrics
           
           Y = evalfis(tstData(:,1:end-1),valFis);
           RMSE = sqrt(mse(Y,tstData(:,end)));
           R2 = Rsq(Y,tstData(:,end));
           NMSE = 1 - R2; 
           NDEI = sqrt(NMSE);
           
           % save metrics
           
           metrics_cv(k,1) = RMSE;
           metrics_cv(k,2) = NMSE;
           metrics_cv(k,3) = NDEI;
           metrics_cv(k,4) = R2;
           
        end
        
        % calculate and save mean average of each metric that has been calculated 5
        % times during cross validation
        
        metrics(i,j,1) = sum(metrics_cv(:,1))/5; % RMSE
        metrics(i,j,2) = sum(metrics_cv(:,2))/5; % NMSE
        metrics(i,j,3) = sum(metrics_cv(:,3))/5; % NDEI
        metrics(i,j,4) = sum(metrics_cv(:,4))/5; % R2
        
        % save RMSE, number of rules, and selected number of features
        
        rmse(i,j) = metrics(i,j,1);
        rules_number(i,j) = size(valFis.rule,2);
        selected_features(i,j) = features;

    end
end

% plot RMSE in relation to number of rules

figure();
scatter(reshape(rmse,1,[]),reshape(rules_number,1,[])); grid on;
xlabel("RMSE"); 
ylabel("Number of rules");
title("RMSE in relation to number of rules ");

% plot RMSE in relation to selected number of features

figure();
scatter(reshape(rmse,1,[]),reshape(selected_features,1,[])); grid on;
xlabel("RMSE"); 
ylabel("Selected number of features");
title("RMSE in relation to selected number of features");