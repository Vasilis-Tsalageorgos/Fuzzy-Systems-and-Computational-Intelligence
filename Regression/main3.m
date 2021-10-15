close all;
clear all;

% load and normalize data

data = csvread('train.csv',1,0);
norm_data = data(:,1:end-1);
norm_data = normalize(norm_data);
data = [norm_data(:,1:end) data(:,end)];

% Evaluation function

Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% best values for number of features and radius' value based on RMSEs
% calculated in main2

features = 13;
r_a = 0.4;

% find the most useful features 

[idx,weights] = relieff(data(:,1:end-1),data(:,end),6); % ranking of features 

% initialize array for speed

metrics_cv = zeros(5,4); % will be used to save metrics of cross validation, 5 because 5-fold

% split data - 80% train, 20% test
        
partition_1 = cvpartition(data(:,end),'KFold',5,'Stratify',true); % 5 folds

a = true; % needed to make sure plots before training are only printed once

for i = 1:partition_1.NumTestSets
            
    % load data from partition_1 

    data_80 = data(training(partition_1,i),:);
    tstData = data(test(partition_1,i),:);

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
    
    % plot MFs before training, once
    
    if a == true
        
        figure();
        plotmf(fis, 'input', 1);
        disp = "Feature 1 MF";
        title(disp);
        
        figure();
        plotmf(fis, 'input', 3);
        disp = "Feature 3 MF";
        title(disp);
        
        figure();
        plotmf(fis, 'input', 5);
        disp = "Feature 5 MF";
        title(disp);
        
        figure();
        plotmf(fis, 'input', 10);
        disp = "Feature 10 MF";
        title(disp);
        
        figure();
        plotmf(fis, 'input', 13);
        disp = "Feature 13 MF";
        title(disp);
        
        a = false;
        
    end
    
    [trnFis,trnError,~,valFis,valError] = anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData); % 100 epochs each
    
    % calculate metrics

    Y = evalfis(tstData(:,1:end-1),valFis);
    RMSE = sqrt(mse(Y,tstData(:,end)));
    R2 = Rsq(Y,tstData(:,end));
    NMSE = 1 - R2; 
    NDEI = sqrt(NMSE);
    
    % save metrics
           
    metrics_cv(i,1) = RMSE;
    metrics_cv(i,2) = NMSE;
    metrics_cv(i,3) = NDEI;
    metrics_cv(i,4) = R2;
    
end

% calculate and save mean average of each metric that has been calculated 5
% times during cross validation
        
RMSE = sum(metrics_cv(:,1))/5; 
NMSE = sum(metrics_cv(:,2))/5; 
NDEI = sum(metrics_cv(:,3))/5; 
R2 = sum(metrics_cv(:,4))/5; 

% array needed for plots

y = zeros(size(Y,1),1);
for i= 1:size(Y,1)
    y(i) = i;
end

% plot MFs after training

figure();
plotmf(valFis, 'input', 1);
disp = "Feature 1 MF";
title(disp);

figure();
plotmf(valFis, 'input', 3);
disp = "Feature 3 MF";
title(disp);

figure();
plotmf(valFis, 'input', 5);
disp = "Feature 5 MF";
title(disp);

figure();
plotmf(valFis, 'input', 10);
disp = "Feature 10 MF";
title(disp);

figure();
plotmf(valFis, 'input', 13);
disp = "Feature 13 MF";
title(disp);

% plot predicted values

figure();
scatter(y,Y); grid on;
xlabel('data'); ylabel('Predicted Values');
title('Predicted Values');

% plot real values

figure();
scatter(y,tstData(:,end)); grid on;
xlabel('data'); ylabel('Real Values');
title('Real Values');

% plot learning curve and error in relation to the number of iterations

figure();
grid on;
plot([trnError valError]);
xlabel('Iterations');
ylabel('Error');
legend('Training error', 'Validation error');
disp = "Learning Curve";
title(disp);