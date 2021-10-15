close all;
clear all;

% load and normalize data

data = csvread('Epileptic-Seizure-Recognition.csv',1,1);
norm_data = data(:,1:end-1);
norm_data = normalize(norm_data);
data = [norm_data(:,1:end) data(:,end)];

% Evaluation function

Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% best values for number of features and radius' value based on RMSEs
% calculated in main2

features = 14;
r_a = 0.3;

% find the most useful features 

[idx,weights] = relieff(data(:,1:end-1),data(:,end),6); % ranking of features

a = true; % needed to make sure plots before training are only printed once

% initialize cv arrays for speed, 5 because 5-fold

PA_1_cv(5) = zeros;
PA_2_cv(5) = zeros;
PA_3_cv(5) = zeros;
PA_4_cv(5) = zeros;
PA_5_cv(5) = zeros;
UA_1_cv(5) = zeros;
UA_2_cv(5) = zeros;
UA_3_cv(5) = zeros;
UA_4_cv(5) = zeros;
UA_5_cv(5) = zeros;
OA_cv(5) = zeros;
K_cv(5) = zeros;

% split data - 80% train, 20% test
        
partition_1 = cvpartition(data(:,end),'KFold',5,'Stratify',true); % 5 folds

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
           
           % Clustering Per Class
           
           [c1,sig1] = subclust(trnData(trnData(:,end)==1,:),r_a);
           [c2,sig2] = subclust(trnData(trnData(:,end)==2,:),r_a);
           [c3,sig3] = subclust(trnData(trnData(:,end)==3,:),r_a);
           [c4,sig4] = subclust(trnData(trnData(:,end)==4,:),r_a);
           [c5,sig5] = subclust(trnData(trnData(:,end)==5,:),r_a);
           num_rules = size(c1,1)+size(c2,1)+size(c3,1)+size(c4,1)+size(c5,1);
           
           % Build FIS From Scratch
           
           fis = newfis('FIS_SC','sugeno');
           
           % Add Input-Output Variables
           
           for m = 1:size(trnData,2)-1
               
               fis = addvar(fis,'input',"in " + m,[0 1]);
               
           end
           
           fis = addvar(fis,'output','out1',[0 1]);
           
           % Add Input Membership Functions

           for m=1:size(trnData,2)-1

               count = 1;

               for n=1:size(c1,1)

                   fis = addmf(fis,'input',m,"MF " + count,'gaussmf',[sig1(m) c1(n,m)]);
                   count = count + 1;

               end

               for n=1:size(c2,1)

                   fis = addmf(fis,'input',m,"MF " + count,'gaussmf',[sig2(m) c2(n,m)]);
                   count = count + 1;

               end
               
               for n=1:size(c3,1)

                   fis = addmf(fis,'input',m,"MF " + count,'gaussmf',[sig3(m) c3(n,m)]);
                   count = count + 1;

               end
               
               for n=1:size(c4,1)

                   fis = addmf(fis,'input',m,"MF " + count,'gaussmf',[sig4(m) c4(n,m)]);
                   count = count + 1;

               end
               
               for n=1:size(c5,1)

                   fis = addmf(fis,'input',m,"MF " + count,'gaussmf',[sig5(m) c5(n,m)]);
                   count = count + 1;

               end

           end
           
           % Add Output Membership Functions
           
           params=[zeros(1,size(c1,1)) 0.25*ones(1,size(c2,1)) 0.5*ones(1,size(c3,1)) 0.75*ones(1,size(c4,1)) ones(1,size(c5,1))];
           
           for m=1:num_rules
               
               fis = addmf(fis,'output',1,"MF " + m,'constant',params(m));
               
           end
           
           % Add FIS Rule Base
           
           ruleList = zeros(num_rules,size(trnData,2));
           
           for m=1:size(ruleList,1)
               
               ruleList(m,:)=m;
               
           end
           
           ruleList = [ruleList ones(num_rules,2)];
           
           fis = addrule(fis,ruleList);
           
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
                plotmf(fis, 'input', 14);
                disp = "Feature 14 MF";
                title(disp);

                a = false;

           end
           
           % Train & Evaluate ANFIS
    
           [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);
           Y=evalfis(tstData(:,1:end-1),valFis);
           Y=round(Y);
           
           % classes are from 1 till 5, need to transform array Y so there are no other
           % values
    
           for m=1:size(Y,1)
        
                if Y(m) < 1
            
                    Y(m) = 1;
            
                elseif Y(m) > 5
            
                    Y(m) = 5;
            
                end
        
           end
           
           % error matrix, 5 classes
    
           error_matrix = confusionmat(tstData(:,end),Y);
           
           % overall accuracy

           N = size(tstData, 1);
           OA_cv(i) = sum(diag(error_matrix))/N;

           % producer's accuracy for each class

           PA_1_cv(i) = error_matrix(1,1)/(sum(error_matrix(1,:)));
           PA_2_cv(i) = error_matrix(2,2)/(sum(error_matrix(2,:)));
           PA_3_cv(i) = error_matrix(3,3)/(sum(error_matrix(3,:))); 
           PA_4_cv(i) = error_matrix(4,4)/(sum(error_matrix(4,:)));
           PA_5_cv(i) = error_matrix(5,5)/(sum(error_matrix(5,:)));
          
           % user's accuracy for each class

           UA_1_cv(i) = error_matrix(1,1)/(sum(error_matrix(:,1)));
           UA_2_cv(i) = error_matrix(2,2)/(sum(error_matrix(:,2)));
           UA_3_cv(i) = error_matrix(3,3)/(sum(error_matrix(:,3)));
           UA_4_cv(i) = error_matrix(4,4)/(sum(error_matrix(:,4)));
           UA_5_cv(i) = error_matrix(5,5)/(sum(error_matrix(:,5)));

           % K 
           
           sum1 = 0;
           
           for j = 1:5
               sum1 = sum1 + sum(error_matrix(:,j)) * sum(error_matrix(j,:));
           end

           K_cv(i) = (N * sum(diag(error_matrix)) - sum1) / (N^2 - sum1);
           
end

% calculate and save mean average of each metric that has been calculated 5
% times during cross validation

sum1 = 0;
sum2 = 0;
sum3 = 0;
sum4 = 0;
sum5 = 0;

for i = 1:5
    sum1 = sum1 + PA_1_cv(i);
    sum2 = sum2 + PA_2_cv(i);
    sum3 = sum3 + PA_3_cv(i);
    sum4 = sum4 + PA_4_cv(i);
    sum5 = sum5 + PA_5_cv(i);
end

PA_1 = sum1/5;
PA_2 = sum2/5;
PA_3 = sum3/5;
PA_4 = sum4/5;
PA_5 = sum5/5;

sum1 = 0;
sum2 = 0;
sum3 = 0;
sum4 = 0;
sum5 = 0;

for i = 1:5
    sum1 = sum1 + UA_1_cv(i);
    sum2 = sum2 + UA_2_cv(i);
    sum3 = sum3 + UA_3_cv(i);
    sum4 = sum4 + UA_4_cv(i);
    sum5 = sum5 + UA_5_cv(i);
end

UA_1 = sum1/5;
UA_2 = sum2/5;
UA_3 = sum3/5;
UA_4 = sum4/5;
UA_5 = sum5/5;

sum1 = 0;
sum2 = 0;

for i = 1:5
    sum1 = sum1 + OA_cv(i);
    sum2 = sum2 + K_cv(i);
end

OA = sum1/5;
K = sum2/5;

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
plotmf(valFis, 'input', 14);
disp = "Feature 14 MF";
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