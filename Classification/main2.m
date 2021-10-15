close all;
clear all;

% load and normalize data

data = csvread('Epileptic-Seizure-Recognition.csv',1,1);
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

% choose values for every array element: number of features (0-179) and clusters' radius (0-1)

% number of combinations will be: size1*size2

features_number = [12 13 14];
r_a_values = [0.3 0.4 0.5];

% find the most useful features 

[idx,weights] = relieff(data(:,1:end-1),data(:,end),6); % ranking of features

% need arrays for OA, number of rules, and selected number of features

OA_cv = zeros(5,1); % 5-fold
OA = zeros(3,3);
rules_number = zeros(3,3);
selected_features = zeros(3,3);

% grid search

for i = 1:3 % for each different number of features
    
    for j = 1:3 % for each different radius' value
        
        features = features_number(i);
        r_a = r_a_values(j);
        
        % split data - 80% train, 20% test
        
        partition_1 = cvpartition(data(:,end),'KFold',5,'Stratify',true); % 5 folds   
                
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
           OA_cv(k) = (error_matrix(1,1) + error_matrix(2,2) + error_matrix(3,3) + error_matrix(4,4) + error_matrix(5,5))/N;
           
        end
        
        % calculate and save mean average of OA that has been calculated 5
        % times during cross validation
        
        OA(i,j) = sum(OA_cv)/5;
        
        % save number of rules and selected number of features
        
        rules_number(i,j) = size(valFis.rule,2);
        selected_features(i,j) = features;
        
    end
    
end

% plot overall accuracy in relation to number of rules

figure();
scatter(reshape(OA,1,[]),reshape(rules_number,1,[])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Number of Rules");
title("Overall Accuracy in relation to number of rules ");

% plot overall accuracy in relation to selected number of features

figure();
scatter(reshape(OA,1,[]),reshape(selected_features,1,[])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Selected number of features");
title("Overall Accuracy in relation to selected number of features ");