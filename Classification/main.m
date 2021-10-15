close all;
clear all;

% load and split data, 60% trn - 20% chk - 20% tst

data = load('haberman.data');
preproc = 1;
[trnData, chkData, tstData] = split_scale(data, preproc);

% clusters' size and rules' number defined by radius' values [0,1]
% need 2 outliers for each of the 2 cases examined:
% class dependent and class independent

r_a = [0.1 0.9];

% train 4 models: 2 class dependent, one for each of the 2 radius' values and
% 2 class independent, one for each of the 2 radius' values respectively

for i = 1:2

    % Clustering Per Class
    
    [c1,sig1] = subclust(trnData(trnData(:,end)==1,:),r_a(i));
    [c2,sig2] = subclust(trnData(trnData(:,end)==2,:),r_a(i));
    num_rules=size(c1,1)+size(c2,1);
    
    % Build FIS From Scratch
    
    fis = newfis('FIS_SC','sugeno');
    
    % Add Input-Output Variables
    
    names_in = {'in1','in2','in3'};
    
    for j=1:size(trnData,2)-1
        
        fis = addvar(fis,'input',names_in{j},[0 1]);
        
    end
    
    fis = addvar(fis,'output','out1',[0 1]);
    
    % Add Input Membership Functions
    
    name = 'MF';
    
    for j=1:size(trnData,2)-1
        
        count = 1;
        
        for m=1:size(c1,1)
            
            fis = addmf(fis,'input',j,"MF " + count,'gaussmf',[sig1(j) c1(m,j)]);
            count = count + 1;
            
        end
        
        for m=1:size(c2,1)
            
            fis = addmf(fis,'input',j,"MF " + count,'gaussmf',[sig2(j) c2(m,j)]);
            count = count + 1;
            
        end
        
    end
    
    % Add Output Membership Functions
    
    params = [zeros(1,size(c1,1)) ones(1,size(c2,1))];
    
    for j=1:num_rules
        
        fis = addmf(fis,'output',1,name,'constant',params(j));
        
    end
    
    % Add FIS Rule Base
    
    ruleList = zeros(num_rules,size(trnData,2));
    
    for j=1:size(ruleList,1)
        
        ruleList(j,:) = j;
        
    end
    
    ruleList = [ruleList ones(num_rules,2)];
    
    fis = addrule(fis,ruleList);
    
    % Train & Evaluate ANFIS
    
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);
    figure();
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    Y=evalfis(tstData(:,1:end-1),valFis);
    Y=round(Y);
    
    % classes are 1 and 2, need to transform array Y so there are no other
    % values
    
    for j=1:size(Y,1)
        
        if Y(j) < 1
            
            Y(j) = 1;
            
        elseif Y(j) > 2
            
            Y(j) = 2;
            
        end
        
    end
    
    diff=tstData(:,end)-Y;
    Acc=(length(diff)-nnz(diff))/length(Y)*100;
    
    % plot class dependent models' MFs for each feature
    
    for j = 1:size(trnData,2)-1
        
        figure();
        plotmf(valFis,'input',j);
        disp = "Class dependent model where radius = " + r_a(i) + " , Feature " + j;
        title(disp);
        
    end
    
    % error matrix, 2 classes
    
    error_matrix = confusionmat(tstData(:,end),Y);
    
    % overall accuracy
    
    N = size(tstData, 1);
    OA = (error_matrix(1,1) + error_matrix(2,2))/N;
    
    % producer's accuracy for each class
    
    PA_1 = error_matrix(1,1)/(error_matrix(1,1) + error_matrix(1,2));
    PA_2 = error_matrix(2,2)/(error_matrix(2,2) + error_matrix(2,1));
    
    % user's accuracy for each class
    
    UA_1 = error_matrix(1,1)/(error_matrix(1,1) + error_matrix(2,1));
    UA_2 = error_matrix(2,2)/(error_matrix(2,2) + error_matrix(1,2));
    
    % K 
    
    K = ( N * (error_matrix(1,1) + error_matrix(2,2)) - ((error_matrix(1,1) + error_matrix(2,1))*(error_matrix(1,1) + error_matrix(1,2)) + (error_matrix(1,2) + error_matrix(2,2))*(error_matrix(2,1) + error_matrix(2,2))))/(N^2 - ((error_matrix(1,1) + error_matrix(2,1))*(error_matrix(1,1) + error_matrix(1,2)) + (error_matrix(1,2) + error_matrix(2,2))*(error_matrix(2,1) + error_matrix(2,2))));  
    
    % save metrics for class dependent models
    
    OA_dp(i,1) = OA;
    PA_dp(i,1) = PA_1;
    PA_dp(i,2) = PA_2;
    UA_dp(i,1) = UA_1;
    UA_dp(i,2) = UA_2;
    K_dp(i,1) = K;
    error_matrix_dp(:,:,i) = error_matrix;
    
    % number of rules
    
    rules_number_dp(i,1) = size(valFis.rule,2);
    
    %Compare with Class-Independent Scatter Partition
    
    fis2 = genfis2(trnData(:,1:end-1),trnData(:,end),r_a(i));
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis2,[100 0 0.01 0.9 1.1],[],chkData);
    figure();
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    Y=evalfis(tstData(:,1:end-1),valFis);
    Y=round(Y);
    
    % classes are 1 and 2, need to transform array Y so there are no other
    % values
    
    for j=1:size(Y,1)
        
        if Y(j) < 1
            
            Y(j) = 1;
            
        elseif Y(j) > 2
            
            Y(j) = 2;
            
        end
        
    end
    
    diff=tstData(:,end)-Y;
    Acc=(length(diff)-nnz(diff))/length(Y)*100;
    
    % plot class independent models' MFs for each feature
    
    for j = 1:size(trnData,2)-1
        
        figure();
        plotmf(valFis,'input',j);
        disp = "Class independent model where radius = " + r_a(i) + " , Feature " + j;
        title(disp);
        
    end
    
    % error matrix, 2 classes
    
    error_matrix = confusionmat(tstData(:,end),Y);
    
    % overall accuracy
    
    N = size(tstData, 1);
    OA = (error_matrix(1,1) + error_matrix(2,2))/N;
    
    % producer's accuracy for each class
    
    PA_1 = error_matrix(1,1)/(error_matrix(1,1) + error_matrix(1,2));
    PA_2 = error_matrix(2,2)/(error_matrix(2,2) + error_matrix(2,1));
    
    % user's accuracy for each class
    
    UA_1 = error_matrix(1,1)/(error_matrix(1,1) + error_matrix(2,1));
    UA_2 = error_matrix(2,2)/(error_matrix(2,2) + error_matrix(1,2));
    
    % K 
    
    K = ( N * (error_matrix(1,1) + error_matrix(2,2)) - ((error_matrix(1,1) + error_matrix(2,1))*(error_matrix(1,1) + error_matrix(1,2)) + (error_matrix(1,2) + error_matrix(2,2))*(error_matrix(2,1) + error_matrix(2,2))))/(N^2 - ((error_matrix(1,1) + error_matrix(2,1))*(error_matrix(1,1) + error_matrix(1,2)) + (error_matrix(1,2) + error_matrix(2,2))*(error_matrix(2,1) + error_matrix(2,2))));  
    
    % save metrics for class independent models
    
    OA_idp(i,1) = OA;
    PA_idp(i,1) = PA_1;
    PA_idp(i,2) = PA_2;
    UA_idp(i,1) = UA_1;
    UA_idp(i,2) = UA_2;
    K_idp(i,1) = K;
    error_matrix_idp(:,:,i) = error_matrix;
    
    % number of rules
    
    rules_number_idp(i,1) = size(valFis.rule,2);
    
end
 
    
    
    
    
    
    
    
    
    
    
    

    