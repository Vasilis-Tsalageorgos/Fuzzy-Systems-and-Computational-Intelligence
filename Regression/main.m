close all;
clear all;

% load and split data

data = load('airfoil_self_noise.dat');
preproc = 1;
[trnData, chkData, tstData] = split_scale(data, preproc);
Perf = zeros(4,4); % 4 models + 4 metrics

% Evaluation function

Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% FIS with grid partition

fis(1)=genfis1(trnData,2,'gbellmf','constant'); % constant for Singleton
fis(2)=genfis1(trnData,3,'gbellmf','constant'); 
fis(3)=genfis1(trnData,2,'gbellmf','linear'); % linear for Polynomial
fis(4)=genfis1(trnData,3,'gbellmf','linear');

% training

for i=1:4
    
    % 100 epochs each
    
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis(i),[100 0 0.01 0.9 1.1],[],chkData);
    
    % MFs plots
    
    for j=1:5 % 5 features
        figure();
        plotmf(valFis, 'input', j);
        disp = "Model " + i + " Feature " + j;
        title(disp);
    end
    
    % learning curve
    
    figure();
    grid on;
    plot([trnError valError]);
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    disp = "Model " + i + " Learning Curve ";
    title(disp);
    
    % predict test data
    
    Y = evalfis(tstData(:,1:end-1),valFis); 
    
    % calculate metrics
    
    R2 = Rsq(Y,tstData(:,end)); 
    RMSE = sqrt(mse(Y,tstData(:,end)));
    NMSE = 1 - R2; 
    NDEI = sqrt(NMSE);
    Perf(:,i) = [R2; RMSE; NMSE; NDEI];
    
    % error prediction
    
    predict_error = tstData(:,end) - Y; 
    figure();
    plot(predict_error);
    grid on;
    xlabel('input');ylabel('Error');
    disp = "Model " + i + " Prediction Error ";
    title(disp);
    
end

% results table

varnames={'Model1', 'Model2', 'Model3', 'Model4'};
rownames={'Rsquared' , 'RMSE' , 'NMSE' , 'NDEI'};
Perf = array2table(Perf,'VariableNames',varnames,'RowNames',rownames);