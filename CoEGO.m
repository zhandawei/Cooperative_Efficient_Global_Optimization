% cooperative efficient global optimizatin algorithm
% Dawei Zhan, Jintao Wu, Huanlai Xing, Tianrui Li, A cooperative approach
% to efficient global optimization. Journal of Global Optimization, 2023,
% Accepted.
clearvars;clc;close all;
% function name
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% box constraints
lower_bound = -5*ones(1,num_vari);
upper_bound = 5*ones(1,num_vari);
% number of initial samples
num_initial = 200;
% maximum number of objective evaluations
max_evaluation = 1000;
% number of variables of the sub-problem
sub_vari = 1;
% initial samples using Latin hypercubic sampling
sample_x = lhsdesign(num_initial,num_vari).*(upper_bound - lower_bound)+lower_bound;
sample_y= feval(fun_name,sample_x);
iteration = 1;
evaluation =  size(sample_x,1);
[fmin,ind]= min(sample_y);
best_x = sample_x(ind,:);
fmin_record(iteration,1) = fmin;
fprintf('CoEGO on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
best_theta = ones(1,num_vari);
while evaluation < max_evaluation
    rand_ind = randperm(num_vari);
    unsovled = min(max_evaluation-evaluation,num_vari/sub_vari);
    for ii = 1:unsovled
        index = rand_ind((ii-1)*sub_vari+1: ii*sub_vari);
        % cooperative training
        kriging_model = Kriging_Train_Cooperative(sample_x,sample_y,lower_bound,upper_bound,best_theta,0.01*ones(1,sub_vari),100*ones(1,sub_vari),index);
        best_theta(index) = kriging_model.theta(index);
        % cooperative infill sampling
        [optimal_x,max_EI]= Optimizer_GA(@(x)-Infill_CoEI(x,kriging_model,fmin,best_x,index),sub_vari,lower_bound(:,index),upper_bound(:,index),4*sub_vari,25);
        infill_x  = best_x;
        infill_x(:,index) = optimal_x;
        infill_y = feval(fun_name,infill_x);
        iteration = iteration + 1;
        sample_x = [sample_x;infill_x];
        sample_y = [sample_y;infill_y];
        [fmin,ind]= min(sample_y);
        best_x = sample_x(ind,:);
        fmin_record(iteration,1) = fmin;
        evaluation = evaluation + size(infill_x,1);
        fprintf('CoEGO on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
    end
end


