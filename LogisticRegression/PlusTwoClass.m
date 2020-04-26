%This is a script that compute Multiple Linear Regression with +2 class
%The data set is data2.mat

%======================================LOAD THE DATA=============================================

load('data2.mat'); %training data stored in arrays X, y
m = size(X, 1);

%=======================================COST FUNCTION===========================================

fprintf('Cost Function with regularization---------------------------------------------------\n\n');

%Initialize some variables
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;

[J grad] = costFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n\n', J);

fprintf('Gradients:\n');
fprintf(' %f \n', grad);

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%========================================ONE-VS-ALL==============================================

fprintf('Training One-vs-All Logistic Regression----------------------------------------------\n\n');

%Initialize some variables
input_layer_size = 400; %20x20 Input Images of Digits
num_labels = 10; %10 labels, from 1 to 10
lambda = 0.1;

%Start the training
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%=========================================PREDICT================================================

fprintf('Making prediction---------------------------------------------------------------------\n\n');

pred = predictOneVsAll(all_theta, X);
fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

