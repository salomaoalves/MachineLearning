%This is a script that compute Multiple Linear Regression
%The data set is ex1data2.txt

%======================================LOAD THE DATA=============================================
data = load('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);


%======================================PLOT THE DATA=============================================

fprintf('Plotting Data------------------------------------------------------------------------\n');

fprintf('First plot:\n');
plotData(X(:,1), y);

fprintf('Second plot:\n');
plotData(X(:,2), y);
title('Second Plot');

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%======================================COST FUNCTION=============================================

fprintf('Cost Function------------------------------------------------------------------------\n');

%Setting the important variables
X = [ones(m, 1), X]; %add a column of ones to x
theta = zeros(3, 1); %initialize fitting parameters (we'll use later)

%With theta = 0
J = costFunction(X, y, theta);
fprintf('With theta = [0 ; 0 ; 0]\nCost computed = %f\n\n', J);

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%======================================NORMAL EQUATION============================================

fprintf('Normal Equation-----------------------------------------------------------------------\n');

%Calculate the parameters from the normal equation
thetaNE = normalEquation(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', thetaNE);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%=====================================GRADIENT DESCENT============================================

%NOTE: Now, I will use feature normalize

%Do the feature normalize
[X mu sigma] = featureNormalize(X(:,2:3));

%Add again a column of ones to x
X = [ones(m, 1) X];

fprintf('Gradient Descent----------------------------------------------------------------------\n');

%Some gradient descent settings
alpha = 0.01;
num_iters = 400;

%Run gradient descent
[thetaGD, J_hist] = gradientDescent(X, y, theta, alpha, iterations);

%Print theta to screen
fprintf('Theta found by gradient descent: \n');
fprintf(' %f \n', thetaGD);
fprintf('\n');

%Plot the convergence graph
figure;
plot(1:numel(J_hist), J_hist, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%==========================================PREDICT================================================

%Using Normal Equation

fprintf('Predict using normal equation---------------------------------------------------------\n');

% Estimate the price of a 1650 sq-ft, 3 br house
x_teste = [1 1650 3];
price = x_teste*thetaNE;
fprintf('Predicted price of a 1650 sq-ft, 3 br house : $%f\n\n', price);

fprintf('Program paused. Press enter to continue.\n\n');
pause;

%Using Gradient Descent

fprintf('Predict using gradient descent--------------------------------------------------------\n');

%Estimate the price of a 1650 sq-ft, 3 br house
x1 = (1650-mu(1)) / sigma(1);
x2 = (3-mu(2)) / sigma(2);
x_teste = [1 x1 x2];
price = x_teste*thetaGD;
fprintf('Predicted price of a 1650 sq-ft, 3 br house : $%f\n\n', price);
