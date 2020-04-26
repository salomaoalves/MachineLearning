%This is a script that compute Simple Linear Regression
%The data set is ex1data1.txt

%======================================LOAD THE DATA=============================================
data = load('data1.txt');
X = data(:, 1); 
y = data(:, 2);
m = length(y);


%======================================PLOT THE DATA=============================================

fprintf('Plotting Data-----------------------------------------------------------------------\n');
plotData(X, y); %two dimensional graphic (use just 2 variables)

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%======================================COST FUNCTION=============================================

fprintf('Cost Function------------------------------------------------------------------------\n');

%Setting the important variables
X = [ones(m, 1), data(:,1)]; %add a column of ones to x
theta = zeros(2, 1); %initialize fitting parameters (we'll use later)
exampleTheta = [-1 ; 2]; %just a example

%With theta = 0
J = costFunction(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

%With theta = -1 e 2
J = costFunction(X, y, exampleTheta);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n\n');

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

%NOTE: I don't use feature normalize here because I have just one input variable

fprintf('Gradient Descent----------------------------------------------------------------------\n');

%Some gradient descent settings
iterations = 1500;
alpha = 0.01;

%Run gradient descent
[thetaGD, J_hist] = gradientDescent(X, y, theta, alpha, iterations);

%Print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', thetaGD);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');

%Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*thetaGD, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%==========================================PREDICT================================================

%NOTE: you can choose change the theta, choosing theta trained by gradient descent or normal equation

fprintf('Making predictions--------------------------------------------------------------------\n');

%Predict values for population sizes of 35,000
pred1 = [1, 3.5] * thetaGD;
fprintf('For population = 35,000, we predict a profit of %f\n',pred1*10000);

%Predict values for population sizes of 70,000    
pred2 = [1, 7] * thetaGD;
fprintf('For population = 70,000, we predict a profit of %f\n',pred2*10000);
