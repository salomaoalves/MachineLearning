%This is a script that compute Multiple Linear Regression with 2 class
%The data set is data1.txt

%======================================LOAD THE DATA=============================================
data = load('data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);


%======================================PLOT THE DATA=============================================

fprintf('Plotting Data-----------------------------------------------------------------------\n\n');
fprintf('NOTE: + indicating (y = 1) examples and o indicating (y = 0) examples.\n');

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%======================================COST FUNCTION=============================================

fprintf('Cost Function------------------------------------------------------------------------\n\n');

%Setting the important variables
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

%Compute Cost Function
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n\n', cost);

fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%========================================fmincg=================================================

%NOTE: I will use a built-in function (fminunc) to find the optimal parameters theta.

fprintf('fmincg function---------------------------------------------------------------------\n\n');

%Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

%Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n\n', cost);

fprintf('Theta: \n');
fprintf(' %f \n', theta);

fprintf('Program paused. Press enter to continue.\n\n');
pause;


%==========================================PREDICT================================================

fprintf('Making prediction---------------------------------------------------------------------\n\n');

%For a student with score 45 on exam 1 and score 85 on exam 2
x_test = [1 45 85];
prob = sigmoid(x_test * theta);
pred = predict(theta,x_test);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n', prob);
fprintf('So, the answer is: %f\n',pred);
