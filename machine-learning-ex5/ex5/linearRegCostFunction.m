function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% linear reression cost function
% J = 1/(2 * m) * SUM((h(x) - y)^2) + (λ/ (2 * m) * SUM(theta^2))
thetaj = theta;
thetaj(1) = 0;

J = 1 / (2 * m) * sum(((X * theta) - y) .^ 2);
regularized = lambda / (2 * m) * sum(thetaj.^2);
J = J + regularized;

% regularized linear regression gradient
% gradient = 1/m * SUM((h(x) - y) * xj) + ((L/m) * thetaj);
lambdaGrad = (lambda / m) * thetaj;
grad = ((1 / m) * (X' * ((X * theta) - y))) + lambdaGrad;


% =========================================================================

grad = grad(:);

end
