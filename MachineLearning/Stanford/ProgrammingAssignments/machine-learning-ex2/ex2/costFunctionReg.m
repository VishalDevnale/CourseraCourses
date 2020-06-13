function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

Hypothesis_OP = sigmoid(X*theta);

P1 = log(Hypothesis_OP);
P2 = log(1 - Hypothesis_OP);

P1 = -y.*P1;
P2 = (1-y).*P2;

Tmp = theta(2:size(X,2)).^2;
J = (sum(P1-P2)/m) + ((lambda/(2*m)) * sum(Tmp)); 

grad1 = ones(size(grad));
grad1(1,1) = 0;

grad = (((Hypothesis_OP - y)' * X)/m) + (((lambda/m)*grad1).* theta)';

% =============================================================

end
