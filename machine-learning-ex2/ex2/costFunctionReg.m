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

h = sigmoid(X * theta);
logh = log(h);
lognegh = log(1-h);

J1 = (1/m) *( -1*y(1) * logh(1) - (1-y(1)) * lognegh(1) );
J = (1/m) *( -1*y(2:end,:)' * logh(2:end,:) - (1-y)(2:end,:)' * lognegh(2:end,:) )+ lambda/(2*m) * theta(2:end,:)' * theta(2:end,:) ;
J += J1;

grad1 = (1/m) * sum(h - y) ;
grad = (1/m) * (h - y)' * X(:,2:end);
grad += lambda/m * theta(2:end,:)';
grad = [grad1 grad];

theta = theta - grad;





% =============================================================

end
