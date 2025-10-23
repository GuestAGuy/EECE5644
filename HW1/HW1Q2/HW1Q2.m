%% Question 2 Part A: Minimum Probability of Error Classification (0-1 loss, MAP rule)
clear all; close all; clc;


%% Parameters and Gaussian Definitions
n = 2;
N = 10000;

% Class priors (equal for all classes)
priors = [0.25, 0.25, 0.25, 0.25];

% Gaussian parameters for each class
% Class 1: Bottom-left
mu(:,1) = [-1; -1];
Sigma(:,:,1) = [1, 0.25; 0.25, 0.8];

% Class 2: Bottom-right  
mu(:,2) = [2; -2];
Sigma(:,:,2) = [0.8, -0.5; -0.5, 1.2];

% Class 3: Top-left
mu(:,3) = [-3; 1];
Sigma(:,:,3) = [1.5, 0.75; 0.75, 1];

% Class 4: Top-right 
mu(:,4) = [1.5; 1.5];
Sigma(:,:,4) = [2, 0.7; 0.7, 1.5];

%% Generate samples from the data distribution
label = zeros(1,N);
x = zeros(n,N); % preallocate

% Randomly select labels according to class priors
for i = 1:N
    label(i) = find(rand < cumsum(priors), 1);
end

% Count samples from each class
Nc = [length(find(label==1)), length(find(label==2)), ...
      length(find(label==3)), length(find(label==4))];

% Draw samples from each class pdf
for l = 1:4
    x(:,label==l) = mvnrnd(mu(:,l),Sigma(:,:,l),Nc(l))';
end

%% Part A: MAP Classification (0-1 loss)
decision_MAP = zeros(1,N);

for i = 1:N
    % Compute likelihood for each class
    likelihoods = zeros(1,4);
    for l = 1:4
        likelihoods(l) = evalGaussian(x(:,i),mu(:,l),Sigma(:,:,l));
    end
    
    % MAP decision (argmax of likelihoods since priors are equal)
    [~, decision_MAP(i)] = max(likelihoods);
end

%% Compute Confusion Matrix and Performance Metrics for MAP
confusionMatrix_MAP = zeros(4,4);
for trueLabel = 1:4
    for decLabel = 1:4
        confusionMatrix_MAP(decLabel,trueLabel) = ...
            length(find(decision_MAP==decLabel & label==trueLabel)) / Nc(trueLabel);
    end
end

fprintf('Part A - MAP Classification (0-1 loss)\n');
fprintf('Confusion Matrix P(D=i|L=j):\n');
fprintf('       L=1     L=2     L=3     L=4\n');
fprintf('     ------------------------------\n');
for i = 1:4
    fprintf('D=%d |', i);
    for j = 1:4
        fprintf(' %6.4f', confusionMatrix_MAP(i,j));
    end
    fprintf('\n');
end

% Calculate per-class and overall performance
fprintf('\nPer-Class Performance:\n');
fprintf('Class | Correct Classification | Probability of Error\n');
fprintf('------|------------------------|--------------------\n');
for j = 1:4
    p_correct_class = confusionMatrix_MAP(j,j);
    p_error_class = 1 - p_correct_class;
    fprintf('  %d   |        %6.4f          |      %6.4f\n', ...
            j, p_correct_class, p_error_class);
end

% Calculate overall probability of correct classification
pCorrect_MAP = 0;
for j = 1:4
    pCorrect_MAP = pCorrect_MAP + confusionMatrix_MAP(j,j) * (Nc(j)/N);
end
pError_MAP = 1 - pCorrect_MAP;

fprintf('\nOverall Performance:\n');
fprintf('Probability of Correct Classification: %.4f\n', pCorrect_MAP);
fprintf('Probability of Error: %.4f\n\n', pError_MAP);

%% Part B: ERM Classification with Loss Matrix
fprintf('Part B - ERM Classification with Loss Matrix\n');

% Define the loss matrix
lossMatrix = [0, 10, 10, 100;
              1, 0, 10, 100;
              1, 1, 0, 100;
              1, 1, 1, 0];

fprintf('Loss Matrix:\n');
disp(lossMatrix);

decision_ERM = zeros(1,N);

for i = 1:N
    x_sample = x(:,i);
    
    % Compute posterior probabilities for each class
    posteriors = zeros(1,4);
    for l = 1:4
        likelihood = evalGaussian(x_sample, mu(:,l), Sigma(:,:,l));
        posteriors(l) = likelihood * priors(l);
    end
    
    % Normalize posteriors
    posteriors = posteriors / sum(posteriors);
    
    % Compute conditional risk for each possible decision
    risks = zeros(1,4);
    for d = 1:4
        risk_d = 0;
        for true_l = 1:4
            risk_d = risk_d + lossMatrix(d, true_l) * posteriors(true_l);
        end
        risks(d) = risk_d;
    end
    
    % Choose decision that minimizes conditional risk
    [~, decision_ERM(i)] = min(risks);
end

%% Compute Confusion Matrix and Performance Metrics for ERM
confusionMatrix_ERM = zeros(4,4);
for trueLabel = 1:4
    for decLabel = 1:4
        confusionMatrix_ERM(decLabel,trueLabel) = ...
            length(find(decision_ERM==decLabel & label==trueLabel)) / Nc(trueLabel);
    end
end

fprintf('ERM Confusion Matrix P(D=i|L=j):\n');
fprintf('       L=1     L=2     L=3     L=4\n');
fprintf('     ------------------------------\n');
for i = 1:4
    fprintf('D=%d |', i);
    for j = 1:4
        fprintf(' %6.4f', confusionMatrix_ERM(i,j));
    end
    fprintf('\n');
end

% Calculate per-class and overall performance for ERM
fprintf('\nERM Per-Class Performance:\n');
fprintf('Class | Correct Classification | Probability of Error\n');
fprintf('------|------------------------|--------------------\n');
for j = 1:4
    p_correct_class = confusionMatrix_ERM(j,j);
    p_error_class = 1 - p_correct_class;
    fprintf('  %d   |        %6.4f          |      %6.4f\n', ...
            j, p_correct_class, p_error_class);
end

% Calculate overall probability of correct classification for ERM
pCorrect_ERM = 0;
for j = 1:4
    pCorrect_ERM = pCorrect_ERM + confusionMatrix_ERM(j,j) * (Nc(j)/N);
end
pError_ERM = 1 - pCorrect_ERM;

fprintf('\nERM Overall Performance:\n');
fprintf('Probability of Correct Classification: %.4f\n', pCorrect_ERM);
fprintf('Probability of Error: %.4f\n', pError_ERM);

%% Compute Expected Risk for ERM
total_loss = 0;
for i = 1:N
    true_label = label(i);
    decision_label = decision_ERM(i);
    total_loss = total_loss + lossMatrix(decision_label, true_label);
end
expected_risk = total_loss / N;

fprintf('\nExpected Risk for ERM: %.4f\n', expected_risk);

%% Comparison Table: MAP vs ERM
fprintf('\n=== COMPARISON TABLE: MAP vs ERM ===\n\n');

% Calculate decision distributions
map_decision_counts = zeros(1,4);
erm_decision_counts = zeros(1,4);
for i = 1:4
    map_decision_counts(i) = sum(decision_MAP == i);
    erm_decision_counts(i) = sum(decision_ERM == i);
end

% Calculate class-wise error rates and risks
map_class_errors = zeros(1,4);
erm_class_risks = zeros(1,4);
for true_class = 1:4
    % MAP: Error rate for each class (1 - diagonal of confusion matrix)
    map_class_errors(true_class) = 1 - confusionMatrix_MAP(true_class, true_class);
    
    % ERM: Average risk for each class
    class_loss = 0;
    class_indices = find(label == true_class);
    for idx = class_indices
        class_loss = class_loss + lossMatrix(decision_ERM(idx), true_class);
    end
    erm_class_risks(true_class) = class_loss / length(class_indices);
end

% Calculate overall metrics CORRECTLY - use weighted average by class counts
pCorrect_MAP = 0;
pCorrect_ERM = 0;
for j = 1:4
    pCorrect_MAP = pCorrect_MAP + confusionMatrix_MAP(j,j) * (Nc(j)/N);
    pCorrect_ERM = pCorrect_ERM + confusionMatrix_ERM(j,j) * (Nc(j)/N);
end
pError_MAP = 1 - pCorrect_MAP;
pError_ERM = 1 - pCorrect_ERM;

% Print comparison table
fprintf('Metric                 |   MAP   |   ERM   |  Change  \n');
fprintf('-----------------------|---------|---------|----------\n');
fprintf('Overall Performance    |         |         |          \n');
fprintf('  Error Probability    | %7.4f | %7.4f | %8.4f\n', ...
        pError_MAP, pError_ERM, pError_ERM - pError_MAP);
fprintf('  Expected Risk        | %7.4f | %7.4f | %8.4f\n', ...
        pError_MAP, expected_risk, expected_risk - pError_MAP);
fprintf('\n');
fprintf('Decision Distribution  |   MAP   |   ERM   |  Change  \n');
fprintf('-----------------------|---------|---------|----------\n');
for i = 1:4
    fprintf('  Classified as C%d     | %7d | %7d | %8d\n', ...
            i, map_decision_counts(i), erm_decision_counts(i), ...
            erm_decision_counts(i) - map_decision_counts(i));
end
fprintf('\n');
fprintf('Per-Class Performance  |   MAP   |   ERM   |  Change  \n');
fprintf('-----------------------|---------|---------|----------\n');
for i = 1:4
    fprintf('  Class %d Error/Risk  | %7.4f | %7.4f | %8.4f\n', ...
            i, map_class_errors(i), erm_class_risks(i), ...
            erm_class_risks(i) - map_class_errors(i));
end

% Calculate how much more conservative ERM is about Class 4
class4_conservatism = (erm_decision_counts(4) - map_decision_counts(4)) / map_decision_counts(4) * 100;
fprintf('\nERM classifies %.1f%% more samples as Class 4 compared to MAP\n', class4_conservatism);

%% Figure 1: MAP Classification Results
figure(1), clf;

% Define markers for each true class
markers = {'.', 'o', '^', 's'};

% Plot each class with correct/incorrect decisions for MAP
for trueClass = 1:4
    % Correct decisions (green)
    indCorrect = find(decision_MAP==trueClass & label==trueClass);
    if ~isempty(indCorrect)
        plot(x(1,indCorrect), x(2,indCorrect), markers{trueClass}, 'Color', [0,0.7,0]);
        hold on;
    end
    
    % Incorrect decisions (red)  
    indIncorrect = find(decision_MAP~=trueClass & label==trueClass);
    if ~isempty(indIncorrect)
        plot(x(1,indIncorrect), x(2,indIncorrect), markers{trueClass}, 'Color', [0.9,0,0]);
        hold on;
    end
end

% Add legend entries manually
h = [];
legendText = {};
for trueClass = 1:4
    % Correct
    h(end+1) = plot(NaN, NaN, markers{trueClass}, 'Color', [0,0.7,0], 'MarkerSize', 10);
    legendText{end+1} = sprintf('Class %d - Correct', trueClass);
    % Incorrect
    h(end+1) = plot(NaN, NaN, markers{trueClass}, 'Color', [0.9,0,0], 'MarkerSize', 10);
    legendText{end+1} = sprintf('Class %d - Incorrect', trueClass);
end

legend(h, legendText, 'Location', 'best');
xlabel('x_1'), ylabel('x_2');
title('MAP Classification: True Labels (Markers) vs Decisions (Colors)');
axis equal;

%% Figure 2: ERM Classification Results
figure(2), clf;

% Plot each class with correct/incorrect decisions for ERM
for trueClass = 1:4
    % Correct decisions (green)
    indCorrect = find(decision_ERM==trueClass & label==trueClass);
    if ~isempty(indCorrect)
        plot(x(1,indCorrect), x(2,indCorrect), markers{trueClass}, 'Color', [0,0.7,0]);
        hold on;
    end
    
    % Incorrect decisions (red)  
    indIncorrect = find(decision_ERM~=trueClass & label==trueClass);
    if ~isempty(indIncorrect)
        plot(x(1,indIncorrect), x(2,indIncorrect), markers{trueClass}, 'Color', [0.9,0,0]);
        hold on;
    end
end

h = [];
legendText = {};
for trueClass = 1:4
    % Correct
    h(end+1) = plot(NaN, NaN, markers{trueClass}, 'Color', [0,0.7,0], 'MarkerSize', 10);
    legendText{end+1} = sprintf('Class %d - Correct', trueClass);
    % Incorrect
    h(end+1) = plot(NaN, NaN, markers{trueClass}, 'Color', [0.9,0,0], 'MarkerSize', 10);
    legendText{end+1} = sprintf('Class %d - Incorrect', trueClass);
end

legend(h, legendText, 'Location', 'best');
xlabel('x_1'), ylabel('x_2');
title('ERM Classification: True Labels (Markers) vs Decisions (Colors)');
axis equal;



%% Gaussian PDF Function
function g = evalGaussian(x,mu,Sigma)
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(Sigma\(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end
