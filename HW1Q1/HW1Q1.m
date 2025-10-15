%% EECE5644 Assignment 1 - Question 1 
% ERM Classification with True Data Distribution

clear all; close all; clc;

%% Data Generation
N = 10000; p0 = 0.65; p1 = 0.35;
u = rand(1,N) >= p0; N0 = sum(u==0); N1 = sum(u==1);

mu0 = [-1/2; -1/2; -1/2];
Sigma0 = [1,-0.5,0.3; -0.5,1,-0.5; 0.3,-0.5,1];
r0 = mvnrnd(mu0, Sigma0, N0);

figure(1);
plot3(r0(:,1), r0(:,2), r0(:,3), '.b'); axis equal; hold on;

mu1 = [1;1;1];
Sigma1 = [1,0.3,-0.2; 0.3,1,0.3; -0.2,0.3,1];
r1 = mvnrnd(mu1, Sigma1, N1);

plot3(r1(:,1), r1(:,2), r1(:,3), '.r'); axis equal; hold on;
title('3D Gaussian Samples'); legend('Class 0','Class 1');
grid on;
saveas(gcf, 'gaussian_classes.png');

%% Prepare data for classification
x = [r0; r1]';  % Transpose to get 3xN matrix
true_labels = [zeros(N0,1); ones(N1,1)];

%% Part A: ERM Classification with True Distributions
[min_error_A, optimal_gamma_A, optimal_TPR_A, optimal_FPR_A, FPR_A, TPR_A, P_error_A, gamma_values_A] = ...
    run_classification(x, true_labels, mu0, Sigma0, mu1, Sigma1, p0, p1, 'Part A: ERM with True Distributions');

%% Part B: Naive Bayes with Model Mismatch
Sigma0_naive = eye(3);  
Sigma1_naive = eye(3);

[min_error_B, optimal_gamma_B, optimal_TPR_B, optimal_FPR_B, FPR_B, TPR_B, P_error_B, gamma_values_B] = ...
    run_classification(x, true_labels, mu0, Sigma0_naive, mu1, Sigma1_naive, p0, p1, 'Part B: Naive Bayes with Model Mismatch');


%% Comparison and Analysis
fprintf('\n=== Performance Comparison ===\n');
fprintf('Parameter        | Part A (Optimal) | Part B (Naive Bayes) | Change\n');
fprintf('-----------------|------------------|----------------------|--------\n');
fprintf('Min P(error)     | %.4f           | %.4f               | +%.4f\n', ...
    min_error_A, min_error_B, min_error_B - min_error_A);
fprintf('Optimal γ        | %.4f           | %.4f               | %+.4f\n', ...
    optimal_gamma_A, optimal_gamma_B, optimal_gamma_B - optimal_gamma_A);
fprintf('FPR at optimal   | %.4f           | %.4f               | %+.4f\n', ...
    optimal_FPR_A, optimal_FPR_B, optimal_FPR_B - optimal_FPR_A);
fprintf('TPR at optimal   | %.4f           | %.4f               | %+.4f\n', ...
    optimal_TPR_A, optimal_TPR_B, optimal_TPR_B - optimal_TPR_A);



%% Part C: Fisher LDA Classification

fprintf('\n\n=== Part C: Fisher LDA Classification ===\n');
[w_LDA, y_LDA, mu0_hat, mu1_hat, Sigma0_hat, Sigma1_hat] = fisher_lda(x, true_labels);
fprintf('Class 0 sample mean: [%.3f, %.3f, %.3f]\n', mu0_hat);
fprintf('Class 1 sample mean: [%.3f, %.3f, %.3f]\n', mu1_hat);
fprintf('LDA projection vector: [%.3f, %.3f, %.3f]\n', w_LDA);
fprintf('Projected scores: Class 0 mean = %.3f, Class 1 mean = %.3f\n', ...
    mean(y_LDA(true_labels == 0)), mean(y_LDA(true_labels == 1)));

%% Generate ROC curve for LDA using projected scores
min_score = min(y_LDA);
max_score = max(y_LDA);
tau_values = linspace(min_score, max_score, 1000); 

TPR_LDA = zeros(size(tau_values));
FPR_LDA = zeros(size(tau_values));
P_error_LDA = zeros(size(tau_values));

for i = 1:length(tau_values)
    tau = tau_values(i);
    decisions = y_LDA > tau; 
    TP = sum(decisions(true_labels == 1) == 1);
    FP = sum(decisions(true_labels == 0) == 1);
    TPR_LDA(i) = TP / N1;
    FPR_LDA(i) = FP / N0;
    P_error_LDA(i) = (FPR_LDA(i) * p0) + ((1 - TPR_LDA(i)) * p1);
end

% Find minimum error for LDA
[min_error_C, min_idx_C] = min(P_error_LDA);
optimal_tau = tau_values(min_idx_C);
optimal_TPR_C = TPR_LDA(min_idx_C);
optimal_FPR_C = FPR_LDA(min_idx_C);

%% Part C Results
fprintf('\nPart C Results (Fisher LDA):\n');
fprintf('Optimal threshold τ: %.4f\n', optimal_tau);
fprintf('Minimum P(error): %.4f\n', min_error_C);
fprintf('Operating point: (FPR, TPR) = (%.4f, %.4f)\n', optimal_FPR_C, optimal_TPR_C);

%% Comprehensive Comparison
fprintf('\n=== Comprehensive Performance Comparison ===\n');
fprintf('Classifier         | Min P(error) | Optimal Threshold | (FPR, TPR)\n');
fprintf('-------------------|--------------|-------------------|------------\n');
fprintf('Part A: Optimal    | %.4f      | γ = %.4f       | (%.3f, %.3f)\n', ...
    min_error_A, optimal_gamma_A, optimal_FPR_A, optimal_TPR_A);
fprintf('Part B: Naive Bayes| %.4f      | γ = %.4f       | (%.3f, %.3f)\n', ...
    min_error_B, optimal_gamma_B, optimal_FPR_B, optimal_TPR_B);
fprintf('Part C: Fisher LDA | %.4f      | τ = %.4f       | (%.3f, %.3f)\n', ...
    min_error_C, optimal_tau, optimal_FPR_C, optimal_TPR_C);


%% Plots
% Part A: Optimal ERM ROC Curve
figure(2);
plot(FPR_A, TPR_A, 'b-', 'LineWidth', 2);
hold on;
plot(optimal_FPR_A, optimal_TPR_A, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'LineWidth', 2);
plot([0,1], [0,1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate (P(D=1|L=0))');
ylabel('True Positive Rate (P(D=1|L=1))'); 
title('Part A: Optimal ERM ROC Curve');
legend('ROC Curve', 'Minimum P(error) Point', 'Random Classifier', 'Location', 'southeast');
grid on;
axis equal;
xlim([0 1]); ylim([0 1]);
saveas(gcf, 'partA_ROC.png');

% Part B: Naive Bayes ROC Curve
figure(3);
plot(FPR_B, TPR_B, 'r-', 'LineWidth', 2);
hold on;
plot(optimal_FPR_B, optimal_TPR_B, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'LineWidth', 2);
plot([0,1], [0,1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate (P(D=1|L=0))');
ylabel('True Positive Rate (P(D=1|L=1))'); 
title('Part B: Naive Bayes ROC Curve');
legend('ROC Curve', 'Minimum P(error) Point', 'Random Classifier', 'Location', 'southeast');
grid on;
axis equal;
xlim([0 1]); ylim([0 1]);
saveas(gcf, 'partB_ROC.png');

% Part C: Fisher LDA ROC Curve
figure(4);
plot(FPR_LDA, TPR_LDA, 'g-', 'LineWidth', 2);
hold on;
plot(optimal_FPR_C, optimal_TPR_C, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'LineWidth', 2);
plot([0,1], [0,1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate (P(D=1|L=0))');
ylabel('True Positive Rate (P(D=1|L=1))'); 
title('Part C: Fisher LDA ROC Curve');
legend('ROC Curve', 'Minimum P(error) Point', 'Random Classifier', 'Location', 'southeast');
grid on;
axis equal;
xlim([0 1]); ylim([0 1]);
saveas(gcf, 'partC_ROC.png');

% Plot: All ROC Curve Comparison
figure(5);
plot(FPR_A, TPR_A, 'b-', 'LineWidth', 2); hold on;
plot(FPR_B, TPR_B, 'r-', 'LineWidth', 2);
plot(FPR_LDA, TPR_LDA, 'g-', 'LineWidth', 2);
plot(optimal_FPR_A, optimal_TPR_A, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'blue');
plot(optimal_FPR_B, optimal_TPR_B, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red');
plot(optimal_FPR_C, optimal_TPR_C, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'green');
plot([0,1], [0,1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate (P(D=1|L=0))');
ylabel('True Positive Rate (P(D=1|L=1))'); 
title('ROC Curve Comparison: All Three Classifiers');
legend('Part A: Optimal ERM', 'Part B: Naive Bayes', 'Part C: Fisher LDA', ...
       'Part A Min Error', 'Part B Min Error', 'Part C Min Error', ...
       'Random Classifier', 'Location', 'southeast');
grid on;
saveas(gcf, 'partC_ROC_comparison_all.png');

% Plot: Data Projection (Visualize LDA Projection)
figure(6);
subplot(2,1,1);
histogram(y_LDA(true_labels == 0), 50, 'FaceColor', 'blue', 'FaceAlpha', 0.6); hold on;
histogram(y_LDA(true_labels == 1), 50, 'FaceColor', 'red', 'FaceAlpha', 0.6);
xline(optimal_tau, 'k--', 'LineWidth', 2, 'Label', 'Optimal Threshold');
xlabel('Projected Score y');
ylabel('Count');
title('LDA Projection: Class Distributions on Optimal Direction');
legend('Class 0', 'Class 1', 'Optimal Threshold');
grid on;

% Plot the LDA projection vector in 3D
subplot(2,1,2);
w_vis = w_LDA / norm(w_LDA) * 2; % Normalize
quiver3(0, 0, 0, w_vis(1), w_vis(2), w_vis(3), 'k', 'LineWidth', 3, 'MaxHeadSize', 1);
hold on;
plot3(x(1, true_labels == 0), x(2, true_labels == 0), x(3, true_labels == 0), '.b', 'MarkerSize', 1);
plot3(x(1, true_labels == 1), x(2, true_labels == 1), x(3, true_labels == 1), '.r', 'MarkerSize', 1);
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('3D Data with LDA Projection Direction');
legend('LDA Projection Vector w', 'Class 0', 'Class 1');
grid on; axis equal;
saveas(gcf, 'partC_LDA_projection.png');

%Part A&B comparison
figure(7); 
plot(FPR_A, TPR_A, 'b-', 'LineWidth', 2); hold on;
plot(FPR_B, TPR_B, 'r-', 'LineWidth', 2);
plot(optimal_FPR_A, optimal_TPR_A, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'blue');
plot(optimal_FPR_B, optimal_TPR_B, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red');
plot([0,1], [0,1], 'k--');
xlabel('False Positive Rate (P(D=1|L=0))'); ylabel('True Positive Rate (P(D=1|L=1))'); 
title('ROC Curve Comparison: Part A vs Part B');
legend('Part A: Optimal ERM', 'Part B: Naive Bayes', 'Part A Min Error', 'Part B Min Error', 'Random Classifier', 'Location', 'southeast');
grid on;
saveas(gcf, 'partAB_ROC_comparison.png');


% P(error) vs τ for Part C
figure(8);
plot(tau_values, P_error_LDA, 'g-', 'LineWidth', 2);
hold on; plot(optimal_tau, min_error_C, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'green');
xlabel('Threshold τ'); ylabel('Probability of Error P(error)');
title('Part C: P(error) vs LDA Threshold τ'); grid on;
saveas(gcf, 'partC_error_vs_tau.png');

%% LDA projection histograms
figure(9);
histogram(y_LDA(true_labels == 0), 50, 'FaceColor', 'blue', 'FaceAlpha', 0.6, 'EdgeAlpha', 0.3);
hold on;
histogram(y_LDA(true_labels == 1), 50, 'FaceColor', 'red', 'FaceAlpha', 0.6, 'EdgeAlpha', 0.3);
xline(optimal_tau, 'k--', 'LineWidth', 2, 'Label', 'Optimal Threshold');
xlabel('Projected Score y'); ylabel('Count');
title('LDA Projection: Class Distributions');
legend('Class 0', 'Class 1', 'Optimal Threshold'); grid on;
saveas(gcf, 'partC_LDA_histograms.png');


%% Gaussian PDF Function
function g = evalGaussian(x, mu, Sigma)
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end

%% Fisher LDA Function
function [w_LDA, y_LDA, mu0_hat, mu1_hat, Sigma0_hat, Sigma1_hat] = fisher_lda(x, true_labels)
    x0 = x(:, true_labels == 0);
    x1 = x(:, true_labels == 1);
    mu0_hat = mean(x0, 2);
    mu1_hat = mean(x1, 2);
    Sigma0_hat = cov(x0');
    Sigma1_hat = cov(x1');
    
    % Calculate scatter matrices
    Sb = (mu0_hat - mu1_hat) * (mu0_hat - mu1_hat)';  
    Sw = Sigma0_hat + Sigma1_hat;                     
    
    % Solve generalized eigenvalue problem
    [V, D] = eig(inv(Sw) * Sb);
    [~, ind] = sort(diag(D), 'descend');
    w_LDA = V(:, ind(1));
    
    % Project all data onto LDA direction
    y_LDA = w_LDA' * x;
    
    % Ensure class 1 has higher projected values on average
    if mean(y_LDA(true_labels == 1)) < mean(y_LDA(true_labels == 0))
        w_LDA = -w_LDA;
        y_LDA = -y_LDA;
    end
end

%% Classification Function 
function [min_error, optimal_gamma, optimal_TPR, optimal_FPR, FPR, TPR, P_error, gamma_values] = ...
         run_classification(x, true_labels, mu0, Sigma0, mu1, Sigma1, p0, p1, description)
    
    fprintf('\n=== %s ===\n', description);
    
    % Calculate likelihoods
    likelihood_L0 = evalGaussian(x, mu0, Sigma0);
    likelihood_L1 = evalGaussian(x, mu1, Sigma1);
    LR = likelihood_L1 ./ likelihood_L0;
    
    % Determine threshold range based on LR values
    min_LR = min(LR);
    max_LR = max(LR);
    
    % Use threshold range that covers the LR range with margin
    gamma_values = logspace(log10(min_LR*0.1), log10(max_LR*10), 1000);
    
    TPR = zeros(size(gamma_values));
    FPR = zeros(size(gamma_values));
    P_error = zeros(size(gamma_values));
    
    N0 = sum(true_labels == 0);
    N1 = sum(true_labels == 1);
    
    for i = 1:length(gamma_values)
        gamma = gamma_values(i);
        decisions = LR > gamma;
        TP = sum(decisions(true_labels == 1) == 1);
        FP = sum(decisions(true_labels == 0) == 1);
        TPR(i) = TP / N1;
        FPR(i) = FP / N0;
        P_error(i) = (FPR(i) * p0) + ((1 - TPR(i)) * p1);
    end
    
    [min_error, min_idx] = min(P_error);
    optimal_gamma = gamma_values(min_idx);
    optimal_TPR = TPR(min_idx);
    optimal_FPR = FPR(min_idx);
    
    % Display results
    gamma_theoretical = p0 / p1;
    fprintf('Theoretical γ: %.4f, Empirical γ: %.4f\n', gamma_theoretical, optimal_gamma);
    fprintf('Minimum P(error): %.4f\n', min_error);
    fprintf('Operating point: (FPR, TPR) = (%.4f, %.4f)\n', optimal_FPR, optimal_TPR);
end
