%% ================= RELDEC OPTIMIZED GPU SCRIPT =================
clear; clc;

%% ---------------- PARITY CHECK MATRIX ----------------
% load("P_520.mat","P_520")
load('wran_384_256.mat','wran_384_256');
% load('p_mackey.mat','p_mackey');
% P = P_520;
blocksize = 1;
% H = sparse(logical(p_mackey));
% blocksize = 16;
H = sparse(logical(wran_384_256));
pcmatrix = H;
% cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
% H = ldpcQuasiCyclicMatrix(blocksize,P);
% H = sparse(logical(p_mackey));
% blocksize = 1 ;
[m, ~] = size(H);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.1;
params.lmax = 500;
params.maxdegree = 11;  % 11 for wran 10 for bg2 6 for mackay

params.maxbits = 4;  % quantization bits   

numSamples = 10000;
n = length(H);
%% ---------------- PRECOMPUTE GRAPH ----------------
CN_neighbors = cell(m,1);
VN_neighbors = cell(n,1);

for c = 1:m
    CN_neighbors{c} = find(H(c,:));
end

for v = 1:n
    VN_neighbors{v} = find(H(:,v));
end

%% ---------------- CLUSTERS ----------------
clusters = num2cell(1:m);

%% ---------------- GENERATE TRAINING DATA ----------------
L_set = cell(numSamples,1);


snr = [1	1.25892541179417	1.58489319246111	1.99526231496888	2.51188643150958 	3.16227766016838	3.98107170553497];
% snr = [0.501187233627272	0.630957344480193	0.794328234724282	1	1.25892541179417];
% snr = [1.122018454301963	1.258925411794167	1.412537544622754	1.584893192461114	1.778279410038923	1.995262314968880 2.238721138568339	2.511886431509580	2.818382931264454	3.162277660168380];
% sigma = 1;
% R = cfgLDPCEnc.NumInformationBits / cfgLDPCEnc.BlockLength;
R = 1/2;
% R = 1/2 % for mackey
sigma = sqrt(1/(2*R*snr(7)));

for i = 1:numSamples
    rx = 1 +  sigma*randn(1,n);
    L_set{i} = 2*rx/(sigma^2);
end

L_set2 = L_set(1:100);

%% ---------------- TRAINING FOR QUANTIZATION ----------------
for i = 1 : 100
    L = L_set2{i};
    for k = 1 : m
        res{k} = zeros(1,numel(CN_neighbors{k}));
    end

    for k = 1 : 5
        cn_s = randperm(m, m);
        for j = 1 : length(cn_s)
            idx2 = CN_neighbors{cn_s(j)};
            vals2 = L(idx2)- res{cn_s(j)};
            res_prev = mean(abs(res{cn_s(j)}));
            res_curr = min(abs(vals2));
            temp = tanh(vals2./2);
            prodLq = prod(temp);

            ratio = prodLq ./ temp;

            ratio = max(min(ratio,1-1e-12),-1+1e-12);

            res{cn_s(j)} = 2*atanh(ratio);
            L(idx2) = vals2 + res{cn_s(j)};
            vals3 = L(idx2);
            res_q(i,(k-1)*2*m + 2*j -1) =  res_curr;
            res_q(i,(k-1)*2*m + 2*j ) = res_prev;
        end
    end
end

res_q = res_q(:);
fprintf('NaN: %d\n', any(isnan(res_q)));
fprintf('Inf: %d\n', any(isinf(res_q)));
fprintf('Max: %g\n', max(res_q));
fprintf('Min: %g\n', min(res_q));
M = 2^params.maxbits;
[partition, codebook, distortion] = lloyds(res_q, M);


%% ---------------- TRAIN ----------------
Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params,codebook, partition);


save("Q_wran_min_sum_6","Q","codebook","partition");
disp('Training completed');


% %% ================= MAIN FUNCTION =================
function Q = RELDEC_CPU_MAIN(L_set, H, CN_neighbors, VN_neighbors, clusters, params,codebook, partition)

alpha   = params.alpha;
beta    = params.beta;
epsilon = params.epsilon;
lmax    = params.lmax;

[m, n] = size(H);
numClusters = length(clusters);
maxStates = 2^(2*(params.maxbits));   % 4 bits each for res_cur and res_prev

% Q = 0.01 * rand(maxStates, numClusters);   % avoid symmetry lock
Q = zeros(maxStates, numClusters,2);  % 3d array for storing parity
N = length(L_set);
tic;
%%
for idx = 1:N
    epsilon_start = 1.0;
    epsilon_end = 0.05;
    epsilon_decay = 0.9999;
    epsilon = max(epsilon_end, epsilon_start * epsilon_decay^idx);
    L = L_set{idx}(:)';   % column
    current_res = zeros(m,1);
    previous_res = zeros(m,1);
    parity = zeros(m,1);
    % Creating Residue Vectors for BP Alogorithm
    for i = 1 : m
        res{i} = zeros(1,numel(CN_neighbors{i}));
    end
    for i = 1:m
        % Initial State
        idx1 = CN_neighbors{i};          % neighbor indices
        vals = L(idx1);                 % extract values
        s_1 = sign(vals);
        s_1(s_1 == 0) = 1;
        parity(i) = 1.5 + 0.5*prod(s_1);
        current_res(i) = min(abs(vals)) ;
        previous_res(i) = mean(abs(res{i}));
    end
    %  Quantized States
    [curr_res_quant,qaunts] = quantiz(current_res,partition,codebook);
    [prev_res_quant,qaunts] = quantiz(previous_res,partition,codebook);
%%

    % Initilization of states for Episode
    s = (16*(prev_res_quant)) + curr_res_quant + 1;

    
    % Initilization of states for Episode

    % Possible Actions in the current state
    vals1 = zeros(1, m);

    for i = 1:m
        vals1(i) = Q(s(i), i,parity(i));
    end
    % start of an episode
    for l = 1:lmax
        %% -------- ACTION --------
        if rand < epsilon
            a = randi(numClusters);
        else
            [~, a] = max(vals1);
        end
        %% -------- CN → VN (exact BP) --------
        idx2 = CN_neighbors{a};          
        vals2 = L(idx2)- res{a};
        temp = tanh(vals2./2);
        prodLq = prod(temp);
        ratio = prodLq ./ temp;

        ratio = max(min(ratio,1-1e-12),-1+1e-12);
        % res_prev = res{a};
        res{a} = 2*atanh(ratio);
        L(idx2) = vals2 + res{a};
        state_hard = L(idx2) < 0;
        %% -------- NEW STATE --------
        current_res = zeros(m,1);
        previous_res = zeros(m,1);
        parity = zeros(m,1);
        for i = 1:m
            idx1 = CN_neighbors{i};          % neighbor indices
            vals = L(idx1);                 % extract values
            s_1 = sign(vals);
            s_1(s_1 == 0) = 1;
            parity(i) = 1.5 + 0.5*prod(s_1);
            current_res(i) = min(abs(vals));
            previous_res(i) = mean(abs(res{i}));
        end
        %  Hard Decoded State
        %  Quantized State
        [curr_res_quant,qaunts] = quantiz(current_res,partition,codebook);
        [prev_res_quant,qaunts] = quantiz(previous_res,partition,codebook);
        
        % possible states next iteration
        s_new = (16*(prev_res_quant)) + curr_res_quant + 1;


        %% -------- REWARD --------
        reward = nnz(state_hard == 0)/(length(CN_neighbors{a}));
        % Possible Actions in the current state
        for i = 1:m
            vals1(i) = Q(s_new(i), i,parity(i));
        end
        %% -------- Q UPDATE --------
        Q(s(a),a,parity(a)) = (1-alpha)*Q(s(a),a,parity(a)) + ...
                 alpha*(reward + beta*max(vals1));

        % State update for next iteration
        s = s_new;
    end

    %% -------- PROGRESS --------
    if mod(idx,1000) == 0
        elapsed = toc;
        rate = idx / elapsed;
        remaining = (N - idx) / rate;

        fprintf('Episode %d/%d (%.2f%%) | ETA: %.1fs\n', ...
            idx, N, 100*idx/N, remaining);
    end

end

end
