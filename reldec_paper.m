function [Q, out] = reldec_paper(db_path, episodes, q_out)
% PAPER-PROTOCOL: function form. Minimally edited copy of Bhargav's reldec_test.m
% (reldec_test.m as of commit 41d06c13) implementing the protocol in PAPER_PROTOCOL_CHANGES.md
% ("Training (paper protocol)"). Every change is marked "% PAPER-PROTOCOL: <item>".
%
%   [Q, out] = reldec_paper('common_db/train_db_wran.mat', 1:15000, 'Q_wran_reldec_paper.mat')
%
%   db_path  : common training DB (train_db_wran.mat, from make_common_db.m)
%   episodes : episode indices = DB frame indices (paper: 1:15000, one pass); Q starts at 0
%   q_out    : .mat path to save Q and meta to ('' = do not save)
%   out      : per episode, for cross-checking: actions (E x lmax, 0 = step not run), steps,
%              bit_errors (nnz(L < 0) over all n bits at episode end), messages (sum of the
%              scheduled checks' degrees), episodes, wall_time
% Run from the repository root.

%% ================= RELDEC OPTIMIZED GPU SCRIPT =================
% clear; clc;   % PAPER-PROTOCOL: function form

%% ---------------- PARITY CHECK MATRIX ----------------
% load("P_520.mat","P_520")
load('wran_384_256.mat','wran_384_256');
% load('p_mackey.mat','p_mackey');
% P = P_520;
blocksize = 10;
% blocksize = 16;
H = sparse(logical(wran_384_256));
% H = ldpcQuasiCyclicMatrix(blocksize,P);
% H = sparse(logical(p_mackey));
% blocksize = 1 ;
[m, ~] = size(H);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.6;   % PAPER-PROTOCOL: epsilon 0.6
params.lmax = 50;       % (already the paper's l_max: learning steps per episode)

params.maxStateBits = 11;   % IMPORTANT   10 for P_520 and 11 for WRAN 6 for mackey


% numSamples = 30000;   % PAPER-PROTOCOL: data from the common DB
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
% PAPER-PROTOCOL: data and random draws from the common DB (replaces the generated L_set,
% which used an undefined R)
db = load(db_path, 'L', 'U', 'meta');
if db.meta.m ~= m || db.meta.n ~= n || size(db.U, 2) ~= params.lmax
    error('reldec_paper: DB %s does not match H (%d x %d) / lmax %d', db_path, m, n, params.lmax);
end
if isempty(episodes) || any(episodes(:) < 1 | episodes(:) > size(db.L, 1) | episodes(:) ~= round(episodes(:)))
    error('reldec_paper: episodes must be integers in 1..%d', size(db.L, 1));
end
L_set = db.L(episodes, :);
U_set = db.U(:, :, episodes);
% L_set = cell(numSamples,1);
%
%
% snr = [1	1.25892541179417	1.58489319246111	1.99526231496888	2.51188643150958 	3.16227766016838	3.98107170553497];
% % snr = [0.501187233627272	0.630957344480193	0.794328234724282	1	1.25892541179417];
% % snr = [1.122018454301963	1.258925411794167	1.412537544622754	1.584893192461114	1.778279410038923	1.995262314968880];
% sigma = sqrt(1/(2*R*snr(1)));
%
% for i = 1:numSamples
%     rx = 1 +  sigma*randn(1,n);
%     L_set{i} = 2*rx/(sigma^2);
% end

%% ---------------- TRAIN ----------------
[Q, out] = RELDEC_CPU_MAIN(L_set, U_set, H, CN_neighbors, VN_neighbors, clusters, params);   % PAPER-PROTOCOL: DB draws in, per-episode outputs out
out.episodes = episodes(:);

% save("Q_mackey_3.mat","Q")
if ~isempty(q_out)   % PAPER-PROTOCOL: explicit Q output path + meta
    meta = struct('script', 'reldec_paper.m', 'db_path', db_path, 'db_engine', db.meta.engine, ...
                  'episodes', episodes(:), 'wall_time', out.wall_time, 'params', params);
    save(q_out, 'Q', 'meta', '-v7');
end
disp('Training completed');
end   % PAPER-PROTOCOL: function form


% %% ================= MAIN FUNCTION =================
function [Q, out] = RELDEC_CPU_MAIN(L_set, U_set, H, CN_neighbors, VN_neighbors, clusters, params)

alpha   = params.alpha;
beta    = params.beta;
epsilon = params.epsilon;
lmax    = params.lmax;

[m, n] = size(H);
numClusters = length(clusters);
maxStates = 2^params.maxStateBits;

% Q = 0.01 * rand(maxStates, numClusters);   % avoid symmetry lock
Q = zeros(maxStates, numClusters);
N = size(L_set, 1);   % PAPER-PROTOCOL: L_set is the DB matrix (one row per episode)
out.actions = zeros(N, lmax);   % PAPER-PROTOCOL: per-episode outputs
out.steps = zeros(N, 1);
out.bit_errors = zeros(N, 1);
out.messages = zeros(N, 1);
tic;

for idx = 1:N

    L = L_set(idx, :);   % PAPER-PROTOCOL: row of the DB
    current_state = zeros(m, params.maxStateBits);
    for i = 1:m
        % Initial State
        idx1 = CN_neighbors{i};          % neighbor indices
        vals = L(idx1);                 % extract values

        % ensure row vector
        vals = vals(:)';

        k = min(length(vals), params.maxStateBits);

        % copy first k elements
        current_state(i,1:k) = vals(1:k);

        % remaining elements already zero (no need to fill)

    end
    %  Hard Decoded State
    state_hard = current_state < 0;
    % Creating Residue Vectors for BP Alogorithm
    for i = 1 : m
        res{i} = zeros(1,numel(CN_neighbors{i}));
    end
    % Initilization of states for Episode
    % bin2dec for Q indexing
    pow2vec = 2.^(params.maxStateBits-1:-1:0);
    s = 1 + state_hard * pow2vec';
    % Possible Actions in the current state
    vals1 = zeros(1, m);

    for i = 1:m
        vals1(i) = Q(s(i), i);
    end
    % start of an episode
    for l = 1:lmax
        %% -------- ACTION --------
        if U_set(1,l,idx) < epsilon   % PAPER-PROTOCOL: u_eps from the DB
            a = floor(U_set(2,l,idx)*numClusters) + 1;   % PAPER-PROTOCOL: u_act from the DB
        else
            ties = find(vals1 == max(vals1));   % PAPER-PROTOCOL: uniform tie-break with u_tie
            a = ties(floor(U_set(3,l,idx)*numel(ties)) + 1);
        end
        %% -------- CN → VN (exact BP) --------
        idx2 = CN_neighbors{a};
        vals2 = L(idx2)- res{a};
        temp = tanh(vals2./2);
        sm = abs(temp) < 1e-12;   % PAPER-PROTOCOL: this repo's CN numerics (clipped)
        temp(sm & temp >= 0) = 1e-12;
        temp(sm & temp < 0) = -1e-12;
        prodLq = prod(temp);
        p = min(max(prodLq ./ temp, -(1 - 1e-15)), 1 - 1e-15);   % PAPER-PROTOCOL: clipped numerics
        res{a} = log((1 + p) ./ (1 - p));   % PAPER-PROTOCOL: clipped numerics
        L(idx2) = vals2 + res{a};
        %% -------- NEW STATE --------
        current_state_updated = zeros(m, params.maxStateBits);
        for i = 1:m
            idx1 = CN_neighbors{i};          % neighbor indices
            vals = L(idx1);                 % extract values

            % ensure row vector
            vals = vals(:)';

            k = min(length(vals), params.maxStateBits);

            % copy first k elements
            current_state_updated(i,1:k) = vals(1:k);

            % remaining elements already zero (no need to fill)

        end
        %  Hard Decoded State
        state_hard_updated = current_state_updated < 0;

        %% -------- REWARD --------
        % prev_correct_bits = sum(state_hard(a,:));
        % new_correct_bits = sum(state_hard_updated(a,:));
        %
        % reward = prev_correct_bits - new_correct_bits;
        % reward = sum((state_hard_updated(a,:) == 0))/(length(CN_neighbors{a}));
        reward = (nnz(state_hard_updated(a,:) == 0)-(params.maxStateBits-length(CN_neighbors{a})))/(length(CN_neighbors{a}));
        % bin2dec for Q indexing fr updated state
        s_new = 1 + state_hard_updated*pow2vec';
        % Possible Actions in the current state
        for i = 1:m
            vals1(i) = Q(s_new(i), i);
        end
        %% -------- Q UPDATE --------
        Q(s(a),a) = (1-alpha)*Q(s(a),a) + ...
                 alpha*(reward + beta*max(vals1));
        vals1(a) = Q(s_new(a), a);   % PAPER-PROTOCOL: next greedy action uses Q after this update (spec v(k) = Q(s_k,k))

        % State update for next iteration
        s = s_new;
        out.actions(idx, l) = a;   % PAPER-PROTOCOL: per-episode outputs
        out.steps(idx) = l;
        out.messages(idx) = out.messages(idx) + numel(idx2);
        if ~any(mod(H * double(L(:) < 0), 2))   % PAPER-PROTOCOL: early stop on zero syndrome after the update
            break;
        end
    end
    out.bit_errors(idx) = nnz(L < 0);   % PAPER-PROTOCOL: per-episode outputs

    %% -------- PROGRESS --------
    if mod(idx,1000) == 0
        elapsed = toc;
        rate = idx / elapsed;
        remaining = (N - idx) / rate;

        fprintf('Episode %d/%d (%.2f%%) | ETA: %.1fs\n', ...
            idx, N, 100*idx/N, remaining);
    end

end
out.wall_time = toc;   % PAPER-PROTOCOL: meta wall time

end
