function [Q, out] = rl_tanh_mi_paper(db_path, episodes, q_out)
% PAPER-PROTOCOL: function form. Minimally edited copy of Bhargav's rl_tanh_mi.m
% (rl_tanh_mi.m as of commit 41d06c13) implementing the protocol in PAPER_PROTOCOL_CHANGES.md
% ("Training (paper protocol)"). Every change is marked "% PAPER-PROTOCOL: <item>".
%
%   [Q, out] = rl_tanh_mi_paper('common_db/train_db_wran.mat', 1:15000, 'Q_wran_tanh_mi_paper.mat')
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
% blocksize = 10;
% blocksize = 16;
H = sparse(logical(wran_384_256));
% H = ldpcQuasiCyclicMatrix(blocksize,P);
% H = sparse(logical(p_mackey));
blocksize = 1 ;
[m, ~] = size(H);

%% ---------------- PARAMETERS ----------------
params.alpha = 0.1;
params.beta = 0.9;
params.epsilon = 0.1;
params.lmax = 50;       % (already the paper's l_max: learning steps per episode)

params.maxlevels = 40;   % PAPER-PROTOCOL: 40 tanh bins of width 0.05


% numSamples = 90000;   % PAPER-PROTOCOL: data from the common DB
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
% PAPER-PROTOCOL: data and random draws from the common DB (replaces the generated L_set)
db = load(db_path, 'L', 'U', 'meta');
if db.meta.m ~= m || db.meta.n ~= n || size(db.U, 2) ~= params.lmax
    error('rl_tanh_mi_paper: DB %s does not match H (%d x %d) / lmax %d', db_path, m, n, params.lmax);
end
if isempty(episodes) || any(episodes(:) < 1 | episodes(:) > size(db.L, 1) | episodes(:) ~= round(episodes(:)))
    error('rl_tanh_mi_paper: episodes must be integers in 1..%d', size(db.L, 1));
end
L_set = db.L(episodes, :);
U_set = db.U(:, :, episodes);
% L_set = cell(numSamples,1);
%
%
% snr = [1	1.25892541179417	1.58489319246111	1.99526231496888	2.51188643150958];
% % snr = [0.501187233627272	0.630957344480193	0.794328234724282	1	1.25892541179417];
% % snr = [1.122018454301963	1.258925411794167	1.412537544622754	1.584893192461114	1.778279410038923	1.995262314968880];
% sigma = 1;
%
% for i = 1:numSamples
%     rx = 1*sqrt(snr(1)) + sigma * randn(1,n);
%     L_set{i} = 2*rx/(sigma^2);
% end

%% ---------------- TRAIN ----------------
[Q, out] = RELDEC_CPU_MAIN(L_set, U_set, H, CN_neighbors, VN_neighbors, clusters, params);   % PAPER-PROTOCOL: DB draws in, per-episode outputs out
out.episodes = episodes(:);

% save("Q_wran_0_tanh_mi.mat","Q")
if ~isempty(q_out)   % PAPER-PROTOCOL: explicit Q output path + meta
    meta = struct('script', 'rl_tanh_mi_paper.m', 'db_path', db_path, 'db_engine', db.meta.engine, ...
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
maxStates = params.maxlevels;

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
    current_state = zeros(m, 1);
    s = zeros(m,1);
    for i = 1:m
        % Initial State
        idx1 = CN_neighbors{i};          % neighbor indices
        vals = L(idx1);                 % extract values

        % ensure row vector
        vals = vals(:)';

        current_state(i) = mean(tanh((vals)));

        % Initilization of states for Episode
        s(i) = min(floor((current_state(i) + 1)/0.05), 39) + 1;   % PAPER-PROTOCOL: 40 bins of width 0.05

    end


    % Creating Residue Vectors for BP Alogorithm
    for i = 1 : m
        res{i} = zeros(1,numel(CN_neighbors{i}));
    end

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
        %% -------- calculating current mutual inforamtion of the cluster --------
        idx2 = CN_neighbors{a};
        x = L(idx2);
        mi_prev = mean(Jfun(sqrt(max(2*x, 0))));   % PAPER-PROTOCOL: MI reward J(sqrt(max(2L,0)))
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
        %% -------- calculating current mutual inforamtion of the cluster --------
        idx2 = CN_neighbors{a};
        x = L(idx2);
        mi_new = mean(Jfun(sqrt(max(2*x, 0))));   % PAPER-PROTOCOL: MI reward J(sqrt(max(2L,0)))
        %% -------- NEW STATE --------
        current_state_updated = zeros(m, 1);
        s_new = zeros(m,1);
        for i = 1:m
            idx1 = CN_neighbors{i};          % neighbor indices
            vals = L(idx1);                 % extract values

            % ensure row vector
            vals = vals(:)';

            current_state_updated(i) = mean(tanh((vals)));

            % Initilization of states for Episode
            s_new(i) = min(floor((current_state_updated(i) + 1)/0.05), 39) + 1;   % PAPER-PROTOCOL: 40 bins of width 0.05

        end

        %% -------- REWARD --------
        reward = mi_new - mi_prev;
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
