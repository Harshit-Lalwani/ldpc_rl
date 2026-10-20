function out = test_rl_mi_tanh_paper(db_path, q_path, snr_idx, frames)
% PAPER-PROTOCOL: function form. Minimally edited copy of Bhargav's test_rl_mi_tanh.m
% (test_rl_mi_tanh.m as of commit 41d06c13) implementing the protocol in PAPER_PROTOCOL_CHANGES.md
% ("Evaluation"). Every change is marked "% PAPER-PROTOCOL: <item>".
%
%   out = test_rl_mi_tanh_paper('common_db/eval_db_wran.mat', 'Q_wran_tanh_mi_paper.mat', 1:6, 1:100000)
%
%   db_path : common eval DB (eval_db_wran.mat, from make_common_db.m)
%   q_path  : .mat holding the trained Q (40 x 128), e.g. from rl_tanh_mi_paper.m
%   snr_idx : indices j into SNR_db = [1 2 3 4 5 6] dB
%   frames  : eval-DB frame indices f
%   out     : per (SNR, frame), rows = snr_idx, cols = frames: bit_errors (nnz(L < 0) over all n),
%             frame_errors, iterations, messages (sum of the scheduled checks' degrees),
%             actions (nS x nF x K*i_max, 0 = not run); per SNR: ber, fer
% Engine: MATLAB requires a meta.engine = 'matlab' DB (Threefry tie draws), Octave an
% 'octave-test' DB (rand('state', [seed; substream]) tie draws).
% Run from the repository root.

% Configuration
% load("P_520.mat","P_520")
% % Q = Q ./ max(abs(Q(:)));

% P = P_520;
% blocksize = 10;
% % epsilon_test = 0.05;
% pcmatrix = ldpcQuasiCyclicMatrix(blocksize,P);
% BlockSize = 5;

% P = P_520;
% BlockSize = 10;
% % epsilon_test = 0.05;
% pcmatrix = ldpcQuasiCyclicMatrix(BlockSize,P);
% Q = readmatrix('qtable_ep015000.csv');
% Q = Q(2:end, :);
% pcmatrix = sparse(logical(readmatrix('WRAN_irreg_384_256 (1).csv')));
%%
load('wran_384_256.mat','wran_384_256');
% BlockSize = 16;
H = sparse(logical(wran_384_256));
% load('p_mackey.mat','p_mackey');
% H = sparse(logical(p_mackey));
BlockSize = 1 ;
pcmatrix = H;
blocksize_wran = 16;
% load("Q_wran_snr_0.mat","Q")
% Q1{1} = Q;
% load("Q_wran_snr_1.mat","Q")
% Q1{2} = Q;
% load("Q_wran_snr_2.mat","Q")
% Q1{3} = Q;
% load("Q_wran_snr_3.mat","Q")
% Q1{4} = Q;
% load("Q_wran_snr_4.mat","Q")
% Q1{5} = Q;
% load("Q_mackey_0.5_tanh_mi.mat","Q")
% Q1{1} = Q;

% load("Q_wran_0_tanh_mi.mat","Q")
S = load(q_path, 'Q');   % PAPER-PROTOCOL: explicit Q input path
Q1{1} = S.Q;


% load("Q_P_520_snr_neg3.mat","Q")
% Q1{1} = Q;
% load("Q_P_520_snr_neg2.mat","Q")
% Q1{2} = Q;
% load("Q_P_520_snr_neg1.mat","Q")
% Q1{3} = Q;
% load("Q_P_520_snr_0.mat","Q")
% Q1{4} = Q;
% load("Q_P_520_snr_1.mat","Q")
% Q1{5} = Q;

[m, ~] = size(pcmatrix);
% P = zeros(m/BlockSize);
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(pcmatrix(c,:));
end
params.maxStateBits = 6;  % 10 for P_520 and % 11 for WRAN  % 6 for mackey
% cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);   % PAPER-PROTOCOL: no Communications Toolbox (all-zero codeword)
n = size(pcmatrix, 2);   % PAPER-PROTOCOL: all n bits
deg = cellfun(@numel, CN_neighbors);   % PAPER-PROTOCOL: messages = sum of scheduled checks' degrees
if ~isequal(size(Q1{1}), [40, m/BlockSize])
    error('test_rl_mi_tanh_paper: Q in %s is %s, expected 40 x %d', q_path, mat2str(size(Q1{1})), m/BlockSize);
end
% cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix);
% cfgLDPCDec = ldpcDecoderConfig(pcmatrix,'layered-bp');
% Parameters for Dec
% numEdges      = length(cfgLDPCDec.derivedParams.columnIndexMap)/2;
% blockLen      = cfgLDPCDec.BlockLength;
% parityLen     = cfgLDPCDec.NumParityCheckBits;
% nRowsPerLayer = cfgLDPCDec.NumRowsPerLayer;
SNR_db = [1 2 3 4 5 6];   % PAPER-PROTOCOL: eval SNRs 1..6 dB
% SNR_db = [-3 -2 -1 0 1];
% SNR_db = [0.5 1 1.5 2 2.5 3];
% SNR
SNR = 10.^(SNR_db/10);
maxnumiter = 5;   % (= i_max)
% R = cfgLDPCEnc.NumInformationBits / cfgLDPCEnc.BlockLength;
R = 1 - m/n;   % PAPER-PROTOCOL: R = 1 - m/n

% PAPER-PROTOCOL: noise and tie draws from the common eval DB
db = load(db_path, 'W', 'meta');
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if is_octave, engine = 'octave-test'; else, engine = 'matlab'; end
if ~strcmp(db.meta.engine, engine)
    error('test_rl_mi_tanh_paper: DB engine %s cannot be used from %s', db.meta.engine, engine);
end
if db.meta.R ~= R || db.meta.n ~= n || db.meta.m ~= m || ~isequal(db.meta.eval_snrs, SNR_db) || db.meta.i_max ~= maxnumiter
    error('test_rl_mi_tanh_paper: DB %s does not match this script (R, n, m, eval_snrs, i_max)', db_path);
end
if isempty(frames) || any(frames(:) < 1 | frames(:) > size(db.W, 1) | frames(:) ~= round(frames(:)))
    error('test_rl_mi_tanh_paper: frames must be integers in 1..%d', size(db.W, 1));
end
if isempty(snr_idx) || any(~ismember(snr_idx(:), 1:numel(SNR_db)))
    error('test_rl_mi_tanh_paper: snr_idx must be in 1..%d', numel(SNR_db));
end
seed_tie = db.meta.seed_tie;
frames = frames(:)';
Wf = db.W(frames, :);
nS = numel(snr_idx); nF = numel(frames); K = height(pcmatrix)/BlockSize;
out.snr_idx = snr_idx(:)'; out.snr_db = SNR_db(snr_idx); out.frames = frames;
out.bit_errors = zeros(nS, nF); out.iterations = zeros(nS, nF); out.messages = zeros(nS, nF);
out.actions = zeros(nS, nF, K*maxnumiter);
tic;

for ii = 1 : nS   % PAPER-PROTOCOL: chosen SNR indices
    i = snr_idx(ii);
    current_state = zeros(m,params.maxStateBits);
    ber = zeros(1, nF); iters = zeros(1, nF); msgs = zeros(1, nF); acts = zeros(nF, K*maxnumiter);
    parfor j = 1 : nF   % PAPER-PROTOCOL: chosen frames
        fr = frames(j);
        % bits = zeros(cfgLDPCEnc.NumInformationBits,1);
        % codeword = ldpcEncode(bits,cfgLDPCEnc);
        % bits = zeros(48,1);
        % codeword = zeros(96,1);
        % codeword_1 = (codeword == 0);
        % codeword_2 = (codeword == 1);
        % data_modulated = sqrt(SNR(i)).*codeword_1 -sqrt(SNR(i)).*codeword_2;
        % % noise = randn(cfgLDPCEnc.BlockLength,1);
        % noise = randn(96,1);
        % data_received = data_modulated + noise;
        % soft_demodulated_output = 2*data_received;
        % % [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,maxnumiter);
        % % Res = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(pcmatrix)/BlockSize);
        % Res = repmat({zeros(96,1)}, 1, height(pcmatrix)/BlockSize);
        % Y_temp = soft_demodulated_output;
        % for k = 1 : m
        %     res{k} = zeros(1,numel(CN_neighbors{k}));
        % end

        % bits = zeros(cfgLDPCEnc.NumInformationBits,1);
        % codeword = ldpcEncode(bits,cfgLDPCEnc);
        codeword = zeros(n,1);   % PAPER-PROTOCOL: all-zero codeword (no ldpcEncode)
        tx = 1 - 2*double(codeword);

        % Noise standard deviation
        sigma = sqrt(1/(2*R*SNR(i)));

        % AWGN
        % noise = sigma * randn(cfgLDPCEnc.BlockLength,1);
        noise = sigma * double(Wf(j,:))';   % PAPER-PROTOCOL: noise from the eval DB

        % Received symbols
        rx = tx + noise;

        % Channel LLRs
        soft_demodulated_output = 2*rx/(sigma^2);

        % PAPER-PROTOCOL: tie draws ut for (SNR i, frame fr), drawn once per frame
        if is_octave
            rand('state', [seed_tie; (i-1)*100000 + fr]);
            ut = rand(1, K*maxnumiter);
        else
            st = RandStream('Threefry', 'Seed', seed_tie);
            st.Substream = (i-1)*100000 + fr;
            ut = rand(st, 1, K*maxnumiter);
        end
        t = 0; it_done = 0; nmsg = 0; act = zeros(1, K*maxnumiter);   % PAPER-PROTOCOL: per-frame outputs

        % Res = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(pcmatrix)/BlockSize);
        Res = repmat({zeros(n,1)}, 1, height(pcmatrix)/BlockSize);   % PAPER-PROTOCOL: n without cfgLDPCEnc
        % Res = repmat({zeros(96,1)}, 1, height(pcmatrix)/BlockSize);
        Y_temp = soft_demodulated_output;

        % --- initialize state ---
        current_state = zeros(m, 1);
        s = zeros(m,1);
        for v = 1:m
            % Initial State
            idx1 = CN_neighbors{v};          % neighbor indices
            vals = Y_temp(idx1);                 % extract values

            % ensure row vector
            vals = vals(:)';

            current_state(v) = mean(tanh((vals)));

            % Initilization of states for Episode
            s(v) = min(floor((current_state(v) + 1)/0.05), 39) + 1;   % PAPER-PROTOCOL: own state per cluster, 40 bins of width 0.05

        end

        % --- track used clusters ---
        used = false(1, height(pcmatrix)/BlockSize);

        % --- sequential scheduling ---

        for u = 1:(height(pcmatrix)/BlockSize)*maxnumiter   % number of clusters

            % ---------- STATE → INDEX ----------

            % ---------- Q LOOKUP ----------
            vals1 = zeros(1, m);
            for f = 1:m
                vals1(f) = Q1{1}(s(f), f);
            end

            % ---------- CLUSTER AGGREGATION ----------
            vals4 = zeros(1, height(pcmatrix)/BlockSize);
            for k = 1:height(pcmatrix)/BlockSize
                start_idx = (k-1)*BlockSize + 1;
                end_idx   = k*BlockSize;
                vals4(k) = sum(vals1(start_idx:end_idx));
            end

            % ---------- MASK USED CLUSTERS ----------
            temp = vals4;
            temp(used) = -inf;

            % ---------- SELECT BEST CLUSTER ----------
            % [~, best_cluster] = max(temp);
            ties = find(temp == max(temp));   % PAPER-PROTOCOL: uniform random ties with ut
            t = t + 1;
            best_cluster = ties(floor(ut(t)*numel(ties)) + 1);
            % mark cluster as used
            used(best_cluster) = true;

            % ---------- BP UPDATE ----------
            [Y_out, res1] = ldpc_cluster_paper(...
                Y_temp, CN_neighbors, best_cluster, ...
                Res{best_cluster}, BlockSize);   % PAPER-PROTOCOL: this repo's CN numerics

            Y_temp = Y_out;
            Res{best_cluster} = res1;
            act(u) = best_cluster;   % PAPER-PROTOCOL: per-frame outputs
            nmsg = nmsg + sum(deg((best_cluster-1)*BlockSize + (1:BlockSize)));

            % ---------- UPDATE STATE AFTER ACTION ----------
            current_state = zeros(m, 1);
            s = zeros(m,1);
            for v = 1:m
                % Initial State
                idx1 = CN_neighbors{v};          % neighbor indices
                vals = Y_temp(idx1);                 % extract values

                % ensure row vector
                vals = vals(:)';

                current_state(v) = mean(tanh((vals)));

                % Initilization of states for Episode
                s(v) = min(floor((current_state(v) + 1)/0.05), 39) + 1;   % PAPER-PROTOCOL: own state per cluster, 40 bins of width 0.05

            end

            if mod(u,height(pcmatrix)/BlockSize) == 0
                used = false(1, height(pcmatrix)/BlockSize);
                it_done = it_done + 1;
                if ~any(mod(pcmatrix * double(Y_temp < 0), 2))   % PAPER-PROTOCOL: stop after an iteration with zero syndrome
                    break;
                end
            end
        end
        output_final_whole = Y_out < 0;
        % output_final = output_final_whole(1:cfgLDPCEnc.NumInformationBits);
        % output_final = output_final_whole(1:48);
        % ber(j) = biterr(output_final,bits);
        ber(j) = nnz(output_final_whole);   % PAPER-PROTOCOL: bit errors over all n bits (all-zero codeword)
        iters(j) = it_done; msgs(j) = nmsg; acts(j, :) = act;
        % ber1(j) = biterr(Y,bits);
        % j   % PAPER-PROTOCOL: no per-frame print
        % i
    end
    % ber_t(i) = mean(ber)/cfgLDPCEnc.NumInformationBits;
    ber_t(i) = mean(ber)/n;   % PAPER-PROTOCOL: BER over all n bits
    % ber_t(i) = mean(ber)/48;
    % ber_t_1(i) = mean(ber1)/cfgLDPCEnc.NumInformationBits;
    out.bit_errors(ii, :) = ber; out.iterations(ii, :) = iters; out.messages(ii, :) = msgs;
    out.actions(ii, :, :) = reshape(acts, [1, nF, K*maxnumiter]);
    fprintf('SNR %g dB: BER %.4e over %d frames (%.1fs)\n', SNR_db(i), ber_t(i), nF, toc);
end
out.frame_errors = out.bit_errors > 0;
out.ber = ber_t(snr_idx);
out.fer = mean(out.frame_errors, 2)';
out.wall_time = toc;
% semilogy(SNR_db,ber_t_1)
% semilogy(SNR,ber_t)
% grid on
end   % PAPER-PROTOCOL: function form
