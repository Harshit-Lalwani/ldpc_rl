function make_common_db(outdir, opts)
% MAKE_COMMON_DB - write the common train/eval databases (spec: PAPER_PROTOCOL_CHANGES.md,
% "Common databases") as train_db_wran.mat / eval_db_wran.mat (-v7) plus common_db.md5 in outdir.
%
%   make_common_db('common_db')                       % full: 15000 train, 100000 eval frames
%   make_common_db('common_db/test', struct('n_train', 400, 'n_eval', 60))
%
% opts (all optional): n_train (<= 15000), n_eval (<= 100000), matrix (path to wran_384_256.mat).
% A smaller n_* gives the first n_* frames of the full database (same per-frame substreams).
%
% MATLAB: Threefry streams exactly as the spec says (meta.engine = 'matlab').
% Octave (local cross-checks only, meta.engine = 'octave-test'): no Threefry, so each frame reseeds
% the Mersenne twister with the VECTOR state [seed; substream]:
%   randn('state', [seed; substream])  /  rand('state', [seed; substream])
% A scalar state does NOT work: Octave saturates scalar states at 2^32-1, so seed*1000003 + substream
% (~2e13) gives the same stream for every frame. The eval tie draws in Octave must use the same rule:
%   rand('state', [20261024; (j-1)*100000 + f]); ut = rand(1, K*i_max);

if nargin < 2, opts = struct(); end
known = {'n_train', 'n_eval', 'matrix'};
bad = setdiff(fieldnames(opts), known);
if ~isempty(bad), error('make_common_db: unknown option(s): %s', strjoin(bad', ', ')); end

N_TRAIN = 15000; N_EVAL = 100000; L_STEPS = 50;
SNRS = [1 2 3 4 5 5.5]; EVAL_SNRS = [1 2 3 4 5 6];
SEED_TRAIN_W = 20261021; SEED_TRAIN_U = 20261022; SEED_EVAL_W = 20261023; SEED_EVAL_TIE = 20261024;
I_MAX = 5;

n_train = getopt(opts, 'n_train', N_TRAIN);
n_eval = getopt(opts, 'n_eval', N_EVAL);
check_count(n_train, N_TRAIN, 'n_train');
check_count(n_eval, N_EVAL, 'n_eval');
here = fileparts(mfilename('fullpath'));
matrix = getopt(opts, 'matrix', fullfile(here, 'wran_384_256.mat'));

S = load(matrix, 'wran_384_256');
H = S.wran_384_256;
[m, n] = size(H);
if m ~= 128 || n ~= 384, error('make_common_db: expected WRAN 128 x 384, got %d x %d', m, n); end
R = 1 - m/n;

is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if is_octave, engine = 'octave-test'; else, engine = 'matlab'; end
if ~exist(outdir, 'dir') && ~mkdir(outdir), error('make_common_db: cannot create %s', outdir); end

% ---- train DB ----
snr_db = reshape(SNRS(mod((1:n_train) - 1, numel(SNRS)) + 1), [], 1);
L = zeros(n_train, n);
U = zeros(3, L_STEPS, n_train);
if ~is_octave
    st = RandStream('Threefry', 'Seed', SEED_TRAIN_W);
    st2 = RandStream('Threefry', 'Seed', SEED_TRAIN_U);
end
for e = 1:n_train
    sigma = sqrt(1 / (2*R*10^(snr_db(e)/10)));
    if is_octave
        randn('state', [SEED_TRAIN_W; e]); w = randn(1, n);
        rand('state', [SEED_TRAIN_U; e]); U(:, :, e) = rand(3, L_STEPS);
    else
        st.Substream = e; w = randn(st, 1, n);
        st2.Substream = e; U(:, :, e) = rand(st2, 3, L_STEPS);
    end
    L(e, :) = 2*(1 + sigma*w)/sigma^2;
end
meta = base_meta(engine, R, m, n, matrix);
meta.n_frames = n_train;
meta.SNRS = SNRS;
meta.snr_rule = 'snr_db(e) = SNRS(mod(e-1, 6) + 1)';
meta.seed_L = SEED_TRAIN_W;
meta.seed_U = SEED_TRAIN_U;
meta.steps = L_STEPS;
meta.U_rows = 'u_eps, u_act, u_tie';
save(fullfile(outdir, 'train_db_wran.mat'), 'L', 'snr_db', 'U', 'meta', '-v7');
clear L U

% ---- eval DB ----
W = zeros(n_eval, n, 'single');
if ~is_octave, st = RandStream('Threefry', 'Seed', SEED_EVAL_W); end
for f = 1:n_eval
    if is_octave
        randn('state', [SEED_EVAL_W; f]); W(f, :) = single(randn(1, n));
    else
        st.Substream = f; W(f, :) = single(randn(st, 1, n));
    end
end
meta = base_meta(engine, R, m, n, matrix);
meta.n_frames = n_eval;
meta.eval_snrs = EVAL_SNRS;
meta.seed_W = SEED_EVAL_W;
meta.seed_tie = SEED_EVAL_TIE;
meta.tie_substream = '(j-1)*100000 + f';
meta.K = m;
meta.i_max = I_MAX;
meta.tie_draws_per_frame = m*I_MAX;
save(fullfile(outdir, 'eval_db_wran.mat'), 'W', 'meta', '-v7');

% ---- checksums (md5sum -c format) ----
[status, out] = system(sprintf('cd "%s" && md5sum train_db_wran.mat eval_db_wran.mat > common_db.md5', outdir));
if status ~= 0, error('make_common_db: md5sum failed: %s', out); end
fprintf('make_common_db: wrote %d train / %d eval frames (%s) to %s\n', n_train, n_eval, engine, outdir);
end

function meta = base_meta(engine, R, m, n, matrix)
meta.generator = 'make_common_db.m (paper_protocol_spec.md v1)';
meta.engine = engine;
if strcmp(engine, 'matlab')
    meta.rng = 'RandStream(''Threefry'', ''Seed'', seed), Substream = frame';
else
    meta.rng = 'Octave Mersenne twister, randn/rand(''state'', [seed; substream]) per frame';
end
meta.created = datestr(now, 'yyyy-mm-dd HH:MM:SS');
meta.matrix = matrix;
meta.R = R;
meta.m = m;
meta.n = n;
meta.codeword = 'all-zero, BPSK +1; L = 2*(1 + sigma*w)/sigma^2, sigma = sqrt(1/(2*R*10^(snr/10)))';
end

function v = getopt(opts, name, default)
if isfield(opts, name), v = opts.(name); else, v = default; end
end

function check_count(v, mx, name)
if ~isscalar(v) || v ~= round(v) || v < 1 || v > mx
    error('make_common_db: %s must be an integer in 1..%d', name, mx);
end
end
