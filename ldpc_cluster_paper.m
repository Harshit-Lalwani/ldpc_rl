function [LLR_out1,Res_int] = ldpc_cluster_paper(LLR_in,C,a,Res,BlockSize)
% PAPER-PROTOCOL: copy of Bhargav's ldpc_cluster.m with this repo's CN numerics
% (PAPER_PROTOCOL_CHANGES.md: |tanh| floored at 1e-12, leave-one-out product
% clipped to +-(1-1e-15), message = log((1+p)/(1-p))).

Res_int = zeros(length(LLR_in),1);
LLR_int = LLR_in;
LLR_int = LLR_int - Res;
for j = 1 : BlockSize
    idx = C{(a-1)*BlockSize+j};
    llr_temp = LLR_int(idx);
    temp = tanh(llr_temp./2);
    sm = abs(temp) < 1e-12;   % PAPER-PROTOCOL: this repo's CN numerics (clipped)
    temp(sm & temp >= 0) = 1e-12;
    temp(sm & temp < 0) = -1e-12;
    prodLq = prod(temp);
    p = min(max(prodLq ./ temp, -(1 - 1e-15)), 1 - 1e-15);   % PAPER-PROTOCOL: clipped numerics
    Res_int(idx) = log((1 + p) ./ (1 - p));   % PAPER-PROTOCOL: clipped numerics
end
LLR_out1 = LLR_int + Res_int;
end
