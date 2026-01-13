% ===== export_bp_for_python.m =====
% 将训练完成的 BP 网络（单隐层：tansig -> purelin）及 mapminmax 参数导出给 Python
clc;

assert(exist('net','var')==1, '❌ 未找到变量 net');
assert(exist('inputps','var')==1, '❌ 未找到变量 inputps');
assert(exist('outputps','var')==1, '❌ 未找到变量 outputps');

assert(net.numLayers==2, '❌ 仅支持 2 层（隐层+输出层）BP 网络');
assert(strcmp(net.layers{1}.transferFcn,'tansig'),  '❌ 隐层激活应为 tansig');
assert(strcmp(net.layers{2}.transferFcn,'purelin'), '❌ 输出层激活应为 purelin');

W1 = net.IW{1,1};  % (hidden, input)
B1 = net.b{1};     % (hidden, 1)
W2 = net.LW{2,1};  % (output, hidden)
B2 = net.b{2};     % (output, 1)

input_size  = size(W1,2);
hidden_size = size(W1,1);
output_size = size(W2,1);

in_ps.xoffset = inputps.xoffset(:);
in_ps.gain    = inputps.gain(:);
in_ps.ymin    = inputps.ymin(1);

out_ps.xoffset = outputps.xoffset(:);
out_ps.gain    = outputps.gain(:);
out_ps.ymin    = outputps.ymin(1);

% 按你当前训练：输入列 [5,10,13,15,23]；输出列 24（1-based）
input_cols_1based  = [5,10,13,15,23];
output_cols_1based = 24;

save('bp_weights.mat', ...
    'W1','B1','W2','B2', ...
    'in_ps','out_ps', ...
    'input_size','hidden_size','output_size', ...
    'input_cols_1based','output_cols_1based');

disp('✅ 已导出 bp_weights.mat');
disp(struct('input_size',input_size,'hidden_size',hidden_size,'output_size',output_size));
