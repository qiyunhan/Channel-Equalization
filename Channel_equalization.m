clear;
clc;
train_num = 535;    %因为后面的是从L到train_num进行迭代，所以长度为500+L
test_num_init = 5035;
SNR_init = 30;
e = 1e-6;
mu = 0.4;           %NLMS的mu
mu1 = 0.0015;       %LMS的mu
L = 35;             %均衡器长度
deta = 15;          %延时
if_LMS = 1;         %1 = LMS; 0 = NLMS
q = 2;              %1=第一问，2=第二问，3=第三问，4=第四问

%filter parameters
a = 1;
b = [0.5,1,1.2,-1,0];

C = zeros(1,L);

if q == 1                                           %第一问
    mode = 16;                                      %输入的模式为16-QAM
    if_LMS = 0;                                     %使用NLMS
    C = train_process(b, a, train_num,SNR_init,...  %进行训练
        L,C, deta, mu, mu1,e, if_LMS);
    test_num = test_num_init;
    test_id = get_test_id(mode);                    %得到测试模式的所有可能符号
    
    [acc_bad, test_x, test_u, C, jd] = ...          %进行测试
        test_process(b, a, mode,test_num, SNR_init,...
        L,C, test_id, deta, mu, mu1, e, if_LMS);
    plot_s_u_s_hat(test_x, test_u, jd);             %绘制s(i),u(i),s_hat(i)
    
elseif q == 2                                       %第2问
    mode = 16;                                      %输入16QAM
    test_id = get_test_id(mode);
    test_num = test_num_init;
    iteration_num = [150,300,500];             %迭代次数
    %iteration_num = [300];
    for i = iteration_num
        %NLMS
        C = zeros(1,L);                             %初始化均衡器系数
        if_LMS = 0;
        C = train_process(b, a, i+L,SNR_init,L,C, deta, mu, mu1,e, if_LMS);
        [~, test_x, test_u, C, jd] = test_process(b, a, mode,...
                test_num, SNR_init,L,C, test_id, deta, mu, mu1, e, if_LMS);
        plot_s_u_s_hat(test_x, test_u, jd);
        %LMS
        C = zeros(1,L);                             %初始化均衡器系数
        if_LMS = 1;
        C = train_process(b, a, i+L,SNR_init,L,C, deta, mu, mu1,e, if_LMS);
        [~, test_x, test_u, C, jd] = test_process(b, a, mode,...
                test_num, SNR_init,L,C, test_id, deta, mu, mu1, e, if_LMS);
        plot_s_u_s_hat(test_x, test_u, jd);
        pause();                                    %每迭代一次，暂停一下，以供分析结果
    end
    
    
elseif q == 3                                       %第3问
    mode = 256;                                     %模式为256-QAM
    test_id = get_test_id(mode);
    C = zeros(1,L);
    if_LMS = 0;
    C = train_process(b, a, train_num ,SNR_init,L,C, deta, mu, mu1,e, if_LMS);
    [~, test_x, test_u, C, jd] = test_process(b, a, mode,...
                test_num_init, SNR_init,L,C, test_id, deta, mu, mu1, e, if_LMS);
    plot_s_u_s_hat(test_x, test_u, jd);
    
elseif q == 4
    figure;
    hold on;
    title("SER-SNR");
    for mode = [4,16,64,256]
        mode
        test_num = 100000;
        test_id = get_test_id(mode);
        C = zeros(1,L);
        if_LMS = 0;
        C = train_process(b, a, train_num*3 ,SNR_init,L,C, deta, mu, mu1,e, if_LMS);
        C_init = C;                         %保存初始的训练的均衡器系数，对每一个SNR刚开始迭代的时候赋值
        SER = zeros(26,1);
        count = 1;                          %第几个SNR   
        for SNR = 5:1:30
            C = C_init;                     %赋初值（训练结果）
            flag = 0;
            acc_bad = 0;
            while(acc_bad == 0)             %当没有判决错误情况时，一直循环
                if flag == 1                %如果没有没有判决错误，则增加测试长度
                    test_num = test_num + 200000;
                end
                if test_num > 1500000       %如果长度大于150W，则认为没有错误，跳出循环
                    break
                end
                [acc_bad, test_x, test_u, C, jd] = test_process(b, a, mode,...
                    test_num, SNR,L,C, test_id, deta, mu, mu1, e, if_LMS);
                if acc_bad == 0
                    flag = 1;               %如果没有出现错误，则赋值flag为1
                end
            end
            if acc_bad == 0
                break
            end
            test_num - L + 1;
            SER(count) = acc_bad * 1.0 / (test_num - L + 1);%计算SER
            count = count + 1;
        end
        SNR = 5:1:30;
        %plot_s_u_s_hat(test_x, test_u, jd);


        semilogy(SNR,SER);
        xlabel("SNR");
        ylabel("SER");
        
    end
    legend("4-QAM","16-QAM","64-QAM","256-QAM");
end

%训练函数
function C = train_process(b, a, train_num,SNR,L,C, deta, mu, mu1, e, if_LMS)
    %train data
    train_a = unidrnd(4,[1,train_num]) - 1;
    train_x = pskmod(train_a,4);%QPSK调制
    [train_t, zf] = filter(b,a,train_x);
    train_u = awgn(train_t, SNR, 'measured');
    
    %training
    for i = L : 1 : train_num
        x = train_u(i:-1:i-L+1);
        ak = train_x(i-deta);
        yk = conj(C)*x.';
        ek = ak - yk;
        if if_LMS == 1
            delt = mu1 * conj(ek) * x;
        else
            delt = mu * conj(ek) * x / (e + sum(abs(x).^2));
        end
        C = C + delt;
    end
end

%判决函数
function ak = judge(yk, id)
    dis = abs(yk - id);
    [v, ind] = min(dis);
    ak = id(ind);
end

%计算该模式所有可能的符号
function id = get_id(x)
    temp = [];
    for i = 1:1:length(x)
        if ismember(x(i), temp)
            continue
        else
            temp = [temp, x(i)];
        end
    end
    id = temp;
end

%测试模式
function [acc_bad, test_x, test_u,C, jd] = test_process(b,a, mode,...
    test_num,SNR,L,C, test_id, deta, mu, mu1, e,if_LMS)
    acc_bad_ = 0;
    test_x = unidrnd(mode,[1,test_num]) - 1;
    test_x = qammod(test_x, mode);
    [test_t, zf] = filter(b,a,test_x);
    %test_t = conv(test_x,[0.5,1,1.2,-1]);
    test_u = awgn(test_t, SNR, 'measured');

    jd = zeros(test_num,1);
    for i = L:1:test_num
        x = test_u(i:-1:i-L+1);
        yk = conj(C)*x.';
        jd(i) = yk;
        ak = judge(jd(i), test_id);
        if ak ~= test_x(i-deta)
            acc_bad_ = acc_bad_ + 1;
        end
        ek = ak - yk;
        if if_LMS == 1
            delt = mu1 * conj(ek) * x;
        else
            delt = mu * conj(ek) * x / (e + sum(abs(x).^2));
            C = C + delt;
        end
        
    end
    acc_bad = acc_bad_;
end

%绘制s(i),u(i),s_hat(i)的散点图
function plot_s_u_s_hat(s, u, s_hat)
%scatter the s(i)
scatterplot(s);
title("S(i)");

%scatter the u(i)
scatterplot(u);
title("U(i)");

%scatter the s_hat(i)
scatterplot(s_hat);
title("S_{hat}(i-delta)");
end

%调用get_id函数，得到test的所有可能符号
function test_id = get_test_id(mode)
    id_ = unidrnd(mode,[1,10000]) - 1;
    id_ = qammod(id_, mode);
    test_id = get_id(id_);
end


