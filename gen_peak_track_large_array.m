function [fx,tx,pv,amp]=gen_peak_track_large_array(x, fs)

addpath('voicebox')
    
%     s = x(1:end-mod(length(x),8));

%     [fx1,txs{1},pv1,fv1]=fxpefac(s(1:end/8),fs);
%     [fx2,txs{2},pv2,fv2]=fxpefac(s((end/8)+1:(end/8)*2),fs);
%     [fx3,txs{3},pv3,fv3]=fxpefac(s(((end/8)*2)+1:(end/8)*3),fs);
%     [fx4,txs{4},pv4,fv4]=fxpefac(s(((end/8)*3)+1:(end/8)*4),fs);
%     [fx5,txs{5},pv5,fv5]=fxpefac(s(((end/8)*4)+1:(end/8)*5),fs);
%     [fx6,txs{6},pv6,fv6]=fxpefac(s(((end/8)*5)+1:(end/8)*6),fs);
%     [fx7,txs{7},pv7,fv7]=fxpefac(s(((end/8)*6)+1:(end/8)*7),fs);
%     [fx8,txs{8},pv8,fv8]=fxpefac(s(((end/8)*7)+1:end),fs);

%     fv = [fv1; fv2; fv3; fv4; fv5; fv6; fv7; fv8];
%     pv = [pv1; pv2; pv3; pv4; pv5; pv6; pv7; pv8];

%     amp = [fv1.amp; fv2.amp; fv3.amp; fv4.amp; fv5.amp; fv6.amp; fv7.amp; fv8.amp];
%     fx = [fv1.ff; fv2.ff; fv3.ff; fv4.ff; fv5.ff; fv6.ff; fv7.ff; fv8.ff];

%     tx = txs{1};
%     for i = 2:8
%         last = tx(end);
%         tx = [tx; last+txs{i}];
%     end
    
     
    [fx,tx,pv,fv]=fxpefac_peak_ah(x,fs,0.01);
    amp = fv.amp;
    fx = fv.ff;
end
