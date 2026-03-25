clc; clear; close all;
fprintf('##############################################\n');
fprintf('##  PQ CLASSIFIER — SIMULINK LIVE DEMO     ##\n');
fprintf('##############################################\n\n');

fs=12800; f0=50; Ts=1/fs; T_sim=0.2;
class_names_list={'Normal','Sag','Swell','Interruption','Harmonics','Transient','Flicker','Notching'};
signal_eqs={
    @(t) sin(2*pi*50*t), ...
    @(t) (1-0.5*(t>=0.04 & t<=0.12)).*sin(2*pi*50*t), ...
    @(t) (1+0.4*(t>=0.04 & t<=0.12)).*sin(2*pi*50*t), ...
    @(t) (1-0.95*(t>=0.05 & t<=0.12)).*sin(2*pi*50*t), ...
    @(t) 0.8*sin(2*pi*50*t)+0.15*sin(6*pi*50*t)+0.07*sin(10*pi*50*t), ...
    @(t) sin(2*pi*50*t)+0.5*exp(-(t-0.08)/0.004).*sin(2*pi*400*(t-0.08)).*(t>=0.08), ...
    @(t) (1+0.1*sin(2*pi*10*t)).*sin(2*pi*50*t), ...
    @(t) sin(2*pi*50*t).*(1-0.7*(mod(t,0.02)<0.001))};

fprintf('Select PQ Event Class:\n');
for k=1:8, fprintf('  %d = %s\n',k,class_names_list{k}); end
class_choice=input('\nEnter class number (1-8): ');
if isempty(class_choice)||class_choice<1||class_choice>8, class_choice=2; end
chosen_class=class_names_list{class_choice};
fprintf('\nSimulating: %s\n\n',chosen_class);

t_vec=(0:round(T_sim/Ts)-1)*Ts;
x_raw=signal_eqs{class_choice}(t_vec);
x_noisy=awgn(x_raw,35,'measured');
fprintf('Signal generated: %d samples\n',length(x_noisy));

peak=max(abs(x_noisy));
if peak>0, x_norm=x_noisy/peak; else, x_norm=x_noisy; end
Wn=5/(fs/2); [b_hp,a_hp]=butter(2,Wn,'high');
x_proc=filtfilt(b_hp,a_hp,x_norm);
fprintf('Preprocessed.\n');

[C,L]=wavedec(x_proc,5,'db4');
feat_dwt=zeros(1,36); feat_idx=1;
for level=1:5
    d=detcoef(C,L,level); c2=d.^2; c2nz=c2(c2>0);
    if isempty(c2nz), ent=0; else, ent=-sum(c2nz.*log(c2nz)); end
    feat_dwt(feat_idx:feat_idx+5)=[sum(d.^2),mean(d),std(d),skewness(d),kurtosis(d),ent];
    feat_idx=feat_idx+6;
end
a5=appcoef(C,L,'db4',5); c2=a5.^2; c2nz=c2(c2>0);
if isempty(c2nz), ent=0; else, ent=-sum(c2nz.*log(c2nz)); end
feat_dwt(feat_idx:feat_idx+5)=[sum(a5.^2),mean(a5),std(a5),skewness(a5),kurtosis(a5),ent];

rms_val=sqrt(mean(x_proc.^2)); peak_val=max(abs(x_proc));
if rms_val>0, crest=peak_val/rms_val; else, crest=0; end
N_fft=length(x_proc); X_fft=abs(fft(x_proc)/N_fft);
freqs=(0:N_fft-1)*fs/N_fft; [~,fi]=min(abs(freqs-f0)); V1=X_fft(fi);
hpow=0;
for h=2:7
    [~,hi]=min(abs(freqs-h*f0));
    if hi<=N_fft/2, hpow=hpow+X_fft(hi)^2; end
end
if V1>0, thd=(sqrt(hpow)/V1)*100; else, thd=0; end
zcr=sum(abs(diff(sign(x_proc))))/2/(length(x_proc)/fs);
ma=mean(abs(x_proc)); if ma>0, formf=rms_val/ma; else, formf=0; end

feat_vector=[feat_dwt,rms_val,peak_val,crest,thd,zcr,formf];
load('../Data/trained_models.mat');
n_tr=size(X_test,2);
if length(feat_vector)>n_tr, feat_vector=feat_vector(1:n_tr);
elseif length(feat_vector)<n_tr, feat_vector(end+1:n_tr)=0; end
fprintf('Feature vector: 1 x %d\n',length(feat_vector));

model_names={'SVM','Random Forest','Boosted Trees','Neural Network'};
results=cell(4,1);
pred_s=predict(mdl_svm,feat_vector); results{1}=char(pred_s);
pred_r=predict(mdl_rf,feat_vector);
if iscell(pred_r), results{2}=pred_r{1}; else, results{2}=char(pred_r); end
pred_b=predict(mdl_boost,feat_vector); results{3}=char(pred_b);
pred_n=classify(mdl_nn,feat_vector); results{4}=char(pred_n);

vote_count=zeros(1,8);
for m=1:4
    for c=1:8
        if strcmpi(results{m},class_names_list{c}), vote_count(c)=vote_count(c)+1; end
    end
end
[max_votes,winner_idx]=max(vote_count);
final_class=class_names_list{winner_idx};
is_correct=strcmpi(final_class,chosen_class);

bar_colours=[0.12 0.31 0.63;0.11 0.45 0.18;0.85 0.33 0.00;0.40 0.08 0.55];
figure('Name','PQ Demo','NumberTitle','off','Position',[50,50,1300,750],'Color',[0.06 0.06 0.12]);

ax1=subplot(3,4,[1 2 3]);
plot(t_vec*1000,x_proc,'Color','#00CFFF','LineWidth',1.1);
set(ax1,'Color',[0.08 0.08 0.18],'XColor','w','YColor','w');
xlabel('Time (ms)','Color','w'); ylabel('Amplitude (pu)','Color','w');
title(sprintf('Signal: %s (SNR=35dB)',chosen_class),'Color','w','FontSize',12,'FontWeight','bold');
grid on;

ax2=subplot(3,4,4);
dwt_e=zeros(1,6);
for lv=1:5, d=detcoef(C,L,lv); dwt_e(lv)=sum(d.^2); end
a5p=appcoef(C,L,'db4',5); dwt_e(6)=sum(a5p.^2);
bclr=[0.8 0.1 0.1;0.9 0.5 0;0.9 0.9 0;0.1 0.7 0.1;0.1 0.5 0.9;0.6 0.1 0.8];
bh=bar(dwt_e,'FaceColor','flat');
for i=1:6, bh.CData(i,:)=bclr(i,:); end
set(ax2,'Color',[0.08 0.08 0.18],'XColor','w','YColor','w');
xticklabels({'D1','D2','D3','D4','D5','A5'});
title('Sub-band Energy','Color','w','FontSize',10,'FontWeight','bold'); grid on;

for m=1:4
    ax=subplot(3,4,4+m);
    if strcmpi(results{m},final_class), bg=[0.04 0.22 0.04]; tc='#00FF80';
    else, bg=[0.22 0.10 0.00]; tc='#FFA040'; end
    set(ax,'Color',bg); axis off;
    text(0.5,0.85,model_names{m},'Color','w','FontSize',10,'FontWeight','bold','HorizontalAlignment','center','Units','normalized');
    text(0.5,0.52,results{m},'Color',tc,'FontSize',15,'FontWeight','bold','HorizontalAlignment','center','Units','normalized');
    text(0.5,0.20,sprintf('Acc:%.1f%%',accuracies(m)),'Color',[0.7 0.7 0.7],'FontSize',9,'HorizontalAlignment','center','Units','normalized');
end

ax_r=subplot(3,4,8);
if is_correct, pb=[0.0 0.3 0.0]; ss='CORRECT'; sc='#00FF80';
else, pb=[0.3 0.08 0.0]; ss=['TRUE: ' chosen_class]; sc='#FF6040'; end
set(ax_r,'Color',pb); axis off;
text(0.5,0.85,'CONSENSUS','Color',[0.8 0.8 0.8],'FontSize',9,'HorizontalAlignment','center','Units','normalized');
text(0.5,0.60,final_class,'Color','#00FF80','FontSize',14,'FontWeight','bold','HorizontalAlignment','center','Units','normalized');
text(0.5,0.35,ss,'Color',sc,'FontSize',10,'FontWeight','bold','HorizontalAlignment','center','Units','normalized');
text(0.5,0.12,sprintf('%d/4 agree',max_votes),'Color',[0.8 0.8 0.8],'FontSize',9,'HorizontalAlignment','center','Units','normalized');

ax_v=subplot(3,4,[9 10 11 12]);
bv=bar(vote_count,'FaceColor','flat');
for c=1:8
    if c==winner_idx, bv.CData(c,:)=[0.0 0.85 0.3];
    else, bv.CData(c,:)=[0.25 0.25 0.4]; end
end
set(ax_v,'Color',[0.08 0.08 0.18],'XColor','w','YColor','w');
xticks(1:8); xticklabels(class_names_list); xtickangle(20);
ylabel('Votes','Color','w'); ylim([0 4.5]); yticks(0:4); grid on;
title('Model Voting','Color','w','FontSize',11,'FontWeight','bold');
for c=1:8
    if vote_count(c)>0
        text(c,vote_count(c)+0.15,num2str(vote_count(c)),'Color','w','FontSize',12,'FontWeight','bold','HorizontalAlignment','center');
    end
end

if is_correct
    btxt=sprintf('CLASSIFIED AS: %s   CORRECT   (%d/4 models agree)',final_class,max_votes);
    bbg=[0.0 0.28 0.0];
else
    btxt=sprintf('CLASSIFIED AS: %s   (True: %s)',final_class,chosen_class);
    bbg=[0.28 0.08 0.0];
end
annotation('textbox',[0 0.93 1 0.07],'String',btxt,'Color','white','FontSize',13,'FontWeight','bold','HorizontalAlignment','center','BackgroundColor',bbg,'EdgeColor','#00BFFF','LineWidth',2);

if ~exist('../Figures','dir'), mkdir('../Figures'); end
fname=sprintf('../Figures/Demo_%s_Result.png',chosen_class);
saveas(gcf,fname);

fprintf('\n##############################################\n');
fprintf('  True class    : %s\n',chosen_class);
fprintf('  Classified as : %s\n',final_class);
if is_correct, fprintf('  Correct?      : YES\n'); else, fprintf('  Correct?      : NO\n'); end
fprintf('  Models agree  : %d/4\n',max_votes);
fprintf('\n  Predictions:\n');
for m=1:4
    if strcmpi(results{m},chosen_class), v='Correct'; else, v='Wrong'; end
    fprintf('    %-15s: %-15s [%s]\n',model_names{m},results{m},v);
end
fprintf('##############################################\n');