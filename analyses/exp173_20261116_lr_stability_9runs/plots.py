"""Standalone scientific figures from audited numerical tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from analyze import ROOT, META, ORDER, GROUPS

FIG = ROOT/'report/figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Noto Sans SC','axes.unicode_minus':False,'font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'medium',
    'axes.edgecolor':'#bac2ca','axes.labelcolor':'#35434d','text.color':'#26333d',
    'xtick.color':'#53616e','ytick.color':'#53616e','figure.facecolor':'white',
    'savefig.facecolor':'white','svg.fonttype':'none'})
D = pd.read_csv(ROOT/'data/validation_metrics.csv')
S = pd.read_csv(ROOT/'data/stability_summary.csv')
N = pd.read_csv(ROOT/'data/native_validation_metrics.csv')
L = pd.read_csv(ROOT/'data/learning_rates.csv')


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=155,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def format_axis(ax):
    ax.grid(alpha=.2,color='#9aabb9',linewidth=.6)
    ax.set_axisbelow(True)


def curve(ax,n,g,k,**kwargs):
    r=D[(D.experiment==f'exp{n:03d}')&(D.group==g)&(D.metric==k)].sort_values('epoch')
    if not len(r):return
    meta=META[n]
    ax.plot(r.epoch,r.value,color=meta['color'],label=meta['label'],**kwargs)
    return r


def overview():
    fig,axes=plt.subplots(7,2,figsize=(13,15),sharex=True,sharey='col')
    for row,n in enumerate(ORDER):
        meta=META[n]
        for col,g in enumerate(['volume','ilm_opl_max']):
            ax=axes[row,col];r=curve(ax,n,g,'ssim',linewidth=1.7,marker='o',markersize=2.8)
            ax.axvspan(meta['epochs']/2,meta['epochs'],color=meta['color'],alpha=.055)
            best=r[r.epoch==meta['best']]
            if len(best):ax.scatter(best.epoch,best.value,s=75,marker='*',color=meta['color'],edgecolors='white',linewidths=.5,zorder=5)
            if n in [59,154,155]:ax.axvline(200,color='#82919c',ls=':',lw=.9)
            ax.set_xlim(0,303);ax.set_ylim((0,1) if col==0 else (.15,.55));format_axis(ax)
            ax.set_title(meta['label']+f'  |  每{meta["every"]}轮验证',fontsize=10,loc='left')
            ax.set_ylabel('SSIM')
        if row==0:
            axes[row,0].text(.5,1.34,'完整3D体积',transform=axes[row,0].transAxes,ha='center',fontsize=14,weight='bold')
            axes[row,1].text(.5,1.34,'ILM–OPL 最大投影',transform=axes[row,1].transAxes,ha='center',fontsize=14,weight='bold')
    for ax in axes[-1]:ax.set_xlabel('训练轮次（epoch）')
    fig.suptitle('同一实验中，整卷稳定性与投影稳定性可以不同',fontsize=18,x=.06,ha='left',y=1.005)
    fig.text(.06,.003,'固定10位验证患者；全分辨率256×304×304；浅色区为各自后半程。星号为原准则best，XOCT E213无同期整卷记录。\n各列纵轴统一；虚线为XOCT系列开始降低学习率的E200。不同采样频率限制横向比较；细线仅连接已测点。',fontsize=9,color='#5e6e7b')
    fig.tight_layout(rect=[0,.045,1,.98],h_pad=1.25)
    save(fig,'01_stability_overview')


def learning_native():
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for n in [18,148,163,164]:
        r=L[L.experiment==f'exp{n:03d}'];axes[0,0].plot(r.epoch,r.learning_rate_used,label=META[n]['label'],color=META[n]['color'],lw=1.7,ls=['-','--',':','-.'][[18,148,163,164].index(n)])
    axes[0,0].set_title('四个FM实验：全程固定2e-4',loc='left')
    for n in [59,154,155,35,40]:
        r=L[L.experiment==f'exp{n:03d}'];axes[0,1].plot(r.epoch,r.learning_rate_used,label=META[n]['label'],color=META[n]['color'],lw=1.5,ls='--' if n in [154,155] else '-')
    axes[0,1].set_title('TransPro与XOCT：按实际更新时序核对',loc='left')
    for ax in axes[0]:
        ax.set_ylabel('本轮实际学习率');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1e-4:g}e-4'));ax.legend(fontsize=7)
        ax.set_ylim(-.000005,.00022)
    r=N[(N.experiment=='exp035')&(N.metric=='mae')];axes[1,0].plot(r.epoch,r.value,color=META[35]['color'],lw=1.1)
    axes[1,0].set_title('TransPro exp035：原随机256³裁块MAE',loc='left');axes[1,0].set_ylabel('MAE（[-1,1]尺度）')
    axes[1,0].axhspan(.023479321,.025587179,alpha=.08,color=META[35]['color'])
    r=N[(N.experiment=='exp040')&(N.group=='native_bscan2d')&(N.metric=='mae')];axes[1,1].plot(r.epoch,r.value,color=META[40]['color'],lw=1.1)
    axes[1,1].set_title('TransPro exp040（补充）：原整卷B-scan MAE',loc='left');axes[1,1].set_ylabel('MAE（原GT min–max尺度）');axes[1,1].axvline(101,ls=':',color='#687681')
    r=N[(N.experiment=='exp059')&(N.metric=='score')];axes[2,0].plot(r.epoch,r.value,color=META[59]['color'],lw=1.1)
    axes[2,0].set_title('XOCT：作者中心裁块三投影SSIM均值',loc='left');axes[2,0].set_ylabel('原投影SSIM均值');axes[2,0].axvline(200,ls=':',color='#687681')
    axes[2,0].scatter([213],[float(r[r.epoch==213].value.iloc[0])],color=META[59]['color'],marker='*',s=90,zorder=5)
    r=N[(N.experiment=='exp040')&(N.group=='native_bscan2d')&(N.metric=='ssim')];axes[2,1].plot(r.epoch,r.value,color=META[40]['color'],lw=1.1)
    axes[2,1].set_title('TransPro exp040：原B-scan SSIM（已÷100）',loc='left');axes[2,1].set_ylabel('2D Gaussian11 SSIM');axes[2,1].axvline(101,ls=':',color='#687681')
    for ax in axes.flat:format_axis(ax);ax.set_xlabel('训练轮次（epoch）')
    fig.suptitle('学习率确实不同，但现有运行没有隔离学习率的因果作用',fontsize=17,x=.07,ha='left',y=.995)
    fig.text(.07,.005,'exp035记录字段是“下一轮学习率”：E200结束后才首次降低；E1–E200训练均为2e-4。\n原生指标的对象、归一化和SSIM窗口不同；这几幅图分别用于查看各自趋势，不能据此比较绝对性能。',fontsize=9,color='#5e6e7b')
    fig.tight_layout(rect=[0,.05,1,.97],h_pad=2)
    save(fig,'02_learning_rates_native_metrics')


def metric_panels(groups,name,title):
    fig,axes=plt.subplots(len(groups),3,figsize=(14,3*len(groups)),squeeze=False)
    for row,g in enumerate(groups):
        for col,k in enumerate(['mae','psnr','ssim']):
            ax=axes[row,col]
            for n in ORDER:curve(ax,n,g,k,linewidth=1,marker='o' if n in [154,155] else None,markersize=3,alpha=.9)
            ax.set_title(GROUPS[g]+' · '+k.upper()+(' ↓' if k=='mae' else ' ↑'),fontsize=10,loc='left')
            ax.set_xlabel('epoch');format_axis(ax)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,.001))
    fig.suptitle(title,fontsize=17,x=.06,ha='left',y=1.002)
    fig.tight_layout(rect=[0,.085 if len(groups)==2 else .035,1,.98],h_pad=1.8)
    save(fig,name)


def volatility():
    fig,axes=plt.subplots(2,2,figsize=(12,7.6))
    for row,(nums,window) in enumerate([([18,148,163,164],'latter_half'),([59,154,155],'shared_late_150_to_240')]):
        for col,g in enumerate(['volume','ilm_opl_max']):
            ax=axes[row,col]
            vals=[]
            for n in nums:
                r=S[(S.experiment==f'exp{n:03d}')&(S.group==g)&(S.metric=='ssim')&(S.window==window)].iloc[0]
                vals.append(r.mean_abs_step)
            bars=ax.bar([f'exp{n:03d}' for n in nums],vals,color=[META[n]['color'] for n in nums],width=.6)
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.4f}',ha='center',va='bottom',fontsize=9)
            ax.set_ylim(0,max(vals)*1.25);ax.set_ylabel('相邻验证SSIM平均绝对变化');format_axis(ax)
            ax.set_title(('E105–E200 · 每5轮' if row==0 else '共同E150/180/210/240 · 每30轮')+'\n'+GROUPS[g],loc='left',fontsize=11)
    fig.suptitle('相同验证间隔下比较波动幅度',fontsize=17,x=.07,ha='left',y=1.005)
    fig.text(.07,.003,'每组统计均为已观测相邻点的变化；上、下两行的窗口与间隔不同。下行只有4个点、3次变化，不能检验短时稳定性。',fontsize=9,color='#5e6e7b')
    fig.tight_layout(rect=[0,.04,1,.98],h_pad=2)
    save(fig,'05_matched_cadence_volatility')


if __name__=='__main__':
    overview();learning_native()
    metric_panels(['volume','ilm_opl_max'],'03_core_complete_metrics','完整3D与主ILM–OPL投影：MAE、PSNR、SSIM全套轨迹')
    metric_panels([g for g in GROUPS if g not in ['volume','ilm_opl_max']],'04_supplementary_complete_metrics','逐B-scan与补充投影：已有记录的完整三指标')
    volatility()
    print('Created 5 PNG and 5 SVG figures')
