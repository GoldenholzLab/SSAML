import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def showSummary(rwd,bias,covp,numLIST,oldALL,survivalTF): 
    if survivalTF==True:
        useme = 'C-index'
    else:
        useme = 'AUC'

    R = pd.read_csv(rwd,delimiter=',',header=None)
    B = pd.read_csv(bias,delimiter=',',header=None)
    C = pd.read_csv(covp,delimiter=',',header=None)

    R.columns = ['howmany','confint','RDW slope','RWD ' + useme,'RWD CIL']
    numLIST = R['howmany']
    R = R.drop('howmany',axis=1)
    B.columns = ['howmany','confint','BIAS slope','BIAS ' + useme,'BIAS CIL']
    B = B.drop(['howmany','confint'],axis=1)
    C.columns = ['howmany','confint','COVP slope','COVP ' + useme,'COVP CIL']
    C = C.drop(['howmany','confint'],axis=1)
    ALL = pd.concat([R,B,C],axis=1)
    ALL.index = numLIST

    print('RWD goal < 0.5, BIAS goal < 5%, COVP > 95%')
    print(ALL.transpose())

    ALL[ALL['COVP slope']<0.95] = np.nan
    ALL[ALL['COVP ' + useme]<0.95]=np.nan
    ALL[ALL['COVP CIL']<0.95]=np.nan
    ALL = ALL.transpose()
    oldALL[np.isnan(ALL)==0] = ALL[np.isnan(ALL)==0]

    return oldALL
  

if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
        
    combs = [(10,1,0.1), (10,1,0.2), (10,10,0.1),(100,1,0.1)]
    survivalTF = False
    if survivalTF==True:
        useme = 'C-index'
    else:
        useme = 'AUC'
    if os.path.exists('bigDs.pickle'):
        with open('bigDs.pickle', 'rb') as ff:
            bigDs = pickle.load(ff)
    else:
        bigDs = []
        for Nfeat, classratio, flipy in combs:
            simulation_folder = f'/data/interesting_side_projects/SSAML/github/simulation/OUTsimulation_Nfeat{Nfeat}_classratio{classratio}_flipy{flipy}_randomseed2020'

            x = pd.read_csv(os.path.join(simulation_folder, 'conflist.setup'), delimiter=' ', header=None)
            clist=np.array(x.iloc[0,])
            x = pd.read_csv(os.path.join(simulation_folder, "RWD_0.955.txt"), delimiter=',', header=None)
            x.columns = ['howmany','confint','RDW slope','RWD C-index','RWD CIL']
            numLIST = np.array(x.howmany)
            numLIST = numLIST.astype(int)

            temp = np.empty((10,len(numLIST)))
            temp[:] = np.nan

            ALL = pd.DataFrame(temp,index=['confint','RDW slope','RWD ' + useme,'RWD CIL','BIAS slope','BIAS ' + useme,'BIAS CIL','COVP slope','COVP ' + useme,'COVP CIL'],columns=numLIST)
            for confint in reversed(clist):
                ALL = showSummary(
                    os.path.join(simulation_folder, 'RWD' + '_' + str(confint) + '.txt'),
                    os.path.join(simulation_folder, 'BIAS' + '_' + str(confint) + '.txt'),
                    os.path.join(simulation_folder, 'COVP' + '_' + str(confint) + '.txt'),
                    numLIST, ALL, survivalTF)
            print('The frankenstein is...')
            print(ALL)
            
            prefixN = 'smallZ'
            for howmany in numLIST:
                fn = prefixN + str(howmany).zfill(4) + '.csv'
                dat = pd.read_csv(os.path.join(simulation_folder, fn),sep=',',header=None)
                dat.columns =['Slope',useme,'CIL']
                dat['N'] = dat.Slope*0 + howmany
                if howmany == numLIST[0]:
                    bigD = dat
                else:
                    bigD = bigD.append(dat,ignore_index=True)
            bigDs.append(bigD)
            
        with open('bigDs.pickle', 'wb') as ff:
            pickle.dump(bigDs, ff)
        
    plt.close()
    fig = plt.figure(figsize=(13,8))
    
    color = (0.5,0.5,0.5)
    cc = 1
    Ns = [500,1000,1500,2000]
    for i in range(3):
        if i==0:
            y = 'Slope'
            ylim = [0.5, 1.5]
        elif i==1:
            y = useme
            ylim = [0.65, 1]
        elif i==2:
            y = 'CIL'
            ylim = [0.85, 1.15]
        for j in range(4):
            bigDs[j]['N'] = bigDs[j].N.astype(int)
            ax = fig.add_subplot(3,4,cc)
            #sns.boxplot(x="N", y=y, data=bigDs[j],showfliers=False)
            ax.boxplot(
                [bigDs[j][y][bigDs[j].N==n].values for n in Ns],
                positions=np.arange(len(Ns)),
                labels=[str(x) for x in Ns],
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=color),
                medianprops=dict(color='k'),
            )
            ax.set_ylim(ylim)
            ax.yaxis.grid(True)
            if j==0:
                ax.set_ylabel(y)
            else:
                ax.set_yticklabels([])
            if i==2:
                ax.set_xlabel('N')
            else:
                ax.set_xticklabels([])
            if i==0:
                ax.set_title(f'# feature = {combs[j][0]}\npos:neg ratio = 1:{combs[j][1]}\nlabel flipping rate = {int(combs[j][2]*100)}%')
            cc += 1
    
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.18)
    if display_type=='pdf':
        plt.savefig('Zplot_simulation.pdf', dpi=600, bbox_inches='tight', pad_inches=0.01)
    elif display_type=='png':
        plt.savefig('Zplot_simulation.png', bbox_inches='tight', pad_inches=0.01)
    else:
        plt.show()
