import matplotlib.pyplot as plt
import json, math
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def visualizeResults(dirs,labels,markers):
    out = []
    fig, ax = plt.subplots()
    plt.xlabel("Privacy Metric")
    plt.ylabel("Utility Metric")
    for i,d in enumerate(dirs) :
        fName = os.path.join(d,"results.json")
        with open(fName) as fd:
            data = json.load(fd)

        idx,utility,privacy, _w_u, _w_pr = getMetrics(data,"blur")
        idx_p,utility_p,privacy_p, _w_u, _w_pr = getMetrics(data,"pix")
        segm_u, segm_pr = getSegm(data) 
        # idx.append("wireframe")
        # utility.append(_w_u)
        # privacy.append(_w_pr)
        ax.plot(utility,privacy,"+-",label="Blur",linewidth=2,markersize=7)
        ax.plot(utility_p,privacy_p,"*-",label="Pixelated",linewidth=2,markersize=7)
        ax.plot(_w_pr,_w_u,"D",label="wireframe",color='r')
        ax.plot(segm_pr, segm_u,"8",label="segmentation",color="b")
        ax.axhline(y=_w_u,linestyle="--",alpha=0.2)
        ax.axvline(x=_w_pr,linestyle="--",alpha=0.2)
        # ax.plot(pur,markers[i],label=labels[i],linewidth=2,markersize=7)
        labelsToAdd = [2,4,6,16,20,30,50,90,"wireframe"]
        for i, txt in enumerate(idx):
            if txt in labelsToAdd :
                ax.annotate(txt, (utility[i]+0.002, privacy[i]+0.002))
                ax.annotate(txt, (utility_p[i]-0.01, privacy_p[i]-0.03))       
    plt.legend()
    plt.xlim(xmin=-0.1)
    plt.ylim(ymin=0)
    plt.rcParams['figure.dpi'] = 1000
    plt.rcParams['savefig.dpi'] = 1000
    plt.grid()
    plt.show()
    # fig.savefig('pix.eps', format='eps', dpi=1000)

def getPUR(p,u):
    try :
        _pur = p/u
        # if _pur < 35 :
        #     return _pur
        # else :
        #     return 
        return mapToOne(_pur)
    except ZeroDivisionError:
        return mapToOne(math.inf)

def mapToOne(x):
  return 1 - math.exp(-x)

def visualizeResults_PUR(dirs,labels,markers):
    out = []
    fig, ax = plt.subplots()
    plt.xlabel("Kernel Size")
    plt.ylabel("PUR")

    for i,d in enumerate(dirs) :
        fName = os.path.join(d,"results.json")
        with open(fName) as fd:
            data = json.load(fd)

        idx,utility,privacy, _w_u, _w_pr  = getMetrics(data,"blur")
        idx_p,utility_p,privacy_p, _w_u, _w_pr  = getMetrics(data,"pix")

        pur = [getPUR(privacy[x],utility[x]) for x in range(len(privacy))]
        pur_p = [getPUR(privacy_p[x],utility_p[x]) for x in range(len(privacy_p))]
        print(pur)
        print(pur_p)
        w_pur = getPUR(_w_pr,_w_u)
        print(len(idx), len(pur), len(pur_p))
        # ax.plot(privacy,utility,markers[i],label=labels[i],linewidth=2,markersize=7)
        ax.plot(idx,pur,markers[i],label=labels[i]+" blur",linewidth=2,markersize=7)
        ax.plot(idx_p,pur_p,markers[i],label=labels[i]+" pix",linewidth=2,markersize=7)
        ax.axhline(y=w_pur,color="red",linestyle="--",label="wireframe")
        # labelsToAdd = [2,4,6,8]
        # for i, txt in enumerate(idx):
        #     if txt in labelsToAdd :
        #         ax.annotate(txt, (idx[i]+0.002, pur[i]+0.002))        
    plt.legend()

    # ax.fill_between(idx[:maxValues],pur[:maxValues])
    # plt.xlim(xmin=0,xmax=100)
    # plt.ylim(ymin=0,ymax=1)
    plt.rcParams['figure.dpi'] = 1000
    plt.rcParams['savefig.dpi'] = 1000
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    plt.grid()
    plt.show()

def getMetrics(_data,k) :
    _final = []
    for i in _data.keys() :
        if i != "wireframe" and i != "segmentation" :
            if type(_data[i][k]["personDetector"]) == list :
                try :
                    _pd_ap = 0.5*(_data[i][k]["personDetector"][4] + _data[i][k]["personDetector"][5])
                    _pd_ar = 0.5*(_data[i][k]["personDetector"][-2] + _data[i][k]["personDetector"][-1])
                    _pd_f1 = (2.0*(_pd_ap*_pd_ar))/(_pd_ap+_pd_ar)
                    _kd_ap = 0.5*(_data[i][k]["keypointDetector"][3] + _data[i][k]["keypointDetector"][4])
                    _kd_ar = 0.5*(_data[i][k]["keypointDetector"][-2]+ _data[i][k]["keypointDetector"][-1])
                    _kd_f1 = 2*(_kd_ap*_kd_ar)/(_kd_ap+_kd_ar)
                    # print(F"{i} => _pd_f1 : {_pd_f1}     _kd_f1 : {_kd_f1}")
                    _final.append({
                        "idx" : int(i),
                        "utility" : 0.5*(_pd_f1 + _kd_f1),
                        "privacy" : 1 - _data[i][k]["similarityIndex"]
                    })
                except ZeroDivisionError :
                    print(F"unable to capture for {i}")
                    print(_data[i][k]["personDetector"])
                    # print(_data[i][k]["personDetector"][5],_data[i][k]["personDetector"][-1])
                    _final.append({
                        "idx" : int(i),
                        "utility" : 0,
                        "privacy" : 1 - _data[i][k]["similarityIndex"]
                    })
        elif i == "wireframe" :
            _wp_ap = 0.5*(_data[i]["personDetector"][4]+_data[i]["personDetector"][5])
            _wp_ar = 0.5*(_data[i]["personDetector"][-2] + _data[i]["personDetector"][-1])
            _wp_f1 = (2.0*(_wp_ap*_wp_ar))/(_wp_ap+_wp_ar)
            _wk_ap = 0.5*(_data[i]["keypointDetector"][3]+_data[i]["keypointDetector"][4])
            _wk_ar = 0.5*(_data[i]["keypointDetector"][-2]+_data[i]["keypointDetector"][-1])
            _wk_f1 = (2.0*(_wk_ap*_wk_ar))/(_wk_ap+_wk_ar)
            _w_pr = _data[i]["similarityIndex"]

            _w_u = 0.5*(_wp_f1+_wk_f1)

            print(F"{i} => _wp_f1 : {_wp_f1}     _wk_f1 : {_wk_f1}    si: {_w_pr}  _w_u {_w_u}")
    _final = sorted(_final,key=lambda x:x["idx"], reverse=False)
    idx = [x["idx"] for x in _final ]
    utility = [x["utility"] for x in _final]
    privacy = [x["privacy"] for x in _final]
    return idx,utility,privacy, _w_u, _w_pr

def getSegm(_data):
    _wp_ap = 0.5*(_data["segmentation"]["personDetector"][4]+_data["segmentation"]["personDetector"][5])
    _wp_ar = 0.5*(_data["segmentation"]["personDetector"][-2] + _data["segmentation"]["personDetector"][-1])
    _wp_f1 = (2.0*(_wp_ap*_wp_ar))/(_wp_ap+_wp_ar)
    _wk_ap = 0.5*(_data["segmentation"]["keypointDetector"][3]+_data["segmentation"]["keypointDetector"][4])
    _wk_ar = 0.5*(_data["segmentation"]["keypointDetector"][-2]+_data["segmentation"]["keypointDetector"][-1])
    _wk_f1 = (2.0*(_wk_ap*_wk_ar))/(_wk_ap+_wk_ar)
    _w_pr = _data["segmentation"]["similarityIndex"]
    _w_u = 0.5*(_wp_f1+_wk_f1)

    print(F" Segmentation _wp_f1 : {_wp_f1}    _wk_f1 : {_wk_f1}    si: {_w_pr}  _w_u {_w_u}")

    return _w_u, _w_pr

if __name__ == "__main__" :
    dirs = ["tmp_mot_16_08","tmp_pevid_1_2"]
    dirs = ["tmp_mot_16_08"]
    labels = ["MOT_16", "PeVID"]
    markers = ["-*","-o"]
    visualizeResults(dirs,labels,markers)
    # visualizeResults_PUR(dirs,labels,markers)