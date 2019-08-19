# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:13:10 2019

@author: Mateusz

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from scipy.stats import ks_2samp
import sys, os

class classifiereval:
    """
    """
    def __init__(self, params = {}):
        """
        
        One graph is generated (if possible) for each model on all defined sets.  We can compare models bettwen sets.
        Constructor uses predict_proba method (sklearn) to get predictions. This requires each metod has this method.
    
        params - dictionary contains all models for evaluation
        params = {'model_name1' : (model1, {'set1_name' : (x1,y1), 'set2_name' : (x2,y2)}), 'model_name2' : ...}
        params = {}
        
        Usage:
        e.g:
        params['logit'] = (logit_clf, {'train' : (train_prepared, y_train), 'test' : (test_prepared, y_test)})
        params['cbc'] = (cbc_clf, {'train' : (train_prepared, y_train), 'test' : (test_prepared, y_test)})
        model_ev = ClassifierEval(params = params)
        model_ev.lift_chart(chart_type = 'lift', cum = True)
        ...
        
        """
        self.eval_data = {}
        self.__set_plot_settings()
        for model in params.keys():
            _eval_data = {}
            for eval_set in params[model][1].keys():
                try:
                    y_proba = params[model][0].predict_proba(params[model][1][eval_set][0])    
                    y_proba = y_proba[:,1]
                    _eval_data[eval_set] = (y_proba, params[model][1][eval_set][1])             
                except:
                    print('Error: Can''t get prediction: ' + model + eval_set + ' Skipping..')
                    print('Error text: {}'.format(sys.exc_info()[1]))
            self.eval_data[model] = _eval_data
            
    def __set_plot_settings(self):
        """ 
        Plot settings
        """
        plt.style.use('seaborn')
        plt.rcParams["figure.figsize"] = (15,10)
        params = {'axes.labelsize': 20,'axes.titlesize':20,  'legend.fontsize': 15, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
        plt.rcParams.update(params)
                    
            
    def __plot_lift_chart(self, y_proba, y_target, chart_type = 'lift', scale = 100, cum = True, label = ''):
        """ 
        Lift chart algorithm;    
        chart_type: 'lift', 'response', 'captured response'
        scale: number of groups (percentiles)
        cum: cumulative or not
        """
        n = len(y_proba)/scale                                                              
        K = sum(y_target)                                                                   
        k = K/scale                                                                          
        data = pd.DataFrame()
        data['target'] = y_target
        data['score'] = y_proba
        data.sort_values(by = 'score', ascending = False, inplace = True)
        class_var = [round(i//(len(data['target'])/scale)+1) for i in range(len(data['target']))]
        data['class_var'] = class_var
        data = data.drop(['score'], axis = 1)
        data = data.groupby('class_var').sum()
        target_group = data['target']
        target_cum = [sum(target_group[:i+1]) for i in range(len(target_group))]
        data['target_cum'] = target_cum      
        if not cum:          
            response = [value/n*100 for value in list(data['target'])]
            data['response'] = response          
            captured_response = [value/K*100 for value in list(data['target'])]
            data['captured response'] = captured_response          
            lift = [value/k for value in list(data['target'])]
            data['lift'] = lift
            if chart_type == 'response': l_b = [k/n*100 for i in range(scale)]              
            elif chart_type == 'captured response': l_b = [k/K*100 for i in range(scale)]
            else: l_b = [1 for i in range(scale)]
            y_label = chart_type
        
        else:       
            response = [value/(n*j)*100 for value,j in zip(list(data['target_cum']),[i+1 for i in range(scale)])]
            data['response'] = response      
            captured_response = [value/K*100 for value in list(data['target_cum'])]
            data['captured response'] = captured_response      
            lift = [value/(k*j) for value,j in zip(list(data['target_cum']),[i+1 for i in range(scale)])]
            data['lift'] = lift      
            if chart_type == 'response': l_b = [k/n*100 for i in range(scale)]                              
            elif chart_type == 'captured response': l_b = [j*k/K*100 for j in [i+1 for i in range(scale)]]
            else: l_b = [1 for i in range(scale)]
            y_label = 'cumulative ' + chart_type
        
        plt.plot(data[chart_type], linewidth = 2, label = label)
        plt.plot(l_b,'k--')
        if chart_type == 'response' or chart_type == 'captured response':
            plt.axis([0,scale,0,110])
        plt.xlabel('Group')
        plt.ylabel(y_label)
        plt.legend(loc = 'best')
        
    def __plot_roc_curve(self, y_proba, y_target, label):   
        """ 
        ROC chart algorithm;
        """      
        fpr, tpr, thressholds = roc_curve(y_target, y_proba)
        auc = str(round(roc_auc_score(y_target, y_proba), 3))
        plt.plot(fpr, tpr, linewidth = 2, label = 'AUC ' + label + ' : ' + auc)   
        plt.plot([0,1],[0,1],'k--')
        plt.axis([0,1,0,1])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc = 'best')
        
    def __plot_score_histogram(self, score, target = None, density = False, cumulative = False, label = ''):
        """
        Scoring chart algorithm;
        If target (list) then histogram by target;
        Density: if true then percentage scale;
        Cumulative: cumulative or not
        """
        bins = np.linspace(0, 1, 100)
        if target is None:
            plt.hist(score, bins, label = 'Score')
            plt.show()
        else:
            df_to_plot = pd.DataFrame({'score' : score, 'target' : target})        
            # KS Test 
            st, pv = ks_2samp(df_to_plot['score'].where(df_to_plot['target'] == 1).dropna(), df_to_plot['score'].where(df_to_plot['target'] == 0).dropna())                
            if cumulative:
                df_to_plot['score'] = 1 - df_to_plot['score']            
            plt.hist(df_to_plot['score'].where(df_to_plot['target'] == 1).dropna(), bins, alpha = 0.5, label = 'Target = 1', density = density, cumulative = cumulative)
            plt.hist(df_to_plot['score'].where(df_to_plot['target'] == 0).dropna(), bins, alpha = 0.5, label = 'Target = 0', density = density, cumulative = cumulative) 
            plt.legend(loc='upper right')
            plt.title('Histograms by target on ' + label + ' KS = ' + str(round(st, 3)) + ' p-value = ' + str(round(pv, 3)))
            
            
    def __plot_precision_recall_vs_thresshold(self, y_target, y_proba, x_thresshold = False, label = None):
        """ 
           Precision recall algorithm;
           x_thresshold = False then Precision Recall Curve; if True then Precision Recall and Thresshold Curve;
        """ 
            
        precisions, recalls, thresholds = precision_recall_curve(y_target, y_proba)       
        if x_thresshold:           
            plt.plot(thresholds, precisions[:-1], "--", label = "Precision")
            plt.plot(thresholds, recalls[:-1], "-", label = "Recall")
            plt.xlabel("Thresshold")
            plt.legend(loc = "center left")
            plt.ylim([0,1])                      
        else:           
            plt.plot(recalls[:-1], precisions[:-1],"-", label = label)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.ylim([0,1])
            plt.xlim([0,1])
            plt.legend(loc = "best")   
        
    def lift_chart(self, chart_type = 'lift', scale = 100, cum = True, save = True):
        """
        chart_type: 'lift', 'response', 'captured response'
        scale: number of groups (percentiles)
        cum: cumulative or not
        """
        for model in self.eval_data.keys():
            for model_set in self.eval_data[model].keys():
                self.__plot_lift_chart(y_proba = list(self.eval_data[model][model_set][0]), y_target = list(self.eval_data[model][model_set][1]), chart_type = chart_type, scale = scale, cum = cum, label = model_set)
            plt.title('Model: ' + model) 
            if save:
                self.__save_fig(name = model + '-' + chart_type, tight_layout=True, GRAPHICS_DIR = './grp')
            plt.show()
            
    def roc_chart(self, save = True):
        """
        """
        for model in self.eval_data.keys():
            for model_set in self.eval_data[model].keys():
                self.__plot_roc_curve(y_proba = list(self.eval_data[model][model_set][0]), y_target = list(self.eval_data[model][model_set][1]), label = model_set)
            plt.title('Model: ' + model) 
            if save:
                self.__save_fig(name = model + '-roc', tight_layout=True, GRAPHICS_DIR = './grp')
            plt.show()
            
    def score_hist_chart(self, density = True, cumulative = False, save = True):
        """
        If target (list) then histogram by target;
        Density: if true then percentage scale;
        Cumulative: cumulative or not
        """
        for model in self.eval_data.keys():
            for model_set in self.eval_data[model].keys():
                self.__plot_score_histogram(score = list(self.eval_data[model][model_set][0]), target = list(self.eval_data[model][model_set][1]), density = density, cumulative = cumulative, label = model + ' ' + model_set)
                if save:
                    self.__save_fig(name = model + '-' + model_set + '-score-hist', tight_layout=True, GRAPHICS_DIR = './grp')
                plt.show()

    def precision_recall_chart(self, x_thresshold = False, save = True):
        """
        x_thresshold = False then Precision Recall Curve; if True then Precision Recall and Thresshold Curve;
        """
        for model in self.eval_data.keys():
            for model_set in self.eval_data[model].keys():
                self.__plot_precision_recall_vs_thresshold(y_target = list(self.eval_data[model][model_set][1]), y_proba = list(self.eval_data[model][model_set][0]), x_thresshold = x_thresshold, label = model_set)
                if x_thresshold:
                    plt.title('Precison Recall Thresshold : ' + model + ' : ' + model_set)
                    if save:
                        self.__save_fig(name = model + '-' + model_set + '-' + 'PRTCurve' , tight_layout=True, GRAPHICS_DIR = './grp')
                    plt.show()
            if not x_thresshold:
                plt.title('Precision Recall Curve : ' + model)
                if save:
                    self.__save_fig(name = model + 'PRCurve' , tight_layout=True, GRAPHICS_DIR = './grp')
                plt.show()
       
    def __save_fig(self, name, tight_layout=True, GRAPHICS_DIR = './grp'):
        """
        Save fig and create dir if not exists.
        """
        if not os.path.exists(GRAPHICS_DIR):
            os.mkdir(GRAPHICS_DIR)
        path = os.path.join(GRAPHICS_DIR, name + ".png")
        print("Zapisywanie rysunku", name)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format='png', dpi=300)
    
    

        
        
        
        
    
    
    
    