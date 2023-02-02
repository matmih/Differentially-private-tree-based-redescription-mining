/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesFinal;

import dptressexpm.twotreesFinal.*;
import dptressexpm.twotreesFinal.*;

/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesFinal package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
 */
public class RedescriptionScore {
    
    private int maxSupport, minSupport, minJaccard;
    
    public RedescriptionScore(){
            maxSupport = 0; minSupport = 0; minJaccard = 0;
    }
    
    public RedescriptionScore(int maxSupp, int minSupp, int minJ){
            maxSupport = maxSupp;
            minSupport = minSupp;
            minJaccard = minJ;
    }
    
    public void setParams(int maxSupp, int minSupp, int minJ){
            maxSupport = maxSupp;
            minSupport = minSupp;
            minJaccard = minJ;
    }
    
    public double computeScore(Redescription r, int minSupport, int maxSupport, double minJaccard, double maxPval){
        double tmp = 0.0;
        
        //if(r.support<minSupport || r.support>maxSupport || r.Jaccard<minJaccard || r.pval>maxPval)
        //    return tmp;
            
        if(r.support>maxSupport)
            return tmp;
        
        tmp = (1-r.pval)*(r.Jaccard)/**r.mn*/;
        
        return tmp;
    }
}
