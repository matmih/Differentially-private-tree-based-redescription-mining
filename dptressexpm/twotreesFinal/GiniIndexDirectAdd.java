package dptressexpm.twotreesFinal;

import dptressexpm.twotreesFinal.*;
import dptressexpm.twotreesFinal.*;
import dptressexpm.twotreesFinal.TwoTrees.Node;
import java.io.Serializable;

/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesFinal package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
 */

public class GiniIndexDirectAdd implements Serializable{
	
	private static final long serialVersionUID = 1L;             

        public double scoreNormalized(Node node, int numEntities){
                
		double finalscore = 0;
	
		if(node==null)
			return 0;
		if(node.data.numInstances() <= 0)                 
			return 0;
		
		if(node.isLeaf == true)                      
		{
			finalscore = leafScore(node);
			return finalscore;
		}
		else{
			for(Node child : node.children)                   
			{
				double childscore = scoreNormalized(child,numEntities);					
				if(child.isLeaf==true){
					finalscore += (((double)child.data.numInstances())/(double)numEntities) * childscore; 
				}
				else{
					finalscore += childscore;
				}
			}
			
			return finalscore;                               
		}	
	}
        
        
	public double score(Node node){
		
		double finalscore = 0;
	
		if(node==null)
			return 0;
		if(node.data.numInstances() <= 0)                 
			return 0;
		
		if(node.isLeaf == true)                      
		{
			finalscore = leafScore(node);
			return finalscore;
		}
		else{
			for(Node child : node.children)                   
			{
				double childscore = score(child);					
				if(child.isLeaf==true){
					finalscore -= (double)child.data.numInstances() * childscore; 
				}
				else{
					finalscore += childscore;
				}
			}
			
			return finalscore;                               
		}	
	}
        
        
        public double score(Node rootFirst, Node rootSecond){
            double finalscore = 0.0;
            
            double s1 = scoreNormalized(rootFirst,rootFirst.data.numInstances());
            double s2 = scoreNormalized(rootSecond,rootSecond.data.numInstances());
            double prod = s1*s2;
            
            //finalscore = (scoreNormalized(rootFirst,rootFirst.data.numInstances())+scoreNormalized(rootFirst,rootFirst.data.numInstances())*scoreNormalized(rootSecond,rootSecond.data.numInstances()))/2.0;
            finalscore = (s1+prod)/2.0;
            System.out.println(finalscore);
            return finalscore;
        }

	private double leafScore(Node node){                   
		
		double leafscore = 0;
		int numInst = node.data.numInstances();     
			
		if(numInst == 0){                         
			return 0;
		}
			
		double leafcount[] = node.count;           
		for(int i=0; i< leafcount.length; ++i){     
			if( leafcount[i] > 0){
				double score = Math.pow(leafcount[i]/numInst, 2);
				leafscore += score;                            
			}
		}
		return leafscore;                                  
	}
	
}
